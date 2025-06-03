import os
import json
import argparse
import pickle

import torch
from functools import partial
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.amp import GradScaler

from codebase import LiTr, train_one_epoch, SetCriterion, HungarianMatcher, CCPDDataset, LiTrPostProcessor

DEVICE = "cuda"

def save_file(history, path):
    with open(path, 'wb') as file:
        pickle.dump(history, file)

def load_file(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)
    return history

def print_metrics(metrics, epoch, mode):
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Total GT Licenses: {metrics['total_gt_licenses']}")
    print(f"  Total Pred Licenses: {metrics['total_pred_licenses']}")
    print(f"  License Correct: {metrics['license_correct']}")

def mosiac_collate(batch, image_size=(640, 640)):
    targets = []
    images = []
    length = len(batch)
    for i in range(0, length, 4):
        i1, t1 = batch[i]
        i2, t2 = batch[(i+1) % length]
        i3, t3 = batch[(i+2) % length]
        i4, t4 = batch[(i+3) % length]

        image = torch.zeros((3, 1280, 1280))
        image[:, :640, :640] = i1
        image[:, :640, 640:] = i2
        image[:, 640:, :640] = i3
        image[:, 640:, 640:] = i4

        t1["boxes"] = 0.5 * t1["boxes"]

        t2["boxes"] = 0.5 * t2["boxes"]
        for i in range(len(t2["boxes"])):
            t2["boxes"][i, 0] = 0.5 + t2["boxes"][i, 0]

        t3["boxes"] = 0.5 * t3["boxes"]
        for i in range(len(t3["boxes"])):
            t3["boxes"][i, 1] = 0.5 + t3["boxes"][i, 1]

        t4["boxes"] = 0.5 * t4["boxes"]
        for i in range(len(t4["boxes"])):
            t4["boxes"][i, 0:2] = 0.5 + t4["boxes"][i, 0:2]

        target = {}
        for key in t1.keys():
            target[key] = torch.cat([t1[key], t2[key], t3[key], t4[key]])

        targets.append(target)
        images.append(image)

    images = torch.stack(images)
    images = F.interpolate(images, size=image_size, mode='bilinear', align_corners=False)

    return images, targets

def main(cfg, checkpoint_path):
    model = LiTr(
        num_classes = cfg['num_classes'],
        backbone_model = cfg['backbone_model'],
        hidden_dim = cfg['hidden_dim'], 
        nhead = cfg['nhead'], 
        ffn_dim = cfg['ffn_dim'], 
        num_encoder_layers = cfg['num_encoder_layers'], 
        eval_spatial_size = cfg['eval_spatial_size'],
        aux_loss = cfg['aux_loss'],
        num_queries = cfg['num_queries'],
        num_decoder_points = cfg['num_decoder_points'],
        num_denoising = cfg['num_denoising'],
        num_decoder_layers = cfg['num_decoder_layers'],
        dropout = cfg['dropout'],
    )
    model.to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)

    matcher = HungarianMatcher(
        weight_dict = cfg['matcher_weight_dict'],
        num_classes = cfg['num_classes'],
    )

    criterion = SetCriterion(
        matcher = matcher,
        weight_dict = cfg['criterion_weight_dict'],
        losses = cfg['compute_losses'],
        num_classes = cfg['num_classes'],
        share_matched_indices = False,
    )

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone')]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 0.00001},
        {'params': other_params, 'lr': 0.00005}
    ], weight_decay=cfg['weight_decay'])

    scaler = GradScaler()

    history = {"train": []}

    train_set = CCPDDataset(cfg["ccpd_train_dir"], eval_size=cfg['eval_spatial_size'], normalize_mean=cfg['normalize_mean'], normalize_std=cfg['normalize_std'], mode='train')
    print("Total Samples in Train Set :", len(train_set))

    collate_with_size = partial(mosiac_collate, image_size=cfg['eval_spatial_size'])
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=10, persistent_workers=True, collate_fn=collate_with_size, prefetch_factor=10, pin_memory=True)
    output_processor = LiTrPostProcessor(num_classes=cfg['num_classes'], num_queries=cfg['num_queries'])   

    if os.path.exists(f"./saved/{cfg['model_name']}_finetune_checkpoint.pth"):
        history = load_file(f"./saved/{cfg['model_name']}_finetune_history.pkl")
        checkpoint = torch.load(f"./saved/{cfg['model_name']}_finetune_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(cfg['epochs']):
        if len(history['train']) > epoch:
            print_metrics(history['train'][epoch], epoch, "Train")
            print()
            continue

        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, output_processor, DEVICE, max_norm=0.1)
        print_metrics(train_metrics, epoch+1, "Train")

        history["train"].append(train_metrics)
        print()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"./saved/{cfg['model_name']}_finetune_checkpoint.pth")
        save_file(history, f"./saved_new/{cfg['model_name']}_finetune_history.pkl")

    torch.save(model.state_dict(), f"./saved/{cfg['model_name']}_finetuned.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="LiTr Finetuning")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint to load")
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    main(config, checkpoint_path=args.checkpoint)
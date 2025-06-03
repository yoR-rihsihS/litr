import os
import json
import argparse
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import GradScaler

from codebase import LiTr, train_one_epoch, evaluate, SetCriterion, HungarianMatcher, CCPDDataset, collate_fn, LiTrPostProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

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

def main(cfg):
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
        {'params': backbone_params, 'lr': cfg['learning_rate_backbone']},
        {'params': other_params, 'lr': cfg['learning_rate']}
    ], weight_decay=cfg['weight_decay'])

    ms_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["steps"], gamma=cfg['gamma'])
    scaler = GradScaler()

    history = {"train": [], "val": []}

    train_set = CCPDDataset(path=cfg['ccpd_train_dir'], mode="train", eval_size=cfg['eval_spatial_size'], normalize_mean=cfg["normalize_mean"], normalize_std=cfg["normalize_std"])
    print("Total Samples in Train Set:", len(train_set))
    val_set = CCPDDataset(path=cfg['ccpd_val_dir'], mode="eval", eval_size=cfg['eval_spatial_size'], normalize_mean=cfg["normalize_mean"], normalize_std=cfg["normalize_std"])
    print("Total Samples in Validation Set:", len(val_set))

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=7, persistent_workers=True, collate_fn=collate_fn, prefetch_factor=10)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=3, persistent_workers=False, collate_fn=collate_fn, prefetch_factor=10)
    
    output_processor = LiTrPostProcessor(num_classes=cfg['num_classes'], num_queries=cfg['num_queries'])   

    if os.path.exists(f"./saved/{cfg['model_name']}_checkpoint.pth"):
        history = load_file(f"./saved/{cfg['model_name']}_history.pkl")
        checkpoint = torch.load(f"./saved/{cfg['model_name']}_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ms_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(cfg['epochs']):
        if len(history['train']) > epoch:
            print_metrics(history['train'][epoch], epoch, "Train")
            print_metrics(history['val'][epoch], epoch, "Validation")
            print()
            continue

        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, output_processor, DEVICE, max_norm=0.1)
        print_metrics(train_metrics, epoch+1, "Train")
        val_metrics = evaluate(model, criterion, val_loader, output_processor, DEVICE)
        print_metrics(val_metrics, epoch+1, "Validation")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print()
        ms_scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": ms_scheduler.state_dict(),
        }, f"./saved/{cfg['model_name']}_checkpoint.pth")
        save_file(history, f"./saved/{cfg['model_name']}_history.pkl")

        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"./saved/{cfg['model_name']}_{epoch+1}.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="LiTr Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    main(config)
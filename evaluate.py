import json
import argparse
import numpy as np
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from codebase import CCPDDataset, LiTr, LiTrPostProcessor, collate_fn, checking

DEVICE = 'cuda'

def print_metrics(metrics, test_set):
    print(f"{test_set} Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Total GT Licenses: {metrics['total_gt_licenses']}")
    print(f"  Total Pred Licenses: {metrics['total_pred_licenses']}")
    print(f"  License Correct: {metrics['license_correct']}")
    print(f"  Mean Time: {metrics['mean_time']:.4f} seconds")
    print(f"  Std Time: {metrics['std_time']:.4f} seconds")
    print(f"  FPS: {metrics['fps']:.2f}")

def evaluate(model, data_loader):
    total_gt_licenses = 0
    total_pred_licenses = 0
    license_correct = 0
    total_iou = 0
    times = []
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            start_time = time()
            boxes, scores, labels = model(samples)
            end_time = time()
            times.append(end_time - start_time)
            processes_outputs = [{
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }]

            num_gt_licences, num_pred_licences, num_correct, iou = checking(processes_outputs, targets)
            total_gt_licenses += num_gt_licences
            total_pred_licenses += num_pred_licences
            license_correct += num_correct
            total_iou += iou
            total_samples += 1

    precision = license_correct / total_pred_licenses if total_pred_licenses > 0 else 0.0
    recall = license_correct / total_gt_licenses if total_gt_licenses > 0 else 0.0
    mean_iou = total_iou / total_gt_licenses if total_gt_licenses > 0 else 0.0
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = total_samples / np.sum(times)

    metrics = {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "total_gt_licenses": total_gt_licenses,
        "total_pred_licenses": total_pred_licenses,
        "license_correct": license_correct,
        "mean_time": mean_time,
        "std_time": std_time,
        "fps": fps,
    }
     
    return metrics


def main(cfg, model_path):
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
    )

    checkpoint = torch.load(model_path, map_location='cpu') 
    model.load_state_dict(checkpoint)

    postprocessor = LiTrPostProcessor(num_classes=cfg['num_classes'], num_queries = cfg['num_queries'])  

    # This model can only be used for inference and the batch size must be 1
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()
            
        def forward(self, images, top_k=100, score_thresh=0.5):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, top_k=top_k, score_thresh=score_thresh)
            return outputs

    model = Model()
    model.to(DEVICE)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    for subset in ['base', 'blur', 'challenge', 'db', 'fn', 'rotate', 'tilt', 'weather']:
        test_set = CCPDDataset(path='../CCPD2019/new_splits/test_ccpd_'+subset+'.txt', mode="eval", eval_size=cfg['eval_spatial_size'], normalize_mean=cfg["normalize_mean"], normalize_std=cfg["normalize_std"])
        print("Total Samples in given Test Set :", len(test_set))
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True, collate_fn=collate_fn, prefetch_factor=20, pin_memory=True)
    
        for i in range(2): # First iteration is for warming up
            test_metrics = evaluate(model, test_loader)
        print_metrics(test_metrics, subset)
        
    print("Evaluation completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiTr Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    main(config, args.model_path)
import torch
from torch.amp import autocast

from .check import checking

def train_one_epoch(model, criterion, data_loader, optimizer, scaler, postprocessor, device, max_norm=0):
    running_loss = 0
    total_gt_licenses = 0
    total_pred_licenses = 0
    license_correct = 0
    total_iou = 0
    model.train()
    criterion.train()
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast(device_type=device, cache_enabled=True):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
        
        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        num_gt_licences, num_pred_licences, num_correct, iou = checking(postprocessor(outputs), targets)
        total_gt_licenses += num_gt_licences
        total_pred_licenses += num_pred_licences
        license_correct += num_correct
        total_iou += iou

    precision = license_correct / total_pred_licenses if total_pred_licenses > 0 else 0.0
    recall = license_correct / total_gt_licenses if total_gt_licenses > 0 else 0.0
    mean_iou = total_iou / total_gt_licenses if total_gt_licenses > 0 else 0.0

    metrics = {
        "loss": running_loss / len(data_loader),
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "total_gt_licenses": total_gt_licenses,
        "total_pred_licenses": total_pred_licenses,
        "license_correct": license_correct,
    }
     
    return metrics


@torch.no_grad()
def evaluate(model, criterion, data_loader, postprocessor, device):
    running_loss = 0
    total_gt_licenses = 0
    total_pred_licenses = 0
    license_correct = 0
    total_iou = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(device_type=device, cache_enabled=True):
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())

            running_loss += loss.item()

            num_gt_licences, num_pred_licences, num_correct, iou = checking(postprocessor(outputs), targets)
            total_gt_licenses += num_gt_licences
            total_pred_licenses += num_pred_licences
            license_correct += num_correct
            total_iou += iou

    precision = license_correct / total_pred_licenses if total_pred_licenses > 0 else 0.0
    recall = license_correct / total_gt_licenses if total_gt_licenses > 0 else 0.0
    mean_iou = total_iou / total_gt_licenses if total_gt_licenses > 0 else 0.0

    metrics = {
        "loss": running_loss / len(data_loader),
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "total_gt_licenses": total_gt_licenses,
        "total_pred_licenses": total_pred_licenses,
        "license_correct": license_correct,
    }
     
    return metrics
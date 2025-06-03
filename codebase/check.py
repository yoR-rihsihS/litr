import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import box_iou, box_cxcywh_to_xyxy

@torch.no_grad()
def checking(results, targets, iou_threshold=None):
    """ 
    Computes the number of correct matches between predictions and targets.
    The matching is done using the Hungarian algorithm on the pairwise IoU matrix.
    Args:
        - results: list of dicts, such that len(results) == batch_size
        see the output specification of the LiTrPostprocessor
            keys:
                - boxes (Tensor): shape (num_pred_objects, 4)
                - scores (Tensor): shape (num_pred_objects, 1)
                - labels (Tensor): shape (num_pred_objects, 7) contains indices of the predicted characters
        - targets: list of dicts, such that len(targets) == batch_size.
            keys:
                - boxes (Tensor): shape (num_objects, 4)
                - label_0 (Tensor): shape (num_objects)
                - label_1 (Tensor): shape (num_objects)
                ...
                - label_6 (Tensor): shape (num_objects)
    Returns:
        - total_targets (int): total number of ground truth boxes
        - total_preds (int): total number of predicted boxes
        - correct_matches (int): number of correct matches
        - total_iou (float): sum of IoU scores for matched boxes
    """
    total_targets = 0
    total_preds = 0
    correct_matches = 0
    total_iou = 0

    for res, tgt in zip(results, targets):
        # ground truths
        gt_boxes = tgt["boxes"]               # (num_tgt, 4)
        num_tgt = gt_boxes.size(0)
        total_targets += num_tgt

        # preds
        pred_boxes = res["boxes"]             # (num_pred, 4)
        pred_labels = res["labels"]           # (num_pred, 7)
        num_pred = pred_boxes.size(0)
        total_preds += num_pred

        if num_tgt == 0 or num_pred == 0:
            continue  # nothing to match

        # Stack GT labels once per image
        gt_labels = torch.stack([tgt[f"label_{i}"] for i in range(7)], dim=1)  # (num_tgt, 7)

        # Pairwise IoU and Hungarian matching
        ious, unions = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
        cost = -ious.cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            iou = ious[r, c].item()
            if iou_threshold is not None and iou < iou_threshold:
                continue
            total_iou += iou
            if torch.equal(pred_labels[r], gt_labels[c]):
                correct_matches += 1

    return total_targets, total_preds, correct_matches, total_iou
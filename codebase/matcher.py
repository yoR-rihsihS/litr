import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, weight_dict, num_classes):
        """Creates the matcher

        Params:
            weight_dict: This is a dict containing the relative weights of the various terms in the matching cost.
                - cost_labels: This is the relative weight of the classification error in the matching cost
                - cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
                - cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            num_classes: dict containing as key the names of object ca and as values their number of object categories, omitting the special no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.cost_labels = weight_dict['cost_labels']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        assert self.cost_labels != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                - "pred_label_0": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits for label_0
                - "pred_label_1": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits for label_1
                ...
                - "pred_label_6": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits for label_6
                - "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                - "label_0": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class label_0
                - "label_1": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class label_1
                ...
                - "label_6": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class label_6
                - "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_label_0"].shape[:2]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the classification cost
        cost_labels = torch.zeros_like(cost_bbox)
        for class_name, num in self.num_classes.items():
            # We flatten to compute the cost matrices in a batch
            lprobas = F.sigmoid(outputs["pred_"+class_name].flatten(0, 1))
            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v[class_name] for v in targets])
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_labels += -lprobas[:, tgt_ids]   
       
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_labels * cost_labels + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
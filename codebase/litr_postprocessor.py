import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class LiTrPostProcessor(nn.Module):
    def __init__(self, num_classes, num_queries, deploy_mode=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.deploy_mode = deploy_mode

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
    
    @torch.no_grad()
    def forward(self, outputs, top_k=100, score_thresh=0.5):
        if top_k > self.num_queries:
            top_k = self.num_queries
            print(f"top_k : {top_k} is larger than num_queries, model can only return upto {self.num_queries} queries.")

        boxes = outputs["pred_boxes"] # cxcywh format in [0, 1]
        bs = boxes.shape[0]
        logits = {}
        for class_name, num in self.num_classes.items():
            logits[class_name] = outputs[f"pred_{class_name}"]

        scores = []
        for class_name, num in self.num_classes.items():
            scores.append(F.sigmoid(logits[class_name]).max(-1).values)
        scores = torch.stack(scores, dim=-1)
        scores = torch.mean(scores, dim=-1)
        topk_scores, topk_indices = scores.topk(top_k, dim=1)

        topk_bboxes = boxes.gather(dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]))       # (bs, k, 4)

        topk_labels = []
        for class_name in sorted(self.num_classes.keys()):
            logits_i = logits[class_name]                                  # (bs, numq, class_dim)
            label_indices = logits_i.argmax(dim=-1)                        # (bs, numq)
            label_topk = label_indices.gather(1, topk_indices)             # (bs, k)
            topk_labels.append(label_topk.unsqueeze(-1))                   # (bs, k, 1)
        topk_labels = torch.cat(topk_labels, dim=-1)                       # (bs, k, num_classes)
        topk_scores = topk_scores.unsqueeze(-1) 

        results = []
        for b in range(bs):
            valid_mask = topk_scores[b].squeeze(-1) >= score_thresh  # (k,)
            results.append({
                "boxes": topk_bboxes[b][valid_mask], 
                "scores": topk_scores[b][valid_mask], 
                "labels": topk_labels[b][valid_mask],
            })
        
        if self.deploy_mode:
            # This is a workaround for the deploy mode
            # The postprocessor in deploy mode can only process one image at a time
            return results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        return results
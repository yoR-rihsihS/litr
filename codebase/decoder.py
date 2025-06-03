import copy 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

from .mlp import MLP
from .utils import bias_init_with_prob
from .utils import get_activation, inverse_sigmoid
from .deformable_attention import MSDeformableAttention
from .denoising import get_contrastive_denoising_training_group


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.1, activation='gelu', n_levels=3, n_points=4):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, target, reference_points, memory, memory_spatial_shapes, attn_mask=None, memory_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, target, ref_points_unact, memory, memory_spatial_shapes, bbox_head, score_head, query_pos_head, attn_mask=None, memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = {k : [] for k, v in self.num_classes.items()}
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                for class_name, num in self.num_classes.items():
                    dec_out_logits[class_name].append(score_head[class_name][i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                for class_name, num in self.num_classes.items():
                    dec_out_logits[class_name].append(score_head[class_name][i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        for class_name, num in self.num_classes.items():
            dec_out_logits[class_name] = torch.stack(dec_out_logits[class_name])
        dec_out_bboxes = torch.stack(dec_out_bboxes)

        return dec_out_bboxes, dec_out_logits

class LiTrDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 d_model,
                 num_queries,
                 feat_channels,
                 feat_strides,
                 num_levels,
                 num_points,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout=0.1,
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.d_model = d_model
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, "gelu", num_levels, num_points)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, num_classes, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            temp = {}
            for class_name, num in self.num_classes.items():
                temp[class_name] = nn.Embedding(num+1, d_model, padding_idx=num)
            self.denoising_class_embed = nn.ModuleDict(temp)

        # decoder embedding
        self.query_pos_head = MLP(4, 2 * d_model, d_model, 2, "gelu")

        self.enc_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        temp = {}
        for class_name, num in self.num_classes.items():
            temp[class_name] = nn.Linear(d_model, num)
        self.enc_score_head = nn.ModuleDict(temp)
        self.enc_bbox_head = MLP(d_model, d_model, 4, 3, "gelu")

        # decoder head
        temp = {}
        for class_name, num in self.num_classes.items():
            temp[class_name] = nn.ModuleList([
                nn.Linear(d_model, num) for _ in range(num_layers)
            ])
        self.dec_score_head = nn.ModuleDict(temp)
        self.dec_bbox_head = nn.ModuleList([
            MLP(d_model, d_model, 4, 3, "gelu") for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()
        
    def _reset_parameters(self):
        # --- Classification heads (encoder)
        cls_bias = bias_init_with_prob(0.01)
        for class_name, head in self.enc_score_head.items():
            init.constant_(head.bias, cls_bias)
            init.xavier_uniform_(head.weight)

        # --- Classification heads (decoder)
        for class_name, head in self.dec_score_head.items():
            for head in head:
                init.constant_(head.bias, cls_bias)
                init.xavier_uniform_(head.weight)

        # --- Regression heads (encoder)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for layer in self.enc_bbox_head.layers[:-1]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias'):
                init.constant_(layer.bias, 0.001)

        # --- Regression heads (decoder)
        for bbox_mlp in self.dec_bbox_head:
            # final layer zero-init
            init.constant_(bbox_mlp.layers[-1].weight, 0)
            init.constant_(bbox_mlp.layers[-1].bias, 0)
            # intermediate layers Xavier
            for layer in bbox_mlp.layers[:-1]:
                if hasattr(layer, 'weight'):
                    init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias'):
                    init.constant_(layer.bias, 0.001)

        # --- Backbone projection
        for proj in self.input_proj:
            init.xavier_uniform_(proj[0].weight)
            if proj[0].bias is not None:
                init.constant_(proj[0].bias, 0.001)

        # --- Encoder output transform
        init.xavier_uniform_(self.enc_output[0].weight)
        init.constant_(self.enc_output[1].weight, 1)
        init.constant_(self.enc_output[1].bias, 0)

        # --- Query position head
        for layer in self.query_pos_head.layers:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias'):
                init.constant_(layer.bias, 0.001)

        # --- Denoising class embeddings
        if hasattr(self, 'denoising_class_embed'):
            for embed in self.denoising_class_embed.values():
                # normal init for all but padding_idx
                init.normal_(embed.weight[:-1])

        # --- LayerNorms
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, 1, bias=False), 
                    nn.BatchNorm2d(self.d_model),
                )
            )
            
        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, 3, 2, padding=1, bias=False),
                    nn.BatchNorm2d(self.d_model),
                )
            )
            in_channels = self.d_model

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        memory = valid_mask.to(memory.dtype) * memory  

        output_memory = self.enc_output(memory)
        enc_outputs_logits = {}
        for class_name, num in self.num_classes.items():
            enc_outputs_logits[class_name] = self.enc_score_head[class_name](output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_dict = [], {}
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        # check this again
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_dict.update(enc_topk_logits)

        content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            denoising_content = denoising_logits["label_0"]
            for class_name, num in self.num_classes.items():
                if class_name == "label_0":
                    continue
                denoising_content += denoising_logits[class_name]
            content = torch.concat([denoising_content, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_dict

    def _select_topk(self, memory, outputs_logits, outputs_coords_unact, topk):
        output_scores = F.sigmoid(outputs_logits["label_0"]).max(-1).values
        for class_name, num in self.num_classes.items():
            if class_name == "label_0":
                continue
            output_scores += F.sigmoid(outputs_logits[class_name]).max(-1).values
        _, topk_ind = torch.topk(output_scores, topk, dim=-1)

        topk_coords = outputs_coords_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits ={}
        for class_name, num in self.num_classes.items():
            topk_logits[class_name] = outputs_logits[class_name].gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits[class_name].shape[-1]))
        
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords


    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes, 
                self.num_queries, 
                self.denoising_class_embed, 
                num_denoising=self.num_denoising, 
                label_noise_ratio=self.label_noise_ratio, 
                box_noise_scale=self.box_noise_scale
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_dict = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact
        )

        # decoder
        temp_bboxes, temp_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask
        )

        dn_out_logits, out_logits = {}, {}
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(temp_bboxes, dn_meta["dn_num_split"], dim=2)
            for class_name, num in self.num_classes.items():
                dn_out_logits[class_name], out_logits[class_name] = torch.split(temp_logits[class_name], dn_meta["dn_num_split"], dim=2)
        else:
            out_logits = temp_logits
            out_bboxes = temp_bboxes

        out = {"pred_boxes": out_bboxes[-1]}
        for class_name, num in self.num_classes.items():
            out[f"pred_{class_name}"] = out_logits[class_name][-1]

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss({k : v[:-1] for k, v in out_logits.items()}, out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss({k : [v] for k, v in enc_topk_logits_dict.items()}, enc_topk_bboxes_list))
            
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_predictions = []
        for i in range(len(outputs_coord)):
            out = {"pred_boxes": outputs_coord[i]}
            for class_name, num in self.num_classes.items():
                out[f"pred_{class_name}"] = outputs_class[class_name][i]
            aux_predictions.append(out)
        return aux_predictions
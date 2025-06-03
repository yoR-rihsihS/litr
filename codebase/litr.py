import torch.nn as nn 

from .backbone import BackBone
from .hybrid_encoder import HybridEncoder
from .decoder import LiTrDecoder

class LiTr(nn.Module):
    def __init__(self,
                 num_classes, backbone_model, hidden_dim, nhead, ffn_dim, num_encoder_layers, eval_spatial_size,
                 aux_loss, num_queries, num_decoder_points, num_denoising, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = BackBone(model=backbone_model, num_levels=3)

        self.in_channels = [128, 256, 512] if backbone_model in ["resnet18", "resnet34"] else [512, 1024, 2048]
        self.feat_strides = [8, 16, 32]
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dim_feedforward = ffn_dim
        self.dropout = dropout
        self.eval_spatial_size = eval_spatial_size
        self.num_encoder_layers = num_encoder_layers
        
        self.encoder = HybridEncoder(
            in_channels = self.in_channels,
            feat_strides = self.feat_strides,
            hidden_dim = self.hidden_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            use_encoder_idx = [2],
            num_encoder_layers = self.num_encoder_layers,
            eval_spatial_size = self.eval_spatial_size
        )

        self.aux_loss = aux_loss
        self.num_queries = num_queries
        self.num_decoder_points = num_decoder_points
        self.num_denoising = num_denoising
        self.num_decoder_layers = num_decoder_layers

        self.decoder = LiTrDecoder(
            num_classes = self.num_classes,
            d_model = self.hidden_dim,
            num_queries = self.num_queries,
            feat_channels = len(self.in_channels) * [hidden_dim],
            feat_strides = self.feat_strides,
            num_levels = len(self.in_channels),
            num_points = self.num_decoder_points,
            nhead = self.nhead,
            num_layers = self.num_decoder_layers,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            num_denoising = self.num_denoising,
            label_noise_ratio = 0.5,
            box_noise_scale = 1.0,
            eval_spatial_size = self.eval_spatial_size,
            aux_loss = self.aux_loss,
        )
        
    def forward(self, x, targets=None):          
        x = self.backbone(x)
        x = self.encoder(x)     
        x = self.decoder(x, targets)
        return x
    
    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
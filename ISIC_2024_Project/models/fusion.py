import torch
import torch.nn as nn
import timm

class FusionModel(nn.Module):
    def __init__(self, backbone_name="tf_efficientnetv2_s", meta_in_dim=0, num_classes=2, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)  # features only
        feat_dim = self.backbone.num_features
        self.meta_in_dim = meta_in_dim

        # small meta MLP
        if meta_in_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_in_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1)
            )
            fusion_dim = feat_dim + 64
        else:
            self.meta_mlp = None
            fusion_dim = feat_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_meta=None):
        f = self.backbone(x_img)  # shape (B, feat_dim)
        if self.meta_mlp is not None and x_meta is not None and x_meta.nelement() != 0:
            m = self.meta_mlp(x_meta)
            f = torch.cat([f, m], dim=1)
        out = self.head(f)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    ResNet50_Weights,
    resnet50,
    ResNet101_Weights,
    resnet101,
)
from pytorch_pretrained_vit import ViT

from torch_nn import InstancewiseVisualPromptCoordNet, Classifier, DomainDiscriminator
from torch_utils import freeze_layers, grad_reverse

def get_backbone(backbone):
    if backbone == "resnet18":
        return resnet18(ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        return resnet50(ResNet50_Weights.IMAGENET1K_V1)
    elif backbone == "resnet101":
        return resnet101(ResNet101_Weights.IMAGENET1K_V1)
    elif backbone == "vit_b_32":
        return ViT("B_32_imagenet1k", pretrained=True)
    elif backbone == "vit_b_16":
        return ViT("B_16_imagenet1k", pretrained=True)
    else:
        raise ValueError("Unsupported backbone architecture")

# def center_crop(x, h_crop, w_crop):
#     _, _, h, w = x.shape
#     start_h = (h - h_crop) // 2
#     start_w = (w - w_crop) // 2
#     return x[:, :, start_h:start_h+h_crop, start_w:start_w+w_crop]

class UModel(nn.Module):
    def __init__(
        self, 
        backbone="vit_b_32",
        hidden_dim=256,
        out_dim=65,
        imgsize=384, 
        scaled_factor = [1, 2, 4], 
        layers = [5, 6, 6], 
        patch_size = [8, 16, 32], 
        freeze_backbone=True
    ):
        super(UModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out_dim = out_dim
        self.total_vrs = len(scaled_factor)
        self.scaled_factor = scaled_factor
        if freeze_backbone:
            freeze_layers([self.backbone])
        prompts_src, prompts_tgt = [], []
        for prompt in (prompts_src, prompts_tgt):
            for i in range(self.total_vrs):
                prompt.append(
                    InstancewiseVisualPromptCoordNet(
                        size = imgsize // scaled_factor[i],
                        layers=layers[i], 
                        patch_size=patch_size[i],
                        channels=3,
                        dropout_p=0.3,
                    )
                )
        self.visual_prompts_src = nn.Sequential(*prompts_src)
        self.visual_prompts_tgt = nn.Sequential(*prompts_tgt)

        self.classifier_head_src = Classifier(
            in_dim=self.in_dim * self.total_vrs,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.2,
        )
        self.classifier_head_tgt = Classifier(
            in_dim=self.in_dim * self.total_vrs,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.2,
        )
        self.domain_discriminator = DomainDiscriminator(
            in_dim=self.in_dim * self.total_vrs,
            hidden_dim=hidden_dim,
            out_dim=2,
            dropout=0.2,
        )

    def forward(self, x, vr_branch, head_branch, grl_alpha=None):
        prompt = self.visual_prompts_src if vr_branch == "src" else self.visual_prompts_tgt
        if head_branch == "src":
            head = self.classifier_head_src
        elif head_branch == "tgt":
            head = self.classifier_head_tgt
        elif head_branch == "domain":
            head = self.domain_discriminator
        else:
            raise ValueError(f"Unknown head branch {head_branch}")

        _, _, h, w = x.shape
        feats = []
        for i in range(self.total_vrs):
            x_down = F.interpolate(x, size=(h // self.scaled_factor[i], w // self.scaled_factor[i]), mode="bilinear", align_corners=False, antialias=True)
            x_prompt = prompt[i](x_down)
            x_up = F.interpolate(x_prompt, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
            feats.append(self.backbone(x_up))
            
        x = torch.cat(feats, dim=1)
        if head_branch == "domain" and grl_alpha is not None:
            x = grad_reverse(x, grl_alpha)
        return head(x)
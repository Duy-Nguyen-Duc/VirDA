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

from torch_nn import (
    InstancewiseVisualPromptCoordNet, 
    Classifier, 
    DomainDiscriminator
)
from torch_utils import (
    freeze_layers, 
    grad_reverse
)

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


class UModel(nn.Module):
    def __init__(
        self, 
        backbone="vit_b_32",
        hidden_dim=256,
        out_dim=65,
        imgsize=384, 
        layer = 6, 
        patch_size = 32,
        freeze_backbone=True
    ):
        super(UModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out_dim = out_dim

        if freeze_backbone:
            freeze_layers([self.backbone])

        self.stu_vr = InstancewiseVisualPromptCoordNet(
            size = imgsize,
            layers=layer, 
            patch_size=patch_size,
            channels=3,
            dropout_p=0.3,
        )
        self.tch_vr = InstancewiseVisualPromptCoordNet(
            size = imgsize,
            layers=layer, 
            patch_size=patch_size,
            channels=3,
            dropout_p=0.3,
        )
        self.stu_cls = Classifier(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.2,
        )
        self.tch_cls = Classifier(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=0.2,
        )
        self.domain_discriminator = DomainDiscriminator(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim,
            out_dim=2,
            dropout=0.2,
        )

    def forward(self, x, branch, saliency_map=None, grl_alpha=None):
        """
            x: [B,3,H,W]
            vr_branch: "stu" or "tch"
            head_branch: "stu" or "tch" or "domain"
            saliency_map: [B,1,H,W] in [0,1]
            grl_alpha: float
        """
        if branch == "stu":
            prompt, head = self.stu_vr, self.stu_cls
        elif branch == "tch":
            prompt, head = self.tch_vr, self.tch_cls
        elif branch == "domain":
            prompt, head = self.stu_vr, self.domain_discriminator
        else:
            raise ValueError(f"Unknown branch {branch}")

        x_1 = prompt(x)

        if saliency_map is not None:
            delta_x = x_1 - x
            x_1 = x + saliency_map * delta_x

        feats = self.backbone(x_1)
        
        if branch == "domain" and grl_alpha is not None:
            feats = grad_reverse(feats, grl_alpha)
        
        logits = head(feats)
        return logits

class EigenCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = None
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def register_hook(self):
        if self.hook_handle is None:
            self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def __call__(self, input_tensor, branch="tch"):
        self.activations = None
        
        with torch.no_grad():
            _ = self.model(input_tensor, branch=branch)
        
        if self.activations is None:
            raise RuntimeError("No activations captured. Check target layer.")
        
        activations = self.activations
        B = activations.shape[0]
        
        if activations.shape[1] > 1:
            activations = activations[:, 1:, :]
        
        original_dtype = activations.dtype
        if activations.dtype == torch.float16:
            activations = activations.float()
        
        mean = activations.mean(dim=2, keepdim=True)
        centered = activations - mean
        
        cov = torch.bmm(centered, centered.transpose(1, 2))
        
        v = torch.randn(B, cov.shape[1], 1, device=cov.device, dtype=cov.dtype)
        
        for _ in range(10):
            v = torch.bmm(cov, v)
            v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-10)
        
        cam = v.squeeze(-1)
        
        num_patches_side = int(cam.shape[1] ** 0.5)
        cam = cam.view(B, num_patches_side, num_patches_side)
        
        cam_min = cam.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam_max = cam.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-10)
        
        if original_dtype == torch.float16:
            cam = cam.half()

        return cam
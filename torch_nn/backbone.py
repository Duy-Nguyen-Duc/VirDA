from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    ResNet50_Weights,
    resnet50,
    ResNet101_Weights,
    resnet101,
)
from pytorch_pretrained_vit import ViT

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
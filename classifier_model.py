import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from src.layers.torch_nn import Classifier
from src.layers.instance_model import InstancewiseVisualPrompt


class BaseClassifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        imgsize=224,
        attribute_layers=5,
        patch_size=16,
        attribute_channels=3,
    ):
        super(BaseClassifier, self).__init__()
        self.backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.visual_prompt = InstancewiseVisualPrompt(
            imgsize, attribute_layers, patch_size, attribute_channels
        )
        self.classifier_head = Classifier(
            in_dim=512, hidden_dim=256, out_dim=num_classes, num_res_blocks=2, dropout=0.5
        )
    
    def forward(self, x, output_type="logits"):
        feat = self.backbone(self.visual_prompt(x))
        if output_type == "feat":
            return feat
        elif output_type == "logits":
            return self.classifier_head(feat)
        else:
            print(f"Not implemented output type {output_type}") 

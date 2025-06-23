import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18, ResNet50_Weights, resnet50, ResNet101_Weights, resnet101

from torch_nn import Classifier, InstancewiseVisualPrompt


class BaseClassifier(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        in_dim=512,
        hidden_dim=256,
        out_dim=10,
        num_res_blocks=2,
        imgsize=64,
        attribute_layers=5,
        patch_size=8,
        attribute_channels=3,
        p_vr_src=0.5,
        p_vr_tgt=0.3,
        p_cls_src=0.3,
        p_cls_tgt=0.7,
    ):
        super(BaseClassifier, self).__init__()
        if backbone == "resnet18":
            self.backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = resnet50(ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet101":
            self.backbone = resnet101(ResNet101_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone architecture")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.visual_prompt_src = InstancewiseVisualPrompt(
            imgsize,
            attribute_layers,
            patch_size,
            attribute_channels,
            dropout_p=p_vr_src,
        )
        self.visual_prompt_tgt = InstancewiseVisualPrompt(
            imgsize,
            attribute_layers,
            patch_size,
            attribute_channels,
            dropout_p=p_vr_tgt,
        )
        self.classifier_head_src = Classifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_res_blocks=num_res_blocks,
            dropout=p_cls_src,
        )
        self.classifier_head_tgt = Classifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_res_blocks=num_res_blocks,
            dropout=p_cls_tgt,
        )

    def forward(
        self, x, branch: str, inf_type: str, out_type: str, mc_samples=None, tau=None
    ):
        if branch == "src":
            components = [self.visual_prompt_src, self.classifier_head_src]
        elif branch == "tgt":
            components = [self.visual_prompt_tgt, self.classifier_head_tgt]
        elif branch == "tgt_mix":
            components = [self.visual_prompt_tgt, self.classifier_head_src]
        else:
            raise f"unknown branch: {branch}"

        if inf_type == "det":
            x = components[0](x)
            x = self.backbone(x)
            if out_type == "feat":
                return x
            elif out_type == "logits":
                return components[1](x)

        elif inf_type == "mc":
            assert mc_samples is not None and tau is not None
            for i in range(len(components)):
                components[i].train()
            B = x.size(0)
            if out_type == "feat":
                feat = torch.zeros(B, self.in_dim, device=x.device)
                for _ in range(mc_samples):
                    feat += self.forward(x, branch, inf_type="det", out_type="feat")
                return feat / mc_samples

            elif out_type == "logits":
                logits = torch.zeros(B, self.out_dim, device=x.device)
                for _ in range(mc_samples):
                    logits += F.softmax(
                        self.forward(x, branch, inf_type="det", out_type="logits")
                        / tau,
                        dim=-1,
                    )
                logits /= mc_samples
                uncertainty = -(logits * (logits + 1e-8).log()).sum(-1)
                return logits, uncertainty

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.layers.grl import grad_reverse
from src.layers.instance_model import InstancewiseVisualPrompt
from src.layers.torch_nn import Classifier


class DomainAdaptationModel(nn.Module):
    def __init__(
        self,
        num_classes_src=10,
        num_classes_tgt=10,
        imgsize=224,
        attribute_layers=5,
        patch_size=16,
        attribute_channels=3,
    ):
        super(DomainAdaptationModel, self).__init__()
        # Pretrained backbone: ResNet-18 with final fc replaced.
        self.backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Generalized classifier heads (you can adjust hidden dimensions, # blocks, etc.)
        self.src_classifier = Classifier(
            in_dim=512,
            hidden_dim=256,
            out_dim=num_classes_src,
            num_res_blocks=2,
            dropout=0.5,
        )
        self.tgt_classifier = Classifier(
            in_dim=512,
            hidden_dim=256,
            out_dim=num_classes_tgt,
            num_res_blocks=2,
            dropout=0.5,
        )
        self.domain_classifier = Classifier(
            in_dim=512, hidden_dim=256, out_dim=2, num_res_blocks=2, dropout=0.5
        )

        # Domain mapper: maps from source prediction space to target prediction space.
        self.domain_mapper = Classifier(
            in_dim=num_classes_src,
            hidden_dim=512,
            out_dim=num_classes_tgt,
            num_res_blocks=2,
            dropout=0.5,
        )

        # Visual prompt module is assumed defined elsewhere.
        self.visual_prompt = InstancewiseVisualPrompt(
            imgsize, attribute_layers, patch_size, attribute_channels
        )

    def forward(self, src_img, tgt_img, alpha, branch="da_train"):
        if branch == "da_train": 
            vis_prompted_img = self.visual_prompt(tgt_img)
        
            with torch.no_grad():
                src_feat = self.backbone(src_img)
                tgt_feat = self.backbone(vis_prompted_img)

            src_logits = self.src_classifier(src_feat)

            tgt_feat_rvs = grad_reverse(tgt_feat, alpha)
            src_domain_logits = self.domain_classifier(src_feat.detach())
            tgt_domain_logits = self.domain_classifier(tgt_feat_rvs)
            return src_logits, src_domain_logits, tgt_domain_logits

        elif branch == "tgt_train":
            # tgt_q is now src_img / tgt_k is now tgt_img
            self.visual_prompt.requires_grad_(False)
            self.visual_prompt.eval()
            self.src_classifier.requires_grad_(False)
            self.src_classifier.eval()

            tgt_q_logits = self.domain_mapper(
                self.src_classifier(self.backbone(self.visual_prompt(src_img)))
            )
            tgt_k_logits = self.tgt_classifier(self.backbone(tgt_img))
            return tgt_q_logits, tgt_k_logits

        elif branch == "src_test":
            src_feat = self.backbone(src_img)
            src_logits = self.src_classifier(src_feat)
            return src_logits

        elif branch == "tgt_test":
            fx = self.backbone(src_img)
            tgt_logits = self.tgt_classifier(fx)
            return tgt_logits
        
        else:
            raise ValueError(
                "Unknown branch: {}. Choose from 'da_train', 'src_test', 'tgt_test', 'tgt_train'.".format(
                    branch
                )
            )

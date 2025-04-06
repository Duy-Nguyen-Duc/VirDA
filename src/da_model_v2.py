import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.layers.grl import grad_reverse
from src.layers.instance_model import InstancewiseVisualPrompt
from src.layers.torch_nn import Classifier
from src.layers.utils import freeze_layers


class DA_model_v2(nn.Module):
    def __init__(
        self,
        num_classes_src=10,
        num_classes_tgt=10,
        imgsize=224,
        attribute_layers=5,
        patch_size=16,
        attribute_channels=3,
    ):
        super(DA_model_v2, self).__init__()

        self.backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        
        self.src_classifier = Classifier(
            in_dim=512,
            hidden_dim=256,
            out_dim=num_classes_src,
            num_res_blocks=3,
            dropout=0.5,
        )
        self.tgt_classifier = Classifier(
            in_dim=512,
            hidden_dim=256,
            out_dim=num_classes_tgt,
            num_res_blocks=3,
            dropout=0.5,
        )
        self.domain_classifier = Classifier(
            in_dim=512, hidden_dim=256, out_dim=2, num_res_blocks=2, dropout=0.5
        )

        self.domain_mapper = Classifier(
            in_dim=num_classes_src,
            hidden_dim=512,
            out_dim=num_classes_tgt,
            num_res_blocks=2,
            dropout=0.5,
        )

        # Visual prompt module is assumed defined elsewhere.
        self.src_visual_prompt = InstancewiseVisualPrompt(
            imgsize+32, attribute_layers, patch_size, attribute_channels #32 is defined in the data loader
        )
        self.tgt_visual_prompt = InstancewiseVisualPrompt(
            imgsize, attribute_layers, patch_size, attribute_channels
        )

    def forward(self, src_img, tgt_img, alpha, branch="da_train"):
        if branch == "da_train": 
            freeze_layers([self.backbone])

            vis_prompted_src_img = self.src_visual_prompt(src_img)
            vis_prompted_tgt_img = self.tgt_visual_prompt(tgt_img)

            src_feat = self.backbone(vis_prompted_src_img)
            tgt_feat = self.backbone(vis_prompted_tgt_img)

            src_logits = self.src_classifier(src_feat)

            tgt_feat_rvs = grad_reverse(tgt_feat, alpha)
            src_domain_logits = self.domain_classifier(src_feat.detach())
            tgt_domain_logits = self.domain_classifier(tgt_feat_rvs)
            return src_logits, src_domain_logits, tgt_domain_logits

        elif branch == "tgt_train":
            # tgt_q is now src_img / tgt_k is now tgt_img
            freeze_layers([self.backbone, self.src_classifier, self.src_visual_prompt])

            tgt_q_logits = self.domain_mapper(
                self.src_classifier(self.backbone(self.src_visual_prompt(src_img)))
            )
            tgt_k_logits = self.tgt_classifier(self.backbone(self.tgt_visual_prompt(tgt_img)))
            return tgt_q_logits, tgt_k_logits

        elif branch == "src_test":
            src_feat = self.backbone(src_img)
            src_logits = self.src_classifier(src_feat)
            return src_logits

        elif branch == "tgt_test_stu":
            fx = self.backbone(src_img)
            tgt_logits = self.tgt_classifier(fx)
            return tgt_logits
        
        elif branch == "tgt_test_tch":
            tgt_logits = self.domain_mapper(
                self.src_classifier(self.backbone(self.src_visual_prompt(src_img)))
            )
            return tgt_logits
        
        else:
            raise ValueError(
                "Unknown branch: {}. Choose from 'da_train', 'src_test','tgt_train', 'tgt_test_stu', 'tgt_test_tch'.".format(
                    branch
                )
            )

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.layers.grl import grad_reverse
from src.layers.instance_model import InstancewiseVisualPrompt
from src.layers.torch_nn import Classifier
from src.layers.utils import freeze_layers


class DA_model_v3(nn.Module):
    def __init__(
        self,
        num_classes_src=10,
        num_classes_tgt=10,
        imgsize=224,
        attribute_layers=5,
        patch_size=16,
        attribute_channels=3,
    ):
        super(DA_model_v3, self).__init__()

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

        # Visual prompt module is assumed defined elsewhere.
        self.visual_prompt = InstancewiseVisualPrompt(
            imgsize, attribute_layers, patch_size, attribute_channels
        )

    def forward(self, img_x: torch.Tensor, img_y: torch.Tensor, alpha: float, branch: str ="da_train"):
        """
        Forward pass for the model

        Args:
            img_x (torch.Tensor): input image for the more confident. It could be source domain image or target weak image
            img_y (torch.Tensor): input image for the less confident. It could be target domain image or target strong image
            alpha (float): _description_
            branch (str, optional): _description_. Defaults to "da_train".

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """        
        
        if branch == "da_train": 
            freeze_layers([self.backbone])

            vis_prompted_img_y = self.visual_prompt(img_y)

            src_feat = self.backbone(img_x)
            tgt_feat = self.backbone(vis_prompted_img_y)

            src_logits = self.src_classifier(src_feat)

            tgt_feat_rvs = grad_reverse(tgt_feat, alpha)
            src_domain_logits = self.domain_classifier(src_feat)
            tgt_domain_logits = self.domain_classifier(tgt_feat_rvs)
            return src_logits, src_domain_logits, tgt_domain_logits

        elif branch == "tgt_train":
            # tgt_q is now src_img / tgt_k is now tgt_img
            freeze_layers([self.backbone, self.src_classifier, self.visual_prompt])

            tgt_q_logits = self.src_classifier(self.backbone(self.visual_prompt(img_x)))
            tgt_k_logits = self.tgt_classifier(self.backbone(img_y))
            return tgt_q_logits, tgt_k_logits

        elif branch == "src_test":
            src_feat = self.backbone(img_x)
            src_logits = self.src_classifier(src_feat)
            return src_logits

        elif branch == "tgt_test_stu":
            fx = self.backbone(img_x)
            tgt_logits = self.tgt_classifier(fx)
            return tgt_logits
        
        elif branch == "tgt_test_tch":
            tgt_logits = self.src_classifier(self.backbone(self.visual_prompt(img_x)))
            return tgt_logits
        
        else:
            raise ValueError(
                "Unknown branch: {}. Choose from 'da_train', 'src_test','tgt_train', 'tgt_test_stu', 'tgt_test_tch'.".format(
                    branch
                )
            )

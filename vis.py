import torch
import cv2
from model import UModel, EigenCAM
from yacs.config import CfgNode as CN
from torch_utils import visualize_salience_map
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from data import transform_map
import torch.nn.functional as F
import matplotlib.pyplot as plt

cfg = CN(new_allowed=True)
cfg.merge_from_file("config.yaml")

model = UModel(
    backbone=cfg.model.backbone.type,
    hidden_dim=cfg.model.backbone.hidden_dim,
    out_dim=cfg.dataset.num_classes,
    imgsize=cfg.img_size,
    freeze_backbone=cfg.model.backbone.freeze,
)

device = torch.device(cfg.device)
ckpt = torch.load("runs/new_28_10/nnRid5/da_best_64.34.pth")
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)

input_path = "data/OfficeHome/Clipart/Mouse/00006.jpg"
img_size = 384
img_cv = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
original_img = np.array(img_pil)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil)
img_tensor = img_tensor.unsqueeze(0).to(device)

def visualize_results(img_tensor, model, device):
    """
    Visualize the original image, program, and reprogrammed image
    """
    with torch.no_grad():
        # Get attention and program
        # x = model.tgt_vr.coord_att(img_tensor)
        x = img_tensor
        att = model.tgt_vr.priority(x)
        attention = (
            att.view(-1, model.tgt_vr.channels, model.tgt_vr.patch_num * model.tgt_vr.patch_num, 1)
                .expand(-1, 3, -1, model.tgt_vr.patch_size * model.tgt_vr.patch_size)
                .view(-1, 3, model.tgt_vr.patch_num, model.tgt_vr.patch_num, model.tgt_vr.patch_size, model.tgt_vr.patch_size)
                .transpose(3, 4)
                .reshape(-1, 3, model.tgt_vr.imgsize, model.tgt_vr.imgsize)
        )
        img_feat = model.tgt_vr.img_agg(x)
        program = model.tgt_vr.program_producer(img_feat)
        
        # Create reprogrammed image
        reprogrammed = x + program * attention
        
        # Convert tensors to numpy arrays for visualization
        # Original image (denormalize)
        img_original = img_tensor[0].cpu().numpy()
        img_original = np.transpose(img_original, (1, 2, 0))
        img_original = img_original * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_original = np.clip(img_original, 0, 1)
        
        # Program visualization (average across channels)
        program_viz = program[0].cpu().numpy()
        program_viz = np.mean(program_viz, axis=0)  # Average across RGB channels
        # program_viz = (program_viz - program_viz.min()) / (program_viz.max() - program_viz.min())
        
        # Reprogrammed image (denormalize)
        reprogrammed_viz = reprogrammed[0].cpu().numpy()
        reprogrammed_viz = np.transpose(reprogrammed_viz, (1, 2, 0))
        reprogrammed_viz = reprogrammed_viz * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        reprogrammed_viz = np.clip(reprogrammed_viz, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Program
        im1 = axes[1].imshow(program_viz, cmap='hot')
        axes[1].set_title('Program (Attention Map)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Reprogrammed image
        axes[2].imshow(reprogrammed_viz)
        axes[2].set_title('Reprogrammed Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualization_results_tgt_2.png', dpi=300, bbox_inches='tight')
        plt.show()
        return program_viz
        
program_viz = visualize_results(img_tensor, model, device)
print(program_viz.mean())

# from model import EigenCAM
# from torch_utils import visualize_salience_map
# cam = EigenCAM(model, model.backbone.transformer.blocks[-1].norm2)
# cam.register_hook()
# visualize_salience_map(input_path, cam, model.tgt_vr, model.tgt_cls, device, "smap_tgt.png")
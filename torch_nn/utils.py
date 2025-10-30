import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

@torch.no_grad()
def freeze_layers(layers: list[nn.Module]):
    """
    Freeze layers listed in the model
    """
    for layer in layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False


def compute_hard_alpha(u: torch.Tensor, t_u: float):
    """
    Compute hard alpha.

    Args:
        u (torch.Tensor): filtering tensor
        t_u (float): threshold

    """
    mask = (u <= t_u).float()
    w = torch.exp(-u) * mask
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha(u: torch.Tensor):
    w = torch.exp(-u)
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha_anneal(u, step, total_steps, min_temp=0.1, max_temp=1.0):
    frac = step / float(total_steps)
    T = max_temp * (1 - frac) + min_temp * frac
    w = torch.exp(-u / T)
    w = w / (w.max().clamp(min=1e-6))
    return w


def decay_thresholds(thres_start, thres_end, total_steps, method="exp"):
    if method == "exp":
        t = np.linspace(0, 1, total_steps)
        values = thres_start * (thres_end / thres_start) ** t
    elif method == "log":
        t = np.logspace(0, 1, total_steps, base=10)
        t = (t - t.min()) / (t.max() - t.min())
        values = thres_start - (thres_start - thres_end) * t
    else:
        raise ValueError("Invalid method. Use 'exp' or 'log'.")
    return list(values)

@torch.no_grad()
def ema_update(model, ema_model, alpha):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def visualize_salience_map(input_path, cam, vr_branch, head_branch, device, outpath, img_size=384):
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
    salience_map = cam(x=img_tensor, vr_branch=vr_branch, head_branch=head_branch)

    salience_upsampled = F.interpolate(
        salience_map, 
        size=(img_size, img_size), 
        mode='bilinear', 
        align_corners=False
    )
    
    salience_np = salience_upsampled[0, 0].cpu().numpy()
    img_resized = cv2.resize(original_img, (img_size, img_size))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(salience_np, cmap='jet', alpha=0.8)
    axes[1].set_title('Salience Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_resized)
    im2 = axes[2].imshow(salience_np, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()

def visualize_multiple_images(input_paths, model, cam, vr_branch, head_branch, device, outpath, img_size=384):
    """
    Visualize multiple images with 4 columns per row: original, program, saliency map, and overlay.
    
    Args:
        input_paths (list[str]): List of image paths to visualize
        model: The UModel instance
        cam: EigenCAM instance
        vr_branch (str): "src" or "tgt" for visual reprogramming branch
        head_branch (str): "src" or "tgt" for classification head branch
        device: torch device
        outpath (str): Path to save the output image
        img_size (int): Size to resize images to (default: 384)
    
    Returns:
        None (saves the visualization to outpath)
    """
    B = len(input_paths)
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get prompt branch
    prompt = model.src_vr if vr_branch == "src" else model.tgt_vr
    
    # Process all images
    all_originals = []
    all_programs = []
    all_saliences = []
    
    with torch.no_grad():
        for input_path in input_paths:
            # Load and preprocess image
            img_cv = cv2.imread(input_path)
            if img_cv is None:
                raise ValueError(f"Could not load image from {input_path}")
            
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            original_img = np.array(img_pil)
            
            # Store resized original
            img_resized = cv2.resize(original_img, (img_size, img_size))
            all_originals.append(img_resized)
            
            # Prepare tensor
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Get program (attention map)
            x_prompt = prompt(img_tensor)
            
            # Program visualization (average across channels)
            program_viz = x_prompt[0].cpu().numpy()
            program_viz = np.mean(program_viz, axis=0)  # Average across RGB channels
            program_viz = (program_viz - program_viz.min()) / (program_viz.max() - program_viz.min() + 1e-8)
            all_programs.append(program_viz)
            
            # Get salience map
            salience_map = cam(x=img_tensor, vr_branch=vr_branch, head_branch=head_branch)
            salience_upsampled = F.interpolate(
                salience_map, 
                size=(img_size, img_size), 
                mode='bilinear', 
                align_corners=False
            )
            salience_np = salience_upsampled[0, 0].cpu().numpy()
            all_saliences.append(salience_np)
    
    # Create figure with B rows and 4 columns
    fig, axes = plt.subplots(B, 4, figsize=(20, 5 * B))
    
    # Handle single image case (axes will be 1D instead of 2D)
    if B == 1:
        axes = axes.reshape(1, -1)
    
    # Fill in the subplots
    for i in range(B):
        # Column 0: Original Image
        axes[i, 0].imshow(all_originals[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Column 1: Program (Attention Map)
        im1 = axes[i, 1].imshow(all_programs[i], cmap='viridis')
        axes[i, 1].set_title('Program (Attention Map)')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Column 2: Salience Map
        im2 = axes[i, 2].imshow(all_saliences[i], cmap='jet', alpha=0.8)
        axes[i, 2].set_title('Salience Map')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Column 3: Overlay (Salience on Original)
        axes[i, 3].imshow(all_originals[i])
        im3 = axes[i, 3].imshow(all_saliences[i], cmap='jet', alpha=0.5)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
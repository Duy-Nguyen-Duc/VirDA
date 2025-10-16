import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.models import (
    ResNet18_Weights, resnet18,
    ResNet50_Weights, resnet50,
    ResNet101_Weights, resnet101,
)

from yacs.config import CfgNode as CN

# Try to import ViT if available
try:
    from pytorch_pretrained_vit import ViT
    VIT_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_pretrained_vit not available. ViT models will not work.")
    VIT_AVAILABLE = False

# Import data configs - will be defined inline if not available
try:
    from data_configs import DATASET_CONFIGS
except ImportError:
    print("Warning: data_configs.py not found. Using minimal configs.")
    DATASET_CONFIGS = {}

# Import torch utilities - will be defined inline if not available
try:
    from torch_utils import freeze_layers, grad_reverse
except ImportError:
    print("Warning: torch_utils.py not found. Defining utilities inline.")
    
    def freeze_layers(layers):
        """Freeze parameters of given layers"""
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None
    
    def grad_reverse(x, alpha=1.0):
        return GradReverse.apply(x, alpha)

# Import torch nn modules - will be defined inline if not available
try:
    from torch_nn import InstancewiseVisualPromptCoordNet, Classifier, DomainDiscriminator
except ImportError:
    print("Warning: torch_nn.py not found. Defining modules inline.")
    
    class InstancewiseVisualPromptCoordNet(nn.Module):
        """Visual Prompt Network with coordinate encoding"""
        def __init__(self, size=384, layers=6, patch_size=16, channels=3, dropout_p=0.3):
            super().__init__()
            self.size = size
            self.patch_size = patch_size
            hidden_dim = 128
            
            # Simple conv-based prompt generator
            self.prompt_net = nn.Sequential(
                nn.Conv2d(channels + 2, hidden_dim, 3, padding=1),  # +2 for coords
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Conv2d(hidden_dim, channels, 3, padding=1),
                nn.Tanh(),
            )
            
        def forward(self, x):
            b, c, h, w = x.shape
            # Add coordinate channels
            y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(b, 1, 1, w)
            x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(b, 1, h, 1)
            x_with_coords = torch.cat([x, y_coords, x_coords], dim=1)
            
            # Generate and apply prompt
            prompt = self.prompt_net(x_with_coords)
            return x + prompt * 0.1  # Scale prompt contribution
    
    class Classifier(nn.Module):
        """Classifier head"""
        def __init__(self, in_dim=768, hidden_dim=256, out_dim=65, dropout=0.2):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        
        def forward(self, x):
            return self.fc(x)
    
    class DomainDiscriminator(nn.Module):
        """Domain discriminator for adversarial training"""
        def __init__(self, in_dim=768, hidden_dim=256, out_dim=2, dropout=0.2):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        
        def forward(self, x):
            return self.fc(x)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def get_backbone(backbone):
    """Get pretrained backbone model"""
    if backbone == "resnet18":
        return resnet18(ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        return resnet50(ResNet50_Weights.IMAGENET1K_V1)
    elif backbone == "resnet101":
        return resnet101(ResNet101_Weights.IMAGENET1K_V1)
    elif backbone == "vit_b_32" and VIT_AVAILABLE:
        return ViT("B_32_imagenet1k", pretrained=True)
    elif backbone == "vit_b_16" and VIT_AVAILABLE:
        return ViT("B_16_imagenet1k", pretrained=True)
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone}")


class UModel(nn.Module):
    """Unified model for domain adaptation"""
    def __init__(
        self, 
        backbone="resnet50",
        hidden_dim=256,
        out_dim=65,
        imgsize=384, 
        scaled_factor=[1, 2, 4], 
        layers=[5, 6, 6], 
        patch_size=[8, 16, 32], 
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
            
        # Create visual prompts
        prompts_src, prompts_tgt = [], []
        for prompt in (prompts_src, prompts_tgt):
            for i in range(self.total_vrs):
                prompt.append(
                    InstancewiseVisualPromptCoordNet(
                        size=imgsize // scaled_factor[i],
                        layers=layers[i], 
                        patch_size=patch_size[i],
                        channels=3,
                        dropout_p=0.3,
                    )
                )
        self.visual_prompts_src = nn.Sequential(*prompts_src)
        self.visual_prompts_tgt = nn.Sequential(*prompts_tgt)

        # Create classification heads
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
            x_down = F.interpolate(
                x, 
                size=(h // self.scaled_factor[i], w // self.scaled_factor[i]), 
                mode="bilinear", 
                align_corners=False, 
                antialias=True
            )
            x_prompt = prompt[i](x_down)
            x_up = F.interpolate(
                x_prompt, 
                size=(h, w), 
                mode="bilinear", 
                align_corners=False, 
                antialias=True
            )
            feats.append(self.backbone(x_up))
            
        x = torch.cat(feats, dim=1)
        if head_branch == "domain" and grl_alpha is not None:
            x = grad_reverse(x, grl_alpha)
        return head(x)


# ============================================================================
# DATA LOADING
# ============================================================================

class StrongWeakAugDataset(Dataset):
    """Dataset with strong and weak augmentation"""
    def __init__(self, dataset_name, root, img_size=224, train=True, download=True):
        self.train = train
        ds_name = dataset_name.lower()
        
        # For demo purposes - you'll need to adapt this to your actual datasets
        if ds_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Dataset {dataset_name} not configured. "
                f"Please add it to data_configs.py or use a configured dataset."
            )

        cfg = DATASET_CONFIGS[ds_name]
        split = "train" if self.train else "test"
        args = cfg["args_fn"](train, root, download, split)
        self.dataset = cfg["cls"](**args)

        to_rgb_flag = cfg["convert_to_rgb"]
        mean = cfg["mean"]
        std = cfg["std"]
        affine_params = cfg["strong_affine"]
        jitter_params = cfg["jitter"]

        if to_rgb_flag:
            convert_to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
        else:
            convert_to_rgb = transforms.Lambda(
                lambda img: img if img.mode == "RGB" else img.convert("RGB")
            )

        self.weak_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.strong_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            convert_to_rgb,
            transforms.RandomAffine(
                degrees=affine_params["degrees"],
                translate=affine_params["translate"],
                scale=affine_params["scale"],
                shear=affine_params["shear"],
            ),
            transforms.ColorJitter(**jitter_params),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        weak_img = self.weak_transform(img)
        if self.train:
            strong_img = self.strong_transform(img)
            return weak_img, strong_img, label
        else:
            return weak_img, label


def make_dataset(
    source_dataset,
    target_dataset,
    img_size,
    train_bs,
    eval_bs,
    num_workers=4,
    distributed=False,
    rank=0,
    world_size=1,
):
    """Create data loaders for source and target domains"""
    source_train_data = StrongWeakAugDataset(
        dataset_name=source_dataset,
        root="./data",
        img_size=img_size,
        train=True,
        download=False,
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset,
        root="./data",
        img_size=img_size,
        train=True,
        download=False,
    )
    k = math.ceil(len(target_train_data) / len(source_train_data))

    # Create samplers for distributed training
    if distributed:
        source_train_sampler = DistributedSampler(
            source_train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        target_train_sampler = DistributedSampler(
            target_train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        source_train_sampler = None
        target_train_sampler = None

    source_train_loader = DataLoader(
        source_train_data,
        batch_size=train_bs,
        shuffle=(source_train_sampler is None),
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=source_train_sampler,
    )
    target_train_loader = DataLoader(
        target_train_data,
        batch_size=train_bs * k,
        shuffle=(target_train_sampler is None),
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=target_train_sampler,
    )

    source_test_data = StrongWeakAugDataset(
        dataset_name=source_dataset, root="./data", img_size=img_size, train=False
    )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root="./data", img_size=img_size, train=False
    )
    
    # For evaluation, use DistributedSampler without shuffling
    if distributed:
        source_test_sampler = DistributedSampler(
            source_test_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        target_test_sampler = DistributedSampler(
            target_test_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        source_test_sampler = None
        target_test_sampler = None
        
    source_test_loader = DataLoader(
        source_test_data,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=source_test_sampler,
    )
    target_test_loader = DataLoader(
        target_test_data,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=target_test_sampler,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_test_loader,
        target_test_loader,
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, branch, test_loader, device, distributed=False):
    """Evaluate model on test set"""
    # Handle DDP wrapper
    if hasattr(model, 'module'):
        model_eval = model.module
    else:
        model_eval = model
        
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Get rank for logging
    rank = dist.get_rank() if distributed else 0

    with torch.no_grad():
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating {branch}", ncols=100)
        else:
            pbar = test_loader
            
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images, vr_branch=branch, head_branch=branch)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Aggregate metrics across all GPUs if distributed
    if distributed:
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics[0].item(), int(metrics[1].item()), int(metrics[2].item())
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy


# ============================================================================
# UTILITIES
# ============================================================================

def setup(cfg):
    """Setup experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.exp_tags}_{timestamp}"
    exp_save_dir = os.path.join("./experiments", exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_save_dir, "config.yaml"), "w") as f:
        f.write(str(cfg))
    
    return exp_save_dir


def clean_exp_savedir(exp_save_dir, best_ckpt_path, prefix="bi"):
    """Remove old checkpoints, keep only the best"""
    if not os.path.exists(exp_save_dir):
        return
    
    for file in os.listdir(exp_save_dir):
        if file.startswith(prefix) and file.endswith(".pth"):
            file_path = os.path.join(exp_save_dir, file)
            if file_path != best_ckpt_path:
                os.remove(file_path)
                print(f"Removed old checkpoint: {file_path}")


# ============================================================================
# BURN-IN TRAINING
# ============================================================================

def run_burn_in(cfg: CN, exp_save_dir: str):
    """Run burn-in phase (source-only training)"""
    # Initialize distributed training if available
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda:0')
    
    if rank == 0:
        print("\n" + "="*60)
        print("BURN-IN PHASE: Source-Only Training")
        print("="*60)
    
    # Load data
    source_train_loader, _, source_test_loader, target_test_loader = make_dataset(
        source_dataset=cfg.dataset.source,
        target_dataset=cfg.dataset.target,
        img_size=cfg.img_size,
        train_bs=cfg.burn_in.train_bs,
        eval_bs=cfg.burn_in.eval_bs,
        num_workers=cfg.burn_in.get('num_workers', 4),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    
    # Create model
    model = UModel(
        backbone=cfg.model.backbone.type,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        imgsize=cfg.img_size,
        freeze_backbone=cfg.model.backbone.freeze,
    )
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Setup optimizer
    scaler = GradScaler('cuda')
    optimizer = torch.optim.AdamW([
        {
            "params": list(model_without_ddp.classifier_head_src.parameters()),
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": list(model_without_ddp.visual_prompts_src.parameters()),
            "lr": cfg.optimizer_vr.lr,
            "weight_decay": cfg.optimizer_vr.weight_decay,
        },
    ])

    epochs = cfg.burn_in.epochs
    total_steps = epochs * len(source_train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    criterion_class = nn.CrossEntropyLoss()
    
    # Only create writer on rank 0
    if rank == 0:
        writer = SummaryWriter(exp_save_dir)
    else:
        writer = None

    # Training loop
    best_test_acc = 0
    ckpt_path = None
    
    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if distributed:
            source_train_loader.sampler.set_epoch(epoch)
            
        model.train()
        running_loss = 0.0
        
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(
                source_train_loader,
                total=len(source_train_loader),
                desc=f"Burn-in Epoch {epoch + 1}/{epochs}",
                ncols=100,
            )
        else:
            pbar = source_train_loader

        for batch_idx, source_data in enumerate(pbar):
            current_step = epoch * len(source_train_loader) + batch_idx
            src_data, _, src_labels = source_data
            src_img = src_data.to(device)
            src_labels = src_labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                p_s = model(src_img, vr_branch="src", head_branch="src")
                loss = criterion_class(p_s, src_labels)
                running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Only log on rank 0
            if rank == 0 and writer is not None:
                writer.add_scalar("Source/Train BatchLoss", loss.item(), current_step)

        # Evaluate
        test_loss_src, test_accuracy_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device, distributed=distributed
        )
        test_loss_tgt, test_accuracy_tgt = evaluate(
            model, branch="src", test_loader=target_test_loader, device=device, distributed=distributed
        )
        
        # Only log and print on rank 0
        if rank == 0:
            if writer is not None:
                writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
                writer.add_scalar("Source/Test Accuracy", test_accuracy_src, epoch)
                writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
                writer.add_scalar("Target/Test Accuracy", test_accuracy_tgt, epoch)

            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Source Loss: {test_loss_src:.4f}, Source Acc: {test_accuracy_src:.2f}% | "
                  f"Target Loss: {test_loss_tgt:.4f}, Target Acc: {test_accuracy_tgt:.2f}%")

        # Save best checkpoint
        if test_accuracy_src > best_test_acc:
            best_test_acc = test_accuracy_src
            
            if rank == 0:
                ckpt_path = os.path.join(exp_save_dir, f"bi_best_{test_accuracy_src:.2f}.pth")
                torch.save({
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }, ckpt_path)
                print(f"✓ New best checkpoint saved: {ckpt_path}")
            
            if distributed:
                dist.barrier()
                
            if test_accuracy_src == 100:
                break
    
    # Cleanup
    if rank == 0:
        clean_exp_savedir(exp_save_dir, ckpt_path, prefix="bi")
    
    if distributed:
        dist.barrier()
        
    return ckpt_path if rank == 0 else None


# ============================================================================
# DOMAIN ADAPTATION TRAINING
# ============================================================================

def run_domain_adapt(cfg: CN, exp_save_dir: str, best_bi_ckpt: str):
    """Run domain adaptation phase"""
    # Initialize distributed training if available
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda:0')
    
    if rank == 0:
        print("\n" + "="*60)
        print("DOMAIN ADAPTATION PHASE")
        print("="*60)
    
    # Load data
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = make_dataset(
        source_dataset=cfg.dataset.source,
        target_dataset=cfg.dataset.target,
        img_size=cfg.img_size,
        train_bs=cfg.domain_adapt.train_bs,
        eval_bs=cfg.domain_adapt.eval_bs,
        num_workers=cfg.domain_adapt.get('num_workers', 4),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    
    n_src = len(source_train_loader)
    n_tgt = len(target_train_loader)
    steps = min(n_src, n_tgt)

    # Create model
    model = UModel(
        backbone=cfg.model.backbone.type,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        imgsize=cfg.img_size,
        freeze_backbone=cfg.model.backbone.freeze,
    )

    # Load checkpoint
    if best_bi_ckpt is not None:
        if rank == 0:
            print(f"Loading checkpoint: {best_bi_ckpt}")
        ckpt = torch.load(best_bi_ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
    
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Setup optimizer
    scaler = GradScaler('cuda')
    optimizer = torch.optim.AdamW([
        {
            "params": list(model_without_ddp.classifier_head_src.parameters())
            + list(model_without_ddp.classifier_head_tgt.parameters()),
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": model_without_ddp.domain_discriminator.parameters(),
            "lr": cfg.optimizer_d.lr,
            "weight_decay": cfg.optimizer_d.weight_decay,
        },
        {
            "params": list(model_without_ddp.visual_prompts_src.parameters())
            + list(model_without_ddp.visual_prompts_tgt.parameters()),
            "lr": cfg.optimizer_vr.lr,
            "weight_decay": cfg.optimizer_vr.weight_decay,
        },
    ])

    epochs = cfg.domain_adapt.epochs
    total_steps = epochs * steps
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Only create writer on rank 0
    if rank == 0:
        writer = SummaryWriter(exp_save_dir)
    else:
        writer = None

    # Initialize target branch with source weights
    with torch.no_grad():
        model_without_ddp.visual_prompts_tgt.load_state_dict(
            model_without_ddp.visual_prompts_src.state_dict(), strict=False
        )
        model_without_ddp.classifier_head_tgt.load_state_dict(
            model_without_ddp.classifier_head_src.state_dict(), strict=False
        )

    # Training loop
    best_test_acc = 0
    ckpt_path = None
    
    for epoch in range(epochs):
        # Set epoch for distributed samplers
        if distributed:
            source_train_loader.sampler.set_epoch(epoch)
            target_train_loader.sampler.set_epoch(epoch)
            
        running_loss = 0.0
        model.train()
        
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(
                zip(source_train_loader, target_train_loader),
                total=steps,
                desc=f"DA Epoch {epoch + 1}/{epochs}",
                ncols=100,
            )
        else:
            pbar = zip(source_train_loader, target_train_loader)
            
        for batch_idx, (source_data, target_data) in enumerate(pbar):
            current_step = epoch * steps + batch_idx
            grl_alpha = 2.0 / (1.0 + np.exp(-10 * (current_step / total_steps))) - 1.0
            
            _, src_k_data, src_labels = source_data
            tgt_q_data, tgt_k_data, _ = target_data

            src_img = src_k_data.to(device)
            src_labels = src_labels.to(device)
            tgt_k_img = tgt_k_data.to(device)
            tgt_q_img = tgt_q_data.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                # Source classification loss
                logit_s = model(src_img, vr_branch="src", head_branch="src")
                loss_cls = criterion(logit_s, src_labels)

                # Self-supervised learning on target
                logit_t_q = model(tgt_q_img, vr_branch="tgt", head_branch="src")
                logit_t_k = model(tgt_k_img, vr_branch="tgt", head_branch="tgt")

                with torch.no_grad():
                    probs_t_q = F.softmax(logit_t_q, dim=1)
                    pseudo_labels = probs_t_q.argmax(dim=1)

                loss_ssl = F.cross_entropy(logit_t_k, pseudo_labels.detach(), reduction="mean")
                loss_div = F.kl_div(
                    F.log_softmax(logit_t_k, dim=1), 
                    probs_t_q.detach(), 
                    reduction="batchmean"
                )

                # Adversarial domain loss
                d_s = model(src_img, vr_branch="src", head_branch="domain")
                d_t = model(tgt_k_img, vr_branch="tgt", head_branch="domain", grl_alpha=grl_alpha)

                s_labels = torch.zeros(d_s.shape[0], dtype=torch.long, device=device)
                t_labels = torch.ones(d_t.shape[0], dtype=torch.long, device=device)

                loss_adv = criterion(d_s, s_labels) + criterion(d_t, t_labels)
                
                # Total loss
                alpha_div = cfg.get('alpha_div', 0.15)
                alpha_adv = cfg.get('alpha_adv', 0.05)
                alpha_ssl = cfg.get('alpha_ssl', 0.15)
                
                loss = (
                    loss_cls
                    + alpha_div * loss_div
                    + alpha_adv * loss_adv
                    + alpha_ssl * loss_ssl
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging - only on rank 0
            if rank == 0 and writer is not None and batch_idx % 10 == 0:
                writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
                writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
                writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
                writer.add_scalar("DA/Train Div loss", loss_div.item(), current_step)
                
        scheduler.step()
        
        # Evaluate
        test_loss_src, test_acc_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device, distributed=distributed
        )
        test_loss_tgt, test_acc_tgt = evaluate(
            model, branch="tgt", test_loader=target_test_loader, device=device, distributed=distributed
        )

        # Only log and print on rank 0
        if rank == 0:
            if writer is not None:
                writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
                writer.add_scalar("Source/Test Accuracy", test_acc_src, epoch)
                writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
                writer.add_scalar("Target/Test Accuracy", test_acc_tgt, epoch)

            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Source: {test_acc_src:.2f}% | Target: {test_acc_tgt:.2f}%")

        # Save best checkpoint
        if test_acc_tgt > best_test_acc:
            best_test_acc = test_acc_tgt
            
            if rank == 0:
                ckpt_path = os.path.join(exp_save_dir, f"da_best_{test_acc_tgt:.2f}.pth")
                torch.save({
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }, ckpt_path)
                print(f"✓ New best checkpoint saved: {ckpt_path}")
            
            if distributed:
                dist.barrier()
    
    # Cleanup
    if rank == 0:
        clean_exp_savedir(exp_save_dir, ckpt_path, prefix="da")
    
    if distributed:
        dist.barrier()
        
    return ckpt_path if rank == 0 else None


# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

def init_distributed_mode():
    """Initialize distributed training mode if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    
    print(f'Distributed init (rank {rank}/{world_size}): local_rank={local_rank}', flush=True)
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="full",
        choices=["burn_in", "domain_adapt", "full", "test"],
        help="Training mode: burn_in, domain_adapt, full, or test"
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for domain_adapt or test mode")
    
    args = parser.parse_args()
    
    # Load config
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    
    # Initialize distributed training
    distributed, rank, world_size, local_rank = init_distributed_mode()
    
    # Setup experiment directory
    if rank == 0:
        print("\n" + "="*60)
        print(f"Experiment: {cfg.exp_tags}")
        print(f"Mode: {args.mode}")
        print(f"Distributed: {distributed} (rank {rank}/{world_size})")
        print("="*60)
        exp_save_dir = setup(cfg)
        print(f"Save directory: {exp_save_dir}\n")
    else:
        exp_save_dir = None
    
    # Broadcast exp_save_dir to all ranks
    if distributed:
        exp_save_dir_list = [exp_save_dir] if rank == 0 else [None]
        dist.broadcast_object_list(exp_save_dir_list, src=0)
        exp_save_dir = exp_save_dir_list[0]
    
    try:
        # Run requested mode
        if args.mode == "burn_in":
            best_ckpt = run_burn_in(cfg, exp_save_dir)
            if rank == 0:
                print(f"\n✓ Burn-in completed. Best checkpoint: {best_ckpt}")
                
        elif args.mode == "domain_adapt":
            if not args.checkpoint:
                raise ValueError("--checkpoint required for domain_adapt mode")
            best_ckpt = run_domain_adapt(cfg, exp_save_dir, args.checkpoint)
            if rank == 0:
                print(f"\n✓ Domain adaptation completed. Best checkpoint: {best_ckpt}")
                
        elif args.mode == "full":
            # Run both phases
            best_ckpt = run_burn_in(cfg, exp_save_dir)
            
            # Broadcast checkpoint path
            if distributed:
                ckpt_list = [best_ckpt] if rank == 0 else [None]
                dist.broadcast_object_list(ckpt_list, src=0)
                best_ckpt = ckpt_list[0]
            
            if rank == 0:
                print(f"\n✓ Burn-in completed. Starting domain adaptation...")
            
            best_ckpt = run_domain_adapt(cfg, exp_save_dir, best_ckpt)
            if rank == 0:
                print(f"\n✓ Full pipeline completed. Final checkpoint: {best_ckpt}")
                
        elif args.mode == "test":
            if not args.checkpoint:
                raise ValueError("--checkpoint required for test mode")
            
            # Load model and run evaluation
            device = torch.device(f"cuda:{local_rank}" if distributed else cfg.device)
            model = UModel(
                backbone=cfg.model.backbone.type,
                hidden_dim=cfg.model.backbone.hidden_dim,
                out_dim=cfg.dataset.num_classes,
                imgsize=cfg.img_size,
                freeze_backbone=cfg.model.backbone.freeze,
            )
            
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            
            if distributed:
                model = DDP(model, device_ids=[local_rank])
            
            # Load test data
            _, _, _, target_test_loader = make_dataset(
                source_dataset=cfg.dataset.source,
                target_dataset=cfg.dataset.target,
                img_size=cfg.img_size,
                train_bs=cfg.domain_adapt.train_bs,
                eval_bs=cfg.domain_adapt.eval_bs,
                num_workers=cfg.domain_adapt.get('num_workers', 4),
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
            
            test_loss, test_acc = evaluate(
                model, branch="tgt", test_loader=target_test_loader, 
                device=device, distributed=distributed
            )
            
            if rank == 0:
                print(f"\n✓ Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()

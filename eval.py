import torch
import torch.nn as nn
import torch.distributed as dist
from yacs.config import CfgNode as CN
from model import UModel
from data import make_dataset
import argparse
from tqdm import tqdm

def evaluate(model, branch, test_loader, device, distributed=False):
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
            pbar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating {branch} branch", ncols=100)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file",)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)

    model = UModel(
        backbone=cfg.model.backbone.type,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        num_res_blocks=cfg.model.backbone.num_res_blocks,
        imgsize=cfg.img_size,
    )
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)
    model.eval()
    _, _, _, target_test_loader = (
        make_dataset(
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            img_size=cfg.img_size,
            train_bs=cfg.domain_adapt.train_bs,
            eval_bs=cfg.domain_adapt.eval_bs,
            num_workers=cfg.domain_adapt.num_workers,
        )
    )
    avg_loss, accuracy = evaluate(model, branch="target", test_loader=target_test_loader, device=cfg.device)
    print(f"Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
from model import VirDA_model
from data import make_dataset
import argparse


def evaluate(model, branch, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images, branch=branch, inf_type="det", out_type="logits")
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file",)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)

    model = VirDA_model(
        backbone=cfg.model.backbone.type,
        in_dim=cfg.model.backbone.in_dim,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        num_res_blocks=cfg.model.backbone.num_res_blocks,
        imgsize=cfg.img_size,
        patch_size=cfg.model.patch_size,
        attribute_layers=cfg.model.attribute_layers,
        p_vr_src=cfg.model.source.vr_dropout,
        p_vr_tgt=cfg.model.target.vr_dropout,
        p_cls_src=cfg.model.source.cls_dropout,
        p_cls_tgt=cfg.model.target.cls_dropout,
    )
    checkpoint = torch.load(args.ckpt, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)
    model.eval()
    _, _, _, target_test_loader = (
        make_dataset(
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            img_size=cfg.img_size,
            train_bs=cfg.domain_adapt.train_bs,
            eval_bs=128,
            num_workers=cfg.domain_adapt.num_workers,
        )
    )
    avg_loss, accuracy = evaluate(model, branch="tgt", test_loader=target_test_loader, device=cfg.device)
    print(f"Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
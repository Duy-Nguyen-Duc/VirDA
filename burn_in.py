import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from base_model import BaseClassifier
from data import make_dataset
from eval import evaluate
from torch_utils import compute_soft_alpha, freeze_layers


def run_bi_step(cfg: CN, exp_save_dir: str):
    source_train_loader, _, source_test_loader, target_test_loader = make_dataset(
        source_dataset=cfg.dataset.source,
        target_dataset=cfg.dataset.target,
        img_size=cfg.img_size,
        train_bs=cfg.dataset.train_bs,
        eval_bs=cfg.dataset.eval_bs,
    )
    model = BaseClassifier(
        backbone=cfg.model.backbone.type,
        in_dim=cfg.model.backbone.in_dim,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        imgsize=cfg.img_size,
        patch_size=cfg.model.patch_size,
        attribute_layers=cfg.model.attribute_layers,
        p_vr_src=cfg.model.source.vr_dropout,
        p_vr_tgt=cfg.model.target.vr_dropout,
        p_cls_src=cfg.model.source.cls_dropout,
        p_cls_tgt=cfg.model.target.cls_dropout,
    )
    device = torch.device(cfg.device)
    model = model.to(device)
    scaler = GradScaler(cfg.device)
    vr_params = list(model.visual_prompt_src.parameters()) + list(
        model.visual_prompt_tgt.parameters()
    )
    params = [p for n, p in model.named_parameters() if "visual_prompt" not in n]
    optimizer = optim.AdamW(
        params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    optimizer_vr = optim.AdamW(
        vr_params, lr=cfg.optimizer_vr.lr, weight_decay=cfg.optimizer_vr.weight_decay
    )

    epochs = cfg.epochs
    total_steps = epochs * len(source_train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scheduler_vr = CosineAnnealingLR(optimizer_vr, T_max=total_steps, eta_min=1e-5)
    criterion_class = nn.CrossEntropyLoss()
    writer = SummaryWriter(exp_save_dir)

    # training script
    best_test_acc = 0
    if cfg.model.backbone.freeze:
        freeze_layers([model.backbone])
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            source_train_loader,
            total=len(source_train_loader),
            desc=f"Epoch {epoch+1}",
            ncols=100,
        )

        for batch_idx, source_data in enumerate(pbar):
            pbar.set_description_str(f"Epoch {epoch+1}", refresh=True)
            current_step = epoch * len(source_train_loader) + batch_idx
            # weak_img, strong_img, label
            src_q_data, src_k_data, src_labels = source_data

            src_q_data = src_q_data.to(device)
            src_k_data = src_k_data.to(device)
            src_labels = src_labels.to(device)

            optimizer.zero_grad()
            optimizer_vr.zero_grad()
            with autocast(device_type=cfg.device):
                p_s_q, u_s_q = model(
                    src_q_data,
                    branch="src",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.tau,
                )
                p_s_k, u_s_k = model(
                    src_k_data,
                    branch="src",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.tau,
                )

                loss_cls = criterion_class(p_s_k, src_labels)
                loss_uncertainty = (u_s_q.mean() - u_s_k.mean()).pow(2)

                p_s_k, p_s_q = p_s_k.clamp(1e-6), p_s_q.detach().clamp(1e-6)
                kl_term = F.kl_div(p_s_k.log(), p_s_q, reduction="none").sum(dim=1)
                w_conf = compute_soft_alpha(u_s_q)

                loss_div = (kl_term * w_conf).mean()
                loss = loss_cls + 0.25 * loss_uncertainty + 0.1 * loss_div
                running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_vr)
            scaler.update()
            scheduler.step()
            scheduler_vr.step()

            writer.add_scalar("Source/Train Cls loss", loss_cls.item(), current_step)
            writer.add_scalar(
                "Source/Train Unt loss", loss_uncertainty.item(), current_step
            )
            writer.add_scalar("Source/Train Div loss", loss_div.item(), current_step)
            writer.add_scalar("Source/Train BatchLoss", loss.item(), current_step)
            writer.add_scalar(
                "Source/Running loss",
                running_loss / len(source_train_loader),
                current_step,
            )

        test_loss_src, test_accuracy_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device
        )
        test_loss_tgt, test_accuracy_tgt = evaluate(
            model, branch="src", test_loader=target_test_loader, device=device
        )
        writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
        writer.add_scalar("Source/Test Accuracy", test_accuracy_src, epoch)

        writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
        writer.add_scalar("Target/Test Accuracy", test_accuracy_tgt, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] Test Loss Source: {test_loss_src:.4f}, Test Accuracy Source: {test_accuracy_src:.2f}%"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] Test Loss Target: {test_loss_tgt:.4f}, Test Accuracy Target: {test_accuracy_tgt:.2f}%"
        )

        if test_accuracy_src > best_test_acc:
            best_test_acc = test_accuracy_src
            ckpt_path = os.path.join(
                exp_save_dir, f"bi_best_{test_accuracy_src:.2f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                ckpt_path,
            )
            print(f"New best checkpoint saved: {ckpt_path}")
        return ckpt_path

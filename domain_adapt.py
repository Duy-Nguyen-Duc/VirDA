import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from data import make_dataset
from eval import evaluate
from model import VirDA_model
from torch_nn import Classifier
from torch_utils import (
    compute_soft_alpha,
    freeze_layers,
    grad_reverse,
    decay_thresholds,
)

import argparse

from utils import setup, clean_exp_savedir


def run_da_step(cfg: CN, exp_save_dir: str, best_bi_ckpt: str):
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = (
        make_dataset(
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            img_size=cfg.img_size,
            train_bs=cfg.domain_adapt.train_bs,
            eval_bs=cfg.domain_adapt.eval_bs,
            num_workers=cfg.domain_adapt.num_workers,
        )
    )
    n_src = len(source_train_loader)
    n_tgt = len(target_train_loader)
    steps = min(n_src, n_tgt)

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

    domain_classifier = Classifier(
        in_dim=cfg.model.backbone.in_dim,
        hidden_dim=cfg.domain_dis.hidden_dim,
        out_dim=2,
        num_res_blocks=cfg.domain_dis.num_res_blocks,
        dropout=cfg.domain_dis.dropout,
    )
    device = torch.device(cfg.device)
    ckpt = torch.load(best_bi_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    domain_classifier = domain_classifier.to(device)

    scaler = GradScaler()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.classifier_head_src.parameters())
                + list(model.classifier_head_tgt.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": domain_classifier.parameters(),
                "lr": cfg.optimizer_d.lr,
                "weight_decay": cfg.optimizer_d.weight_decay,
            },
            {
                "params": list(model.visual_prompt_src.parameters())
                + list(model.visual_prompt_tgt.parameters()),
                "lr": cfg.optimizer_vr.lr,
                "weight_decay": cfg.optimizer_vr.weight_decay,
            },
        ]
    )

    epochs = cfg.domain_adapt.epochs
    total_steps = epochs * steps
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss(reduction="none")
    writer = SummaryWriter(exp_save_dir)

    best_test_acc = 0
    if cfg.model.backbone.freeze:
        freeze_layers([model.backbone])

    # Init with same weights
    with torch.no_grad():
        model.visual_prompt_tgt.load_state_dict(
            model.visual_prompt_src.state_dict(), strict=False
        )
        model.classifier_head_tgt.load_state_dict(
            model.classifier_head_src.state_dict(), strict=False
        )

    alignment_mode = cfg.alignment_mode
    # threshold
    unc_threshold = decay_thresholds(
        cfg.domain_adapt.unc_thres_start, cfg.domain_adapt.unc_thres_end, total_steps
    )
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        domain_classifier.train()
        pbar = tqdm(
            zip(source_train_loader, target_train_loader),
            total=steps,
            desc=f"Epoch {epoch + 1}",
            ncols=100,
        )
        for batch_idx, (source_data, target_data) in enumerate(pbar):
            pbar.set_description_str(f"Epoch {epoch + 1}", refresh=True)
            current_step = epoch * steps + batch_idx
            grl_alpha = 2.0 / (1.0 + np.exp(-10 * (current_step / total_steps))) - 1.0
            t_u = unc_threshold[current_step]
            _, src_k_data, src_labels = source_data
            tgt_q_data, tgt_k_data, _ = target_data

            src_img = src_k_data.to(device)
            src_labels = src_labels.to(device)

            tgt_k_img = tgt_k_data.to(device)  # strong
            tgt_q_img = tgt_q_data.to(device)  # weak
            optimizer.zero_grad()
            with autocast():
                # 1. Source cls loss
                p_s, u_s = model(
                    src_img,
                    branch="src",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.model.source.tau_s,
                )
                loss_cls = criterion_class(p_s, src_labels)
                del p_s, src_labels

                # 2. Uncertainty loss
                p_t_q, u_t_q = model(
                    tgt_q_img,
                    branch="tgt_mix",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.model.source.tau_t,
                )
                loss_unc = (u_s.mean() - u_t_q.mean()).pow(2)

                # 3. SSL loss
                p_t_k, u_t_k = model(
                    tgt_k_img,
                    branch="tgt",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.model.target.tau,
                )

                p_t_k, p_t_q = p_t_k.clamp(1e-6), p_t_q.clamp(1e-6)
                with torch.no_grad():
                    pseudo_labels = p_t_q.argmax(dim=1)
                    if alignment_mode == "soft":
                        w_soft = compute_soft_alpha(u_t_q).float()
                        mask = torch.ones_like(u_t_q, dtype=torch.bool)
                    elif alignment_mode == "hard":
                        w_soft = torch.ones_like(u_t_q).float()
                        mask = u_t_q < t_u
                    w_s = compute_soft_alpha(u_s)
                    w_t = compute_soft_alpha(u_t_k)
                del u_t_k, u_s

                if mask.sum().item() == 0:
                    loss_ssl = torch.tensor(0.0, dtype=torch.long, device=device)
                    loss_div = torch.tensor(0.0, dtype=torch.long, device=device)
                else:
                    # 3. SSL loss
                    loss_ssl = (
                        F.nll_loss(
                            p_t_k.log()[mask], pseudo_labels[mask], reduction="none"
                        )
                        * w_soft[mask]
                    ).mean()

                    # 4. KL divergencence
                    loss_div = (
                        F.kl_div(
                            p_t_k.log()[mask], p_t_q.detach()[mask], reduction="none"
                        ).sum(dim=1)
                        * w_soft[mask]
                    ).mean()
                del p_t_k, p_t_q

                # 5. Adv loss
                f_s = model(src_img, branch="src", inf_type="det", out_type="feat")
                f_t = model(tgt_k_img, branch="tgt", inf_type="det", out_type="feat")

                if alignment_mode == "soft":
                    d_s = domain_classifier(f_s.detach())
                elif alignment_mode == "hard":
                    d_s = domain_classifier(grad_reverse(f_s, grl_alpha))
                d_t = domain_classifier(grad_reverse(f_t, grl_alpha))

                s_labels = torch.zeros(d_s.shape[0], dtype=torch.long, device=device)
                t_labels = torch.ones(d_t.shape[0], dtype=torch.long, device=device)

                per_s = criterion_domain(d_s, s_labels)
                per_t = criterion_domain(d_t, t_labels)
                loss_adv = (per_s * w_s).mean() + (per_t * w_t).mean()
                loss = (
                    loss_cls
                    + cfg.alpha_unc * loss_unc
                    + cfg.alpha_div * loss_div
                    + cfg.alpha_adv * loss_adv
                    + cfg.alpha_ssl * loss_ssl
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            del f_s, f_t

            # Logging
            writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
            writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
            writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
            writer.add_scalar("DA/Train Unt loss", loss_unc.item(), current_step)
            writer.add_scalar("DA/Train Div loss", loss_div.item(), current_step)
            writer.add_scalar("DA/Train BatchLoss", loss.item(), current_step)
        scheduler.step()
        test_loss_src, test_acc_src = evaluate(
            model, branch="src", test_loader=source_test_loader, device=device
        )
        test_loss_tgt, test_acc_tgt = evaluate(
            model, branch="tgt", test_loader=target_test_loader, device=device
        )

        writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
        writer.add_scalar("Source/Test Accuracy", test_acc_src, epoch)
        writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
        writer.add_scalar("Target/Test Accuracy", test_acc_tgt, epoch)
        writer.add_scalar(
            "DA/Epoch loss", running_loss / len(source_train_loader), epoch
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Source Loss: {test_loss_src:.4f}, Source Acc: {test_acc_src:.2f}%"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Target Loss: {test_loss_tgt:.4f}, Target Acc: {test_acc_tgt:.2f}%"
        )

        # Save the best model checkpoint (including optimizer, scheduler, scaler, etc.)
        if test_acc_tgt > best_test_acc:
            best_test_acc = test_acc_tgt
            ckpt_path = os.path.join(exp_save_dir, f"da_best_{test_acc_tgt:.2f}.pth")

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
    clean_exp_savedir(exp_save_dir, ckpt_path, prefix="da")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint file"
    )
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    exp_save_dir = setup(cfg)

    print("Running DA step")
    best_ckpt = args.ckpt
    print("Loading best checkpoint from burn-in step:", best_ckpt)
    # Run domain adaptation step
    run_da_step(cfg, exp_save_dir=exp_save_dir, best_bi_ckpt=best_ckpt)

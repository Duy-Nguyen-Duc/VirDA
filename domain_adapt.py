import os
import argparse
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast        
from torch.optim.lr_scheduler import CosineAnnealingLR  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from data.data import make_dataset, transform_map
from eval import evaluate
from torch_nn.model import UModel, EigenCAM
from utils import setup, clean_exp_savedir


def run_da_step(cfg: CN, exp_save_dir: str, best_bi_ckpt: str):
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = (
        make_dataset(
            root=cfg.dataset.root,
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            imgsize=cfg.img_size,
            train_bs=cfg.domain_adapt.train_bs,
            eval_bs=cfg.domain_adapt.eval_bs,
            num_workers=cfg.domain_adapt.num_workers,
        )
    )
    n_src = len(source_train_loader)
    n_tgt = len(target_train_loader)
    steps = min(n_src, n_tgt)

    model = UModel(
        backbone=cfg.model.backbone.type,
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=cfg.dataset.num_classes,
        imgsize=cfg.img_size,
        freeze_backbone=cfg.model.backbone.freeze,
    )

    device = torch.device(cfg.device)
    ckpt = torch.load(best_bi_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    cam = EigenCAM(model, target_layer=model.backbone.transformer.blocks[-1].norm2)
    cam.register_hook()

    scaler = GradScaler('cuda')
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.src_cls.parameters()) + list(model.tgt_cls.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": model.domain_discriminator.parameters(),
                "lr": cfg.optimizer_d.lr,
                "weight_decay": cfg.optimizer_d.weight_decay,
            },
            {
                "params": list(model.src_vr.parameters()) + list(model.tgt_vr.parameters()),    
                 "lr": cfg.optimizer_vr.lr,
                "weight_decay": cfg.optimizer_vr.weight_decay,
            },
        ]
    )

    epochs = cfg.domain_adapt.epochs
    total_steps = epochs * steps
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(exp_save_dir)

    # Init with same weights
    with torch.no_grad():
        model.tgt_vr.load_state_dict(
            model.src_vr.state_dict(), strict=False
        )
        model.tgt_cls.load_state_dict(
            model.src_cls.state_dict(), strict=False
        )

    # Training loop
    best_test_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
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
            t_conf = 0.8
            _, src_strong_data, src_labels, _ = source_data
            tgt_weak_data, tgt_strong_data, _, _ = target_data

            src_img = src_strong_data.to(device)
            src_labels = src_labels.to(device)
            tgt_strong_img = tgt_strong_data.to(device)
            tgt_weak_img = tgt_weak_data.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):              
                # 1. Source cls loss
                # Teacher branch for source, student branch for target
                logit_s = model(src_img, vr_branch="src", head_branch="src")
                loss_cls = criterion(logit_s, src_labels)

                # 2. Target weak view forward pass 
                logit_t_weak = model(tgt_weak_img, vr_branch="tgt", head_branch="src")
                #loss_unc = (u_s.mean() - u_t.mean())**2
                
                with torch.no_grad():
                    probs_weak = F.softmax(logit_t_weak, dim=1)
                    conf, pseudo_labels = probs_weak.max(dim=1)
                    mask = (conf > t_conf).float()

                    smap_s = cam(x=src_img, vr_branch="src", head_branch="src")
                    smap_s = transform_map(smap_s, None, transform_params=[0.0,1.0], imgsize=cfg.img_size)
                                
                # 5. Target strong view forward pass
                logit_t_strong = model(tgt_strong_img, vr_branch="tgt", head_branch="tgt")

                # 6. Target strong view SSL loss
                loss_ssl = (F.cross_entropy(logit_t_strong, pseudo_labels, reduction="none") * mask).mean()

                # 7. Target strong view KL divergence loss
                loss_div = F.kl_div(
                    F.log_softmax(logit_t_strong, dim=1),
                    F.softmax(logit_t_weak, dim=1),
                    reduction="batchmean",
                )

                # 8. Adv loss
                d_s = model(src_img, vr_branch="src", head_branch="domain", salience_map=smap_s, grl_alpha=grl_alpha).detach()
                d_t = model(tgt_strong_img, vr_branch="tgt", head_branch="domain", grl_alpha=grl_alpha)

                s_labels = torch.zeros(d_s.shape[0], dtype=torch.long, device=device)
                t_labels = torch.ones(d_t.shape[0], dtype=torch.long, device=device)

                loss_adv = criterion(d_s, s_labels) + criterion(d_t, t_labels)
                loss = (
                    loss_cls
                    # + 0.30 * loss_unc
                    + cfg.alpha_div * loss_div
                    + cfg.alpha_adv * loss_adv
                    + cfg.alpha_ssl * loss_ssl
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            del d_s, d_t

            # Logging
            writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
            # writer.add_scalar("DA/Train Unc loss", loss_unc.item(), current_step)
            writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
            writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
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

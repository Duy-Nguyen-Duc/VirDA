import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast        
from torch.optim.lr_scheduler import CosineAnnealingLR  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode as CN

from data import make_dataset, transform_map
from eval import evaluate
from model import UModel, EigenCAM
from torch_utils import visualize_salience_map
from utils import setup, clean_exp_savedir


def run_da_step(cfg: CN, exp_save_dir: str, best_bi_ckpt: str):
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = (
        make_dataset(
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
    cam = EigenCAM(model, target_layer=model.backbone.transformer.blocks[-1].norm2)
    cam.register_hook()

    device = torch.device(cfg.device)
    ckpt = torch.load(best_bi_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    scaler = GradScaler('cuda')
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.stu_cls.parameters()) + list(model.tch_cls.parameters()),
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": model.domain_discriminator.parameters(),
                "lr": cfg.optimizer_d.lr,
                "weight_decay": cfg.optimizer_d.weight_decay,
            },
            {
                "params": list(model.stu_vr.parameters()) + list(model.tch_vr.parameters()),    
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
        model.tch_vr.load_state_dict(
            model.stu_vr.state_dict(), strict=False
        )
        model.tch_cls.load_state_dict(
            model.stu_cls.state_dict(), strict=False
        )
    #freeze_layers([model.tch_vr, model.tch_cls])

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
            _, src_strong_data, src_labels, _ = source_data
            tgt_weak_data, tgt_strong_data, _, affine_params = target_data

            src_img = src_strong_data.to(device)
            src_labels = src_labels.to(device)
            tgt_strong_img = tgt_strong_data.to(device)
            tgt_weak_img = tgt_weak_data.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                # 1. Source cls loss
                # Teacher branch for source, student branch for target
                logit_s = model(src_img, vr_branch="tch", head_branch="tch")
                loss_cls = criterion(logit_s, src_labels)

                # 2. Target 
                logit_t_weak = model(tgt_weak_img, vr_branch="tch", head_branch="tch")
                
                with torch.no_grad():
                    pseudo_labels = F.softmax(logit_t_weak, dim=1).argmax(dim=1)
                    salience_map_t= cam(x=tgt_weak_img, vr_branch="tch", head_branch="tch")
                    salience_map_t = transform_map(
                        salience_map_t, affine_params, transform_params=[0.0,1.0], 
                        imgsize=cfg.img_size
                    )
                    salience_map_t = salience_map_t.to(device)
                
                logit_t_strong = model(tgt_strong_img, vr_branch="stu", head_branch="stu", saliency_map=salience_map_t)
                
                loss_ssl = criterion(logit_t_strong, pseudo_labels)

                loss_div = F.kl_div(
                    F.log_softmax(logit_t_strong, dim=1),
                    F.softmax(logit_t_weak, dim=1),
                    reduction="batchmean",
                )

                # 4. Adv loss
                d_s = model(src_img, vr_branch="tch", head_branch="domain", grl_alpha=grl_alpha)
                d_t = model(tgt_strong_img, vr_branch="stu", head_branch="domain", saliency_map=salience_map_t, grl_alpha=grl_alpha)

                s_labels = torch.zeros(d_s.shape[0], dtype=torch.long, device=device)
                t_labels = torch.ones(d_t.shape[0], dtype=torch.long, device=device)

                loss_adv = criterion(d_s, s_labels) + criterion(d_t, t_labels)
                loss = (
                    loss_cls
                    + cfg.alpha_div * loss_div
                    + cfg.alpha_adv * loss_adv
                    + cfg.alpha_ssl * loss_ssl
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ema_update(model.stu_vr, model.tch_vr, 0.996)
            # ema_update(model.stu_cls, model.tch_cls, 0.996)
            del d_s, d_t

            # Logging
            writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
            writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
            writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
            writer.add_scalar("DA/Train Div loss", loss_div.item(), current_step)
            writer.add_scalar("DA/Train BatchLoss", loss.item(), current_step)
        scheduler.step()
        test_loss_src, test_acc_src = evaluate(
            model, branch="tch", test_loader=source_test_loader, device=device
        )
        test_loss_tgt, test_acc_tgt = evaluate(
            model, branch="stu", test_loader=target_test_loader, device=device
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
        if (epoch) % 4 == 0:
            # visualize samples (on both source and target)
            # 1. Source samples
            visualize_salience_map(
                "data/OfficeHome/Art/Computer/00014.jpg",
                cam, 
                vr_branch="tch",
                head_branch="tch",
                device=device,
                outpath=os.path.join(exp_save_dir, f"da_epoch{epoch+1}_source_salience.png"),
                img_size=cfg.img_size,
            )
            # 2. Target samples
            visualize_salience_map(
                "data/OfficeHome/Clipart/Computer/00083.jpg",
                cam, 
                vr_branch="stu",
                head_branch="stu",
                device=device,
                outpath=os.path.join(exp_save_dir, f"da_epoch{epoch+1}_target_salience.png"),
                img_size=cfg.img_size,
            )
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

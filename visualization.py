## training step

import os
from itertools import cycle

import numpy as np
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
from eval import evaluate, vis_evaluate
from torch_nn import Classifier
from torch_utils import compute_soft_alpha, freeze_layers, grad_reverse

import argparse

from utils import setup

def vis(cfg: CN, exp_save_dir: str, best_bi_ckpt: str, source_samples_loader, target_samples_loader):
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = (
        make_dataset(
            source_dataset=cfg.dataset.source,
            target_dataset=cfg.dataset.target,
            img_size=cfg.img_size,
            train_bs=cfg.dataset.train_bs,
            eval_bs=cfg.dataset.eval_bs,
            num_workers=cfg.dataset.num_workers,
        )
    )
    model = BaseClassifier(
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
        hidden_dim=cfg.model.backbone.hidden_dim,
        out_dim=2,
        num_res_blocks=0,
        dropout=0.5,
    )
    device = torch.device(cfg.device)
    ckpt = torch.load(best_bi_ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    domain_classifier = domain_classifier.to(device)

    scaler = GradScaler(cfg.device)

    vr_params = list(model.visual_prompt_src.parameters()) + list(
        model.visual_prompt_tgt.parameters()
    )
    domain_params = list(domain_classifier.parameters())
    params = [p for n, p in model.named_parameters() if "visual_prompt" not in n]

    optimizer = optim.AdamW(
        params+domain_params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    optimizer_vr = optim.AdamW(
        vr_params, lr=cfg.optimizer_vr.lr, weight_decay=cfg.optimizer_vr.weight_decay
    )
    epochs = cfg.epochs
    total_steps = epochs * len(source_train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    scheduler_vr = CosineAnnealingLR(optimizer_vr, T_max=total_steps, eta_min=1e-5)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    writer = SummaryWriter(exp_save_dir)

    # Freeze backbone and prepare AMP
    #if cfg.model.backbone.freeze:
    freeze_layers([model.backbone])

    # Init with same weights
    with torch.no_grad():
        model.visual_prompt_tgt.load_state_dict(
            model.visual_prompt_src.state_dict(), strict=False
        )
        model.classifier_head_tgt.load_state_dict(
            model.classifier_head_src.state_dict(), strict=False
        )

    vis_evaluate(
        model,
        branch="src",
        test_loader=source_samples_loader,
        device=device,
        epoch=0,
        vis_root=os.path.join("vis", "src"),
    )

    vis_evaluate(
        model,
        branch="tgt",
        test_loader=target_samples_loader,
        device=device,
        epoch=0,
        vis_root=os.path.join("vis", "tgt"),
    )
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        tgt_cycle = cycle(target_train_loader)
        model.train()
        domain_classifier.train()
        pbar = tqdm(
            source_train_loader,
            total=len(source_train_loader),
            desc=f"Epoch {epoch+1}",
            ncols=100,
        )

        for batch_idx, source_data in enumerate(pbar):
            pbar.set_description_str(f"Epoch {epoch+1}", refresh=True)
            target_data = next(tgt_cycle)
            current_step = epoch * len(source_train_loader) + batch_idx
            grl_alpha = 2.0 / (1.0 + np.exp(-10 * (current_step / total_steps))) - 1.0

            _, src_k_data, src_labels = source_data
            tgt_q_data, tgt_k_data, _ = target_data

            src_img = src_k_data.to(device)
            src_labels = src_labels.to(device)
            tgt_k_img = tgt_k_data.to(device)  # strong
            tgt_q_img = tgt_q_data.to(device)  # weak

            optimizer.zero_grad()
            optimizer_vr.zero_grad()

            with autocast(device_type=cfg.device):
                # 1. Source cls loss
                p_s, u_s = model(
                    src_img,
                    branch="src",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.tau,
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
                    tau=cfg.tau,
                )
                loss_unc = (u_s.mean() - u_t_q.mean()).pow(2)

                # 3. SSL loss
                p_t_k, u_t_k = model(
                    tgt_k_img,
                    branch="tgt",
                    inf_type="mc",
                    out_type="logits",
                    mc_samples=cfg.mc_samples,
                    tau=cfg.tau,
                )
                p_t_k, p_t_q = p_t_k.clamp(1e-6), p_t_q.clamp(1e-6)
                with torch.no_grad():
                    pseudo_labels = p_t_q.argmax(dim=1)
                    w_conf = compute_soft_alpha(u_t_q)
                    w_s = compute_soft_alpha(u_s)
                    w_t = compute_soft_alpha(u_t_k)
                del u_t_k, u_t_q, u_s
                loss_ssl = (
                    F.nll_loss(p_t_k.log(), pseudo_labels, reduction="none") * w_conf
                ).mean()

                # 4. KL divergence loss
                kl_term = F.kl_div(p_t_k.log(), p_t_q.detach(), reduction="none").sum(dim=1)
                loss_div = (kl_term * w_conf).mean()
                del p_t_k, p_t_q

                # 5. Adv loss
                f_s = model(src_img, branch="src", inf_type="det", out_type="feat")
                f_t = model(tgt_k_img, branch="tgt", inf_type="det", out_type="feat")

                d_s = domain_classifier(f_s.detach())
                d_t = domain_classifier(grad_reverse(f_t, grl_alpha))

                loss_adv_s = (criterion_domain(d_s, torch.zeros_like(d_s)) * w_s).mean()
                loss_adv_t = (criterion_domain(d_t, torch.ones_like(d_t)) * w_t).mean()
                loss_adv = loss_adv_s + loss_adv_t

                loss = (
                    loss_cls
                    + 0.25 * loss_unc
                    + 0.15 * loss_div
                    + 0.02 * loss_adv
                    + 0.15 * loss_ssl
                )

            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_vr)
            scaler.update()
            scheduler.step()
            scheduler_vr.step()

            del f_s, f_t

            # Logging
            writer.add_scalar("DA/Train Cls loss", loss_cls.item(), current_step)
            writer.add_scalar("DA/Train Adv loss", loss_adv.item(), current_step)
            writer.add_scalar("DA/Train Ssl loss", loss_ssl.item(), current_step)
            writer.add_scalar("DA/Train Unt loss", loss_unc.item(), current_step)
            writer.add_scalar("DA/Train Div loss", loss_div.item(), current_step)
            writer.add_scalar("DA/Train BatchLoss", loss.item(), current_step)

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
            f"Epoch [{epoch+1}/{epochs}] "
            f"Source Loss: {test_loss_src:.4f}, Source Acc: {test_acc_src:.2f}%"
        )
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Target Loss: {test_loss_tgt:.4f}, Target Acc: {test_acc_tgt:.2f}%"
        )
        vis_evaluate(
            model,
            branch="src",
            test_loader=source_samples_loader,
            device=device,
            epoch=epoch+1,
            vis_root=os.path.join("vis", "src"),)

        vis_evaluate(
            model,
            branch="tgt",
            test_loader=target_samples_loader,
            device=device,
            epoch=epoch+1,
            vis_root=os.path.join("vis", "tgt"),)


if __name__=="__main__":
    from torch.utils.data import DataLoader, Subset
    from data import StrongWeakAugDataset

    source_dataset = "mnist"
    target_dataset = "usps"
    img_size=64
    source_test_data = StrongWeakAugDataset(
            dataset_name=source_dataset, root="./data", img_size=img_size, train=False
        )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root="./data", img_size=img_size, train=False
    )

    vis_src_dataset = Subset(source_test_data, [4824,6783,1514,18,736,3558,3762,7260,3062,4224])
    vis_src_loader = DataLoader(
        vis_src_dataset,
        batch_size=10, 
        shuffle=False,     
        num_workers=4, 
    )

    vis_tgt_dataset = Subset(target_test_data, [1096,1019, 1316,122,265,793,698,1118,198,1095])
    vis_tgt_loader = DataLoader(
        vis_tgt_dataset,
        batch_size=10, 
        shuffle=False,     
        num_workers=4, 
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint file"
    )
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    exp_save_dir = setup(cfg)

    print("Running Vis step")
    print("Experiment name:", cfg.exp_tags)
    best_ckpt = args.ckpt
    print("Loading best checkpoint from burn-in step:", best_ckpt)
    # Run domain adaptation step
    vis(cfg, exp_save_dir=exp_save_dir, best_bi_ckpt=best_ckpt, source_samples_loader=vis_src_loader, target_samples_loader=vis_tgt_loader)
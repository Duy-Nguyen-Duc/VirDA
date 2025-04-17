# Source data train only
from src.data import source_train_loader, source_test_loader, target_train_loader, target_test_loader
from classifier_model import BaseClassifier 
from src.layers.torch_nn import Classifier

from src.layers.utils import freeze_layers
from src.eval import evaluate, evaluate_wo_vr
import torch
import torch.nn as nn
import torch.optim as optim
from src.layers.grl import grad_reverse
from src.layers.utils import update_ema
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import os

# setup
epochs = 50
stu_model = BaseClassifier()
stu_model.classifier_head = nn.Identity()
tch_model = BaseClassifier()
domain_classifier = Classifier(
    in_dim=512, hidden_dim=256, out_dim=2, num_res_blocks=2, dropout=0.5
)
device = torch.device("cuda:1")
stu_model = stu_model.to(device)
tch_model = tch_model.to(device)
domain_classifier = domain_classifier.to(device)

layers = (
    [param for name, param in stu_model.named_parameters() if "visual_prompt" not in name] +
    [param for name, param in tch_model.named_parameters() if "visual_prompt" not in name] +
    list(domain_classifier.parameters())
)

optimizer = optim.AdamW(layers, lr=0.01)
vr_layers = list(stu_model.visual_prompt.parameters()) + list(tch_model.visual_prompt.parameters())
optimizer_vr = optim.AdamW(vr_layers, lr=0.001)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

log_dir = os.path.join("runs", "da_exp_v5")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

#training script
best_test_acc = 0
freeze_layers([stu_model.backbone, tch_model.backbone])
total_steps = epochs * len(source_train_loader)
for epoch in tqdm(range(epochs), desc="Overall Epochs", total=epochs):
    stu_model.train()
    tch_model.train()
    domain_classifier.train()
    running_loss = 0.0

    if epoch <= 30:
        keep_rate = 0.9996
    elif 30 < epoch <= 60:
        keep_rate = 0.9998
    else:
        keep_rate = 1.0
    
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        current_step = epoch * len(source_train_loader) + batch_idx

        src_q_data, src_k_data, src_labels = source_data
        tgt_q_data, tgt_k_data, _ = target_data

        src_img = src_k_data.to(device)
        src_labels = src_labels.to(device)
        tgt_img = tgt_k_data.to(device)

        optimizer.zero_grad()

        class_logits = tch_model(src_img, output_type="logits")
        loss_class = criterion_class(class_logits, src_labels)

        p = current_step / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        # Domain loss
        src_feat = tch_model(src_img, output_type="feat")
        tgt_feat = grad_reverse(stu_model(tgt_img, output_type="feat"), alpha = alpha)

        domain_src_logits = domain_classifier(src_feat.detach())
        domain_tgt_logits = domain_classifier(tgt_feat)

        domain_labels_source = torch.zeros(src_img.size(0), dtype=torch.long).to(device)
        domain_labels_target = torch.ones(tgt_img.size(0), dtype=torch.long).to(device)
        domain_logits = torch.cat([domain_src_logits, domain_tgt_logits], dim=0)
        domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
        loss_domain = criterion_domain(domain_logits, domain_labels)

        loss = loss_class + 0.001 * loss_domain
        
        loss.backward()
        optimizer.step()
        optimizer_vr.step()

        update_ema(tch_model.visual_prompt,stu_model.visual_prompt, decay=keep_rate)

        writer.add_scalar("Burn-in/Train Cls loss", loss_class.item(), current_step)
        writer.add_scalar("Burn-in/Train Dis loss", loss_domain.item(), current_step)
        writer.add_scalar("Burn-in/Train BatchLoss", loss.item(), current_step)

    test_loss_src, test_accuracy_src = evaluate(tch_model, test_loader=source_test_loader, device=device)
    test_loss_tgt, test_accuracy_tgt = evaluate(tch_model, test_loader=target_test_loader, device=device)
    test_loss_tgt_wo_vr, test_accuracy_tgt_wo_vr = evaluate_wo_vr(tch_model, test_loader=target_test_loader, device=device)

    writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
    writer.add_scalar("Source/Test Accuracy", test_accuracy_src, epoch)

    writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
    writer.add_scalar("Target/Test Accuracy", test_accuracy_tgt, epoch)

    writer.add_scalar("Target/Test EpochLoss (w/o VR)", test_loss_tgt_wo_vr, epoch)
    writer.add_scalar("Target/Test Accuracy (w/o VR)", test_accuracy_tgt_wo_vr, epoch)    

    print(
        f"Epoch [{epoch + 1}/{epochs}] Test Loss Source: {test_loss_src:.4f}, Test Accuracy Source: {test_accuracy_src:.2f}%"
    )
    print(
        f"Epoch [{epoch + 1}/{epochs}] Test Loss Target: {test_loss_tgt:.4f}, Test Accuracy Target: {test_accuracy_tgt:.2f}%"
    )
    print(
        f"Epoch [{epoch + 1}/{epochs}] Test Loss Target (w/o VR): {test_loss_tgt_wo_vr:.4f}, Test Accuracy Target (w/o VR): {test_accuracy_tgt_wo_vr:.2f}%"
    )

    # Save the best model based on test accuracy.
    if  test_accuracy_src > best_test_acc:
        best_test_acc = test_accuracy_src
        best_checkpoint_path = os.path.join("checkpoints", "best_model_v5.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(tch_model.state_dict(), best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}]: New best model saved with test accuracy: {test_accuracy_src:.2f}%"
        )
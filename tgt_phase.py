import os

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.da_model_v2 import DA_model_v2
from src.data import source_train_loader, target_train_loader, target_test_loader
from src.layers.kl_div import kl_divergence_loss
from src.layers.utils import update_ema

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_tch = DA_model_v2().to(device)
model_stu = DA_model_v2().to(device)

num_epochs_da = 100
batch_size = 256
steps_per_epoch = len(source_train_loader)
total_steps = num_epochs_da * steps_per_epoch

tgt_train_optimizer = optim.AdamW(model_stu.parameters(), lr=0.001)
tgt_train_scheduler = optim.lr_scheduler.MultiStepLR(
    tgt_train_optimizer, milestones=[int(0.5 * num_epochs_da), int(0.72 * num_epochs_da)], gamma=0.1
)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

log_dir = os.path.join("runs", "domain_adaptation_experiment")
writer = SummaryWriter(log_dir)

best_test_tgt_acc = 0.0
global_step = 0


def evaluate(model, test_loader, branch, device):
    """Evaluate model accuracy on the MNIST test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # During evaluation, we need only the classification head.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Set alpha = 0 so that GRL does not affect the output.
            class_logits = model(images, None, None, branch=branch)
            loss = criterion(class_logits, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

for epoch in range(num_epochs_da):
    best_checkpoint_path = os.path.join("checkpoints/v2", "best_model.pth")
    model_tch.load_state_dict(torch.load(best_checkpoint_path))
    model_stu.load_state_dict(torch.load(best_checkpoint_path))
    model_stu.train()
    running_loss = 0.0

    for batch_idx, target_data in enumerate(target_train_loader):

        current_step = epoch * len(target_train_loader) + batch_idx
        alpha = (2.0 / (1.0 + np.exp(-10 * (current_step / total_steps)))) - 1
        tgt_q_data, tgt_k_data, _ = target_data

        tgt_q_data = tgt_q_data.to(device)
        tgt_k_data = tgt_k_data.to(device)

        tgt_train_optimizer.zero_grad()

        tgt_q_logits, tgt_k_logits = model_stu(
            tgt_q_data, tgt_k_data, alpha, branch="tgt_train"
        )

        with torch.no_grad():
            probs = F.softmax(tgt_q_logits, dim=1)
            confidences, pseudo_labels = torch.max(probs, dim=1)
            idx = confidences > 0.7 # confidence threshold

        pseudo_labels = pseudo_labels[idx]
        tgt_k_logits = tgt_k_logits[idx]
        tgt_q_logits = tgt_q_logits[idx]

        cls_loss = criterion_class(tgt_k_logits, pseudo_labels)
        kl_loss = kl_divergence_loss(tgt_q_logits, tgt_k_logits)
        
        loss = 0.7 * cls_loss + 0.3* kl_loss
        loss.backward()
        tgt_train_optimizer.step()

        update_ema(model_tch, model_stu, decay = 0.996)

        # Accumulate the loss for tracking.
        running_loss += loss.item()

        writer.add_scalar("DA/Train Loss", loss.item(), current_step)
        writer.add_scalar("DA/Train Cls Loss", cls_loss.item(), current_step)
        writer.add_scalar("DA/Train Div Loss", kl_loss.item(), current_step)

        if current_step % 50 == 0:
            print(
                f"Step [{current_step}/{total_steps}], Batch Loss: {loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}, Div Loss: {kl_loss.item():.4f}"
            )
    # Compute and print the average training loss for the epoch.
    avg_train_loss = running_loss / len(target_train_loader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs_da}] Average Training Loss: {avg_train_loss:.4f}"
    )
    writer.add_scalar("DA/EpochLoss", avg_train_loss, epoch)

    # Evaluate on the USPS test set.
    test_loss_stu, test_accuracy_stu = evaluate(model_tch, test_loader=target_test_loader, branch="tgt_test_stu", device=device)
    test_loss_tch, test_accuracy_tch = evaluate(model_tch, test_loader=target_test_loader, branch="tgt_test_tch", device=device)

    print(
        f"Epoch [{epoch + 1}/{num_epochs_da}] Test Loss Student: {test_loss_stu:.4f}, Test Accuracy Student: {test_accuracy_stu:.2f}%"
    )
    print(
        f"Epoch [{epoch + 1}/{num_epochs_da}] Test Loss Teacher: {test_loss_tch:.4f}, Test Accuracy Teacher: {test_accuracy_tch:.2f}%"
    )
    writer.add_scalar("Target/Test EpochLoss Student", test_loss_stu, epoch)
    writer.add_scalar("Target/Test Accuracy Student", test_accuracy_stu, epoch)
    writer.add_scalar("Target/Test EpochLoss Teacher", test_loss_tch, epoch)
    writer.add_scalar("Target/Test Accuracy Teacher", test_accuracy_tch, epoch)

    # Save the best model based on test accuracy.
    if max(test_accuracy_tch, test_accuracy_stu) > best_test_tgt_acc:
        best_test_tgt_acc = max(test_accuracy_tch, test_accuracy_stu)
        best_checkpoint_path = os.path.join("checkpoints/v2", "best_model_da.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model_tch.state_dict(), best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}]: New best model saved with test accuracy: {max(test_accuracy_tch, test_accuracy_stu):.2f}%"
        )
import os

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np

from src.domain_adaptation_model import DomainAdaptationModel
from src.data import source_train_loader, source_test_loader, target_train_loader, target_test_loader
from src.layers.kl_div import kl_divergence_loss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = DomainAdaptationModel().to(device)

num_epochs_burn_in = 100
num_epochs_da = 100
num_epochs = num_epochs_burn_in + num_epochs_da
batch_size = 128
steps_per_epoch = len(source_train_loader)
total_steps = num_epochs * steps_per_epoch

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(0.5 * num_epochs), int(0.72 * num_epochs)], gamma=0.1
)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

log_dir = os.path.join("runs", "domain_adaptation_experiment")
writer = SummaryWriter(log_dir)


best_test_acc = 0.0
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

for epoch in range(num_epochs_burn_in):
    model.train()
    running_loss = 0.0

    # Iterate over paired batches from source and target training loaders.
    for batch_idx, (source_data, target_data) in enumerate(
        zip(source_train_loader, target_train_loader)
    ):
        # Calculate global step based on epoch and batch index.
        current_step = epoch * len(source_train_loader) + batch_idx
        alpha = (2.0 / (1.0 + np.exp(-10 * (current_step / total_steps)))) - 1

        # Extract weak and strong data for source and target
        src_q_data, src_k_data, src_labels = source_data
        tgt_q_data, tgt_k_data, _ = target_data

        src_img = src_k_data.to(device)
        src_labels = src_labels.to(device)
        tgt_img = tgt_k_data.to(device)

        # Zero the gradients for this step.
        optimizer.zero_grad()

        # Forward pass through the model.
        class_logits, domain_source_logits, domain_target_logits = model(
            src_img, tgt_img, alpha, branch="da_train"
        )

        # Compute the classification loss for source domain.
        loss_class = criterion_class(class_logits, src_labels)

        # Compute the domain classification loss for both domains.
        domain_labels_source = torch.zeros(src_img.size(0), dtype=torch.long).to(device)
        domain_labels_target = torch.ones(tgt_img.size(0), dtype=torch.long).to(device)
        domain_logits = torch.cat([domain_source_logits, domain_target_logits], dim=0)
        domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
        loss_domain = criterion_domain(domain_logits, domain_labels)

        # Total loss is the sum of classification and domain loss.
        loss = loss_class + 0.001 * loss_domain
        loss.backward()
        optimizer.step()

        # Accumulate the loss for tracking.
        running_loss += loss.item()

        # Log training loss for each iteration.
        writer.add_scalar("Burn-in/Train Cls loss", loss_class.item(), current_step)
        writer.add_scalar("Burn-in/Train Dis loss", loss_domain.item(), current_step)
        writer.add_scalar("Burn-in/Train BatchLoss", loss.item(), current_step)

        # Print loss and update global_step after every 50 steps.
        if current_step % 50 == 0:
            print(
                f"Step [{current_step}/{total_steps}], Batch Loss: {loss.item():.4f}, Cls Loss: {loss_class.item():.4f}, Dis Loss: {loss_domain.item():.4f}"
            )

    # Compute and print the average training loss for the epoch.
    avg_train_loss = running_loss / len(source_train_loader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}"
    )
    writer.add_scalar("Burn-in/EpochLoss", avg_train_loss, epoch)

    # Evaluate on the MNIST test set.
    test_loss, test_accuracy = evaluate(model, test_loader=source_test_loader, branch="src_test", device=device)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
    )
    writer.add_scalar("Source/Test EpochLoss", test_loss, epoch)
    writer.add_scalar("Source/Test Accuracy", test_accuracy, epoch)

    # Save the best model based on test accuracy.
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_checkpoint_path = os.path.join("checkpoints", "best_model.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}]: New best model saved with test accuracy: {test_accuracy:.2f}%"
        )

for epoch in range(num_epochs_da):
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.train()
    running_loss = 0.0

    for batch_idx, target_data in enumerate(target_train_loader):

        current_step = epoch * len(target_train_loader) + batch_idx
        tgt_q_data, tgt_k_data, _ = target_data

        tgt_q_data = tgt_q_data.to(device)
        tgt_k_data = tgt_k_data.to(device)

        optimizer.zero_grad()

        tgt_q_logits, tgt_k_logits = model(
            tgt_q_data, tgt_k_data, alpha, branch="tgt_train"
        )

        loss = kl_divergence_loss(tgt_q_logits, tgt_k_logits)
        loss.backward()
        optimizer.step()

        # Accumulate the loss for tracking.
        running_loss += loss.item()

        writer.add_scalar("DA/Train KL-div Loss", loss.item(), current_step)
    
    # Compute and print the average training loss for the epoch.
    avg_train_loss = running_loss / len(target_train_loader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}"
    )
    writer.add_scalar("DA/EpochLoss", avg_train_loss, epoch)

    # Evaluate on the USPS test set.
    test_loss, test_accuracy = evaluate(model, test_loader=target_test_loader, branch="tgt_test", device=device)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
    )
    writer.add_scalar("Target/Test EpochLoss", test_loss, epoch)
    writer.add_scalar("Target/Test Accuracy", test_accuracy, epoch)

    # Save the best model based on test accuracy.
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_checkpoint_path = os.path.join("checkpoints", "best_model_da.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}]: New best model saved with test accuracy: {test_accuracy:.2f}%"
        )
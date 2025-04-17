import torch
import torch.nn as nn
import copy

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images, output_type="logits")
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate_wo_vr(model, test_loader, device):
    model_wo_vr = copy.deepcopy(model)
    # makes the visual prompt becames identity
    model_wo_vr.visual_prompt = nn.Identity()

    return evaluate(model_wo_vr, test_loader, device)
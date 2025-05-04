# Source data train only
from src.data import source_train_loader, source_test_loader, target_test_loader
from src.layers.torch_nn import Classifier

from src.layers.utils import freeze_layers
from src.eval import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import os


from torchvision.models import ResNet18_Weights, resnet18
from src.layers.instance_model import InstancewiseVisualPrompt_v2
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    def __init__(
        self,
        num_classes=10,
        imgsize=224,
        vr_blocks=2,
        attribute_layers=[5,6],
        patch_size=[16,32],
        attribute_channels=3,
    ):
        super(BaseClassifier, self).__init__()
        self.backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        assert len(attribute_layers) == len(patch_size) == vr_blocks

        self.visual_prompt = nn.ModuleList([
            InstancewiseVisualPrompt_v2(
                imgsize, attribute_layers[idx], patch_size[idx], attribute_channels, dropout_p=0.5
        ) for idx in range(vr_blocks)])
        self.classifier_head = Classifier(
            in_dim=512, hidden_dim=256, out_dim=num_classes, num_res_blocks=2, dropout=0.5
        )
    
    def forward(self, x, output_type="logits"):
        for layers in self.visual_prompt:
            x = layers(x)
        feat = self.backbone(x)

        if output_type == "feat":
            return feat
        elif output_type == "logits":
            return self.classifier_head(feat)
        else:
            print(f"Not implemented output type {output_type}") 

    def mc_forward(self, x, mc_samples = None, tau = None, output_type="logits"):
        probs = []
        for _ in range(mc_samples):
            logits = self.forward(x, output_type=output_type) / tau 
            probs.append(F.softmax(logits, dim = -1))
        
        P = torch.stack(probs, 0)
        p_mean = P.mean(0)
        ent = -(p_mean*(p_mean+1e-8).log()).sum(-1)
        return p_mean, ent

epochs = 10

stu_model = BaseClassifier(vr_blocks=1, patch_size=[32], attribute_layers = [6])
tch_model = BaseClassifier(vr_blocks=1, patch_size=[32], attribute_layers = [6])
device = torch.device("cuda:0")
stu_model = stu_model.to(device)
tch_model = tch_model.to(device)

layers = (
    [param for name, param in stu_model.named_parameters() if "visual_prompt" not in name] +
    [param for name, param in tch_model.named_parameters() if "visual_prompt" not in name] 
)

optimizer = optim.AdamW(layers, lr=0.01)
vr_layers = list(stu_model.visual_prompt.parameters()) + list(tch_model.visual_prompt.parameters()) 
optimizer_vr = optim.AdamW(vr_layers, lr=0.001)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

log_dir = os.path.join("runs", "da_exp_v12")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

#training script
best_test_acc = 0
mc_samples = 4
tau = 1.0
freeze_layers([stu_model.backbone, tch_model.backbone])

for epoch in range(epochs):
    stu_model.train()
    tch_model.train()
    running_loss = 0.0

    pbar = tqdm(source_train_loader, total=len(source_train_loader), desc=f"Epoch {epoch+1}", ncols=100)
    
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

        p_s_q, u_s_q = tch_model(src_q_data, mc_samples, tau, output_type = "mc_inf")
        p_s_k, u_s_k = stu_model(src_k_data, mc_samples, tau, output_type = "mc_inf")

        loss_cls = criterion_class(p_s_q, src_labels)
        loss_uncertainty = (u_s_q.mean() - u_s_k.mean()).pow(2)
        
        loss = loss_cls + 0.25 * loss_uncertainty
        loss.backward()
        optimizer.step()
        optimizer_vr.step()

        # update_ema(tch_model.visual_prompt[0],stu_model.visual_prompt, decay=0.9996)

        writer.add_scalar("Source/Train Cls loss", loss_cls.item(), current_step)
        writer.add_scalar("Source/Train Unt loss", loss_uncertainty.item(), current_step)
        writer.add_scalar("Source/Train BatchLoss", loss.item(), current_step)

    test_loss_src, test_accuracy_src = evaluate(tch_model, test_loader=source_test_loader, device=device)
    test_loss_tgt, test_accuracy_tgt = evaluate(tch_model, test_loader=target_test_loader, device=device)
    writer.add_scalar("Source/Test EpochLoss", test_loss_src, epoch)
    writer.add_scalar("Source/Test Accuracy", test_accuracy_src, epoch)

    writer.add_scalar("Target/Test EpochLoss", test_loss_tgt, epoch)
    writer.add_scalar("Target/Test Accuracy", test_accuracy_tgt, epoch)

    # writer.add_scalar("Target/Test EpochLoss (w/o VR)", test_loss_tgt_wo_vr, epoch)
    # writer.add_scalar("Target/Test Accuracy (w/o VR)", test_accuracy_tgt_wo_vr, epoch)    

    print(
        f"Epoch [{epoch + 1}/{epochs}] Test Loss Source: {test_loss_src:.4f}, Test Accuracy Source: {test_accuracy_src:.2f}%"
    )
    print(
        f"Epoch [{epoch + 1}/{epochs}] Test Loss Target: {test_loss_tgt:.4f}, Test Accuracy Target: {test_accuracy_tgt:.2f}%"
    )

    # Save the best model based on test accuracy.
    if  test_accuracy_src > best_test_acc:
        best_test_acc = test_accuracy_src
        best_checkpoint_path = os.path.join("checkpoints", "best_model_v12.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(tch_model.state_dict(), best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}]: New best model saved with test accuracy: {test_accuracy_src:.2f}%"
        )
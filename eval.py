import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, branch, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images, branch=branch, inf_type="det", out_type="logits")
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def vis_evaluate(
    model,
    branch,
    test_loader,
    device,
    epoch,
    vis_root="vis",
    gamma: float = 0.8,
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    save_dpi: int = 200
):
    model.eval()
    vr = model.visual_prompt_src if branch == "src" else model.visual_prompt_tgt

    epoch_dir = os.path.join(vis_root, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            reprogrammed = vr(images)
            logits, uncertainty = model(
                images,
                branch=branch,
                inf_type="mc",
                out_type="logits",
                mc_samples=16,
                tau=0.8
            )
            preds = logits.argmax(dim=1)

            for rep, gt, pred, unc in zip(reprogrammed, labels, preds, uncertainty):
                # prep image
                img = rep.cpu().clone()
                img_np = img.permute(1,2,0).numpy() if img.ndim==3 else img.squeeze().numpy()
                lo, hi = np.percentile(img_np, (pct_low, pct_high))
                img_np = np.clip((img_np - lo)/(hi-lo+1e-8), 0,1)
                if gamma != 1.0:
                    img_np = np.power(img_np, gamma)

                # plot
                fig, ax = plt.subplots(figsize=(2,2), dpi=save_dpi//2)
                if img_np.ndim == 2:
                    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(img_np, vmin=0, vmax=1, interpolation="nearest")

                title = f"unc={unc:.3f}  pred={pred.item()}  gt={gt.item()}"
                ax.set_title(title, fontsize=8, pad=6)         # ← add pad here
                ax.axis("off")

                # ensure layout leaves space
                fig.tight_layout(pad=0.5)                       # ← small padding
                fig.subplots_adjust(top=0.88)                   # ← lower the axes a bit

                # save
                fname = f"gt{gt.item()}_pred{pred.item()}_unc{unc:.3f}.png"
                out_path = os.path.join(epoch_dir, fname)
                fig.savefig(out_path,
                            dpi=save_dpi,
                            bbox_inches="tight",
                            pad_inches=0)
                plt.close(fig)

            break  # only first batch

    print(f"Saved 10 images → {epoch_dir}")
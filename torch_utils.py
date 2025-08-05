import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

def freeze_layers(layers: list[nn.Module]):
    """
    Freeze layers listed in the model
    """
    for layer in layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False


def compute_hard_alpha(u: torch.Tensor, t_u: float):
    """
    Compute hard alpha.

    Args:
        u (torch.Tensor): filtering tensor
        t_u (float): threshold

    """
    mask = (u <= t_u).float()
    w = torch.exp(-u) * mask
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha(u: torch.Tensor):
    w = torch.exp(-u)
    return w / (w.sum() + 1e-12) * u.numel()


def compute_soft_alpha_anneal(u, step, total_steps, min_temp=0.1, max_temp=1.0):
    frac = step / float(total_steps)
    T = max_temp * (1 - frac) + min_temp * frac
    w = torch.exp(-u / T)
    w = w / (w.max().clamp(min=1e-6))
    return w


def decay_thresholds(thres_start, thres_end, total_steps, method="exp"):
    if method == "exp":
        t = np.linspace(0, 1, total_steps)
        values = thres_start * (thres_end / thres_start) ** t
    elif method == "log":
        t = np.logspace(0, 1, total_steps, base=10)
        t = (t - t.min()) / (t.max() - t.min())
        values = thres_start - (thres_start - thres_end) * t
    else:
        raise ValueError("Invalid method. Use 'exp' or 'log'.")
    return list(values)

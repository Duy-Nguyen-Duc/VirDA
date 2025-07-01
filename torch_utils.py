import torch
import torch.nn as nn
from torch.autograd import Function


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


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    """
    Updates the parameters of ema_model to be the exponential moving average (EMA)
    of the parameters of model.

    Args:
        ema_model (nn.Module): The model to update (teacher/EMA model).
        model (nn.Module): The source model (student) whose parameters are used.
        decay (float): EMA decay rate (usually close to 1, e.g., 0.99 or 0.999).
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1 - decay)


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
import torch
import torch.nn as nn

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    """
    Updates the parameters of ema_model to be the exponential moving average (EMA)
    of the parameters of model.
    
    Args:
        ema_model (nn.Module): The model to update (teacher/EMA model).
        model (nn.Module): The source model (student) whose parameters are used.
        decay (float): EMA decay rate (usually close to 1, e.g., 0.99 or 0.999).
    """
    # Loop over both parameter sets and update the EMA model.
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1 - decay)

def freeze_layers(layers: str):
    """
    Freeze layers listed in the model
    """
    for layer in layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False

    
def freeze_layers(layers: str):
    """
    Freeze layers listed in the model
    """
    for layer in layers:
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False

    
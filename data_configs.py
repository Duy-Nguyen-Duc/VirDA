
"""Minimal dataset configurations for testing"""

class SyntheticDataset:
    def __init__(self, root, train=True, transform=None, download=False):
        import numpy as np
        from PIL import Image
        from torch.utils.data import Dataset
        
        self.num_samples = 100 if train else 50
        self.num_classes = 10
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        label = np.random.randint(0, self.num_classes)
        return img, label

DATASET_CONFIGS = {
    "synthetic_source": {
        "cls": SyntheticDataset,
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download,
        },
        "convert_to_rgb": True,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "strong_affine": {
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
    },
    "synthetic_target": {
        "cls": SyntheticDataset,
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download,
        },
        "convert_to_rgb": True,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "strong_affine": {
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
    },
}

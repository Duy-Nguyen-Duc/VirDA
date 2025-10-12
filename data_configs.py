import os
from torchvision.datasets import MNIST, USPS, SVHN, ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_CONFIGS = {
    "mnist": {
        "cls": MNIST,
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download,
        },
        "convert_to_rgb": True,
        "mean": [0.1307] * 3,
        "std": [0.3081] * 3,
        "strong_affine": {
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 10,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
        },
    },
    "usps": {
        "cls": USPS,
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download,
        },
        "convert_to_rgb": True,
        "mean": [0.17] * 3,
        "std": [0.3652] * 3,
        "strong_affine": {
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 10,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
        },
    },
    "svhn": {
        "cls": SVHN,
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "split": split,
            "download": download,
        },
        "convert_to_rgb": False,
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
        "strong_affine": {
            "degrees": 10,
            "translate": (0.05, 0.05),
            "scale": (0.95, 1.05),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
        },
    },
    "office31_amazon": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OFFICE31", "amazon/images"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "office31_dslr": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OFFICE31", "dslr/images"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "office31_webcam": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OFFICE31", "webcam/images"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 5,
            "translate": (0.05, 0.05),
            "scale": (0.95, 1.05),
            "shear": 2,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "officehome_art": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OfficeHome", "Art"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "officehome_clipart": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OfficeHome", "Clipart"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "officehome_product": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OfficeHome", "Product"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
    "officehome_realworld": {
        "cls": ImageFolder,
        "args_fn": lambda train, root, download, split: {
            "root": os.path.join(root, "OfficeHome", "RealWorld"),
            "transform": None,
        },
        "convert_to_rgb": False,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "strong_affine": {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 5,
        },
        "jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
        },
    },
}

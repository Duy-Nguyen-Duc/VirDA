import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.datasets import MNIST, USPS, SVHN

from torchvision.datasets import MNIST, USPS, SVHN
from torch.utils.data import Dataset
from torchvision import transforms

# ─── 1. Dataset‐specific configuration ───────────────────────────────────────
# For each dataset we specify:
#   • `cls`: the dataset class
#   • `args_fn`: how to construct its __init__ arguments from (train, root, download, split)
#   • `convert_to_rgb`: whether to force‐convert every image to RGB
#   • `mean` / `std`: lists of length 3 (even for grayscale we duplicate)
#   • `strong_affine`: parameters for RandomAffine
#   • `jitter`: parameters for ColorJitter
DATASET_CONFIGS = {
    "mnist": {
        "cls": MNIST,
        # MNIST’s __init__ is MNIST(root, train, download)
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download
        },
        "convert_to_rgb": True,
        # duplicate the single‐channel stats three times
        "mean": [0.1307, 0.1307, 0.1307],
        "std":  [0.3081, 0.3081, 0.3081],
        "strong_affine": {     # these were “too strong” for SVHN; OK for MNIST/USPS
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 10,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast":   0.2,
        },
    },

    "usps": {
        "cls": USPS,
        # USPS(root, train, download)
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "train": train,
            "download": download
        },
        "convert_to_rgb": True,
        # approximate USPS grayscale stats (you can compute exact if desired)
        "mean": [0.1700, 0.1700, 0.1700],
        "std":  [0.3652, 0.3652, 0.3652],
        "strong_affine": {
            "degrees": 15,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": 10,
        },
        "jitter": {
            "brightness": 0.2,
            "contrast":   0.2,
        },
    },

    "svhn": {
        "cls": SVHN,
        # SVHN(root, split, download)
        "args_fn": lambda train, root, download, split: {
            "root": root,
            "split": split,
            "download": download
        },
        "convert_to_rgb": False,  
        # SVHN is already RGB, so no need to re‐convert
        "mean": [0.4377, 0.4438, 0.4728],
        "std":  [0.1980, 0.2010, 0.1970],
        "strong_affine": {
            # milder than MNIST/USPS, because SVHN digits are small and color‐busy
            "degrees": 10,
            "translate": (0.05, 0.05),
            "scale": (0.95, 1.05),
            "shear": 5,
        },
        "jitter": {
            "brightness":   0.1,
            "contrast":     0.1,
            "saturation":   0.1,  # keep some color jitter
        },
    },
}


class StrongWeakAugDataset(Dataset):
    def __init__(self, dataset_name, root, img_size=224, train=True, download=True):
        self.train = train
        ds_name = dataset_name.lower()
        if ds_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        cfg = DATASET_CONFIGS[ds_name]

        # Determine which “split” argument to pass for SVHN vs. others:
        split = "train" if self.train else "test"

        # Build the dataset instance:
        args = cfg["args_fn"](train, root, download, split)
        self.dataset = cfg["cls"](**args)  # e.g. MNIST(root=..., train=..., download=...)

        # Shortcut references:
        to_rgb_flag = cfg["convert_to_rgb"]
        mean = cfg["mean"]
        std  = cfg["std"]
        affine_params = cfg["strong_affine"]
        jitter_params = cfg["jitter"]

        # 2.1. A small helper: “Convert any PIL.Image → RGB if requested”
        if to_rgb_flag:
            # For grayscale datasets (MNIST/USPS), .convert("RGB") replicates into 3 channels.
            convert_to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
        else:
            # For SVHN, we can pass-through (already RGB)
            convert_to_rgb = transforms.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB"))

        # 2.2. “Weak” transform always does: Resize → (maybe convert→RGB) → ToTensor → Normalize
        self.weak_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # 2.3. “Strong” transform does: Resize → (maybe convert→RGB) → RandomAffine → ColorJitter → ToTensor → Normalize
        strong_list = [
            transforms.Resize((img_size, img_size)),
            convert_to_rgb,
            transforms.RandomAffine(
                degrees=affine_params["degrees"],
                translate=affine_params["translate"],
                scale=affine_params["scale"],
                shear=affine_params["shear"],
            ),
            transforms.ColorJitter(**jitter_params),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.strong_transform = transforms.Compose(strong_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        weak_img = self.weak_transform(img)
        if self.train:
            strong_img = self.strong_transform(img)
            return weak_img, strong_img, label
        else:
            return weak_img, label


def make_dataset(
    source_dataset,
    target_dataset,
    img_size,
    train_bs,
    eval_bs,
    num_workers=4,
    data_frac: float = 1.0,
):
    source_train_data = StrongWeakAugDataset(
        dataset_name=source_dataset, root="./data", img_size=img_size, train=True
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root="./data", img_size=img_size, train=True
    )

    def make_sampler(dataset_length, frac, seed=42):
        if frac < 1.0:
            idxs = list(range(dataset_length))
            random.seed(seed)
            random.shuffle(idxs)
            cut = int(frac * dataset_length)
            return SubsetRandomSampler(idxs[:cut])
        else:
            return None

    source_sampler = make_sampler(len(source_train_data), data_frac)

    source_train_loader = DataLoader(
        source_train_data,
        batch_size=train_bs,
        sampler=source_sampler,
        shuffle=(source_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    target_train_loader = DataLoader(
        target_train_data,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    source_test_data = StrongWeakAugDataset(
        dataset_name=source_dataset, root="./data", img_size=img_size, train=False
    )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root="./data", img_size=img_size, train=False
    )
    source_test_loader = DataLoader(
        source_test_data,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    target_test_loader = DataLoader(
        target_test_data,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_test_loader,
        target_test_loader,
    )

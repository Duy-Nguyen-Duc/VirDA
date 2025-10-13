import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from data_configs import DATASET_CONFIGS
import math


class StrongWeakAugDataset(Dataset):
    def __init__(self, dataset_name, root, img_size=224, train=True, download=True):
        self.train = train
        ds_name = dataset_name.lower()
        if ds_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        cfg = DATASET_CONFIGS[ds_name]
        split = "train" if self.train else "test"
        args = cfg["args_fn"](train, root, download, split)
        self.dataset = cfg["cls"](**args)

        to_rgb_flag = cfg["convert_to_rgb"]
        mean = cfg["mean"]
        std = cfg["std"]
        affine_params = cfg["strong_affine"]
        jitter_params = cfg["jitter"]

        if to_rgb_flag:
            convert_to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
        else:
            convert_to_rgb = transforms.Lambda(
                lambda img: img if img.mode == "RGB" else img.convert("RGB")
            )

        self.weak_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

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
):
    source_train_data = StrongWeakAugDataset(
        dataset_name=source_dataset,
        root="./data",
        img_size=img_size,
        train=True,
        download=False,
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset,
        root="./data",
        img_size=img_size,
        train=True,
        download=False,
    )
    k = math.ceil(len(target_train_data) / len(source_train_data))

    source_train_loader = DataLoader(
        source_train_data,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    target_train_loader = DataLoader(
        target_train_data,
        batch_size=train_bs * k,
        shuffle=True,
        drop_last=True,
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
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    target_test_loader = DataLoader(
        target_test_data,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_test_loader,
        target_test_loader,
    )

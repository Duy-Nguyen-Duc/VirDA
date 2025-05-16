import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.datasets import MNIST, USPS


class StrongWeakAugDataset(Dataset):
    def __init__(self, dataset_name, root, img_size=224, train=True, download=True):
        self.train = train
        if dataset_name.lower() == "mnist":
            self.dataset = MNIST(root=root, train=train, download=download)
        elif dataset_name.lower() == "usps":
            self.dataset = USPS(root=root, train=train, download=download)
        else:
            raise ValueError("Unsupported dataset: {}".format(dataset_name))

        self.weak_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.strong_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(3),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
        num_workers=4,
        pin_memory=True,
    )
    target_train_loader = DataLoader(
        target_train_data,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4,
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
        num_workers=4,
        pin_memory=True,
    )
    target_test_loader = DataLoader(
        target_test_data,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return (
        source_train_loader,
        target_train_loader,
        source_test_loader,
        target_test_loader,
    )

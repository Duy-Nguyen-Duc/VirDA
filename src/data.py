import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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

        # Weak augmentation (used in both training and testing)
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

        # Strong augmentation (only used during training)
        self.strong_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(3),
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
            # For testing, return only the weakly augmented image.
            return weak_img, label


# Train loaders (with dual augmentation for teacher/student training)
source_train_data = StrongWeakAugDataset(
    dataset_name="mnist", root="./data", img_size=224, train=True
)
target_train_data = StrongWeakAugDataset(
    dataset_name="usps", root="./data", img_size=224, train=True
)
source_train_loader = DataLoader(
    source_train_data, batch_size=256, shuffle=True, num_workers=2
)
target_train_loader = DataLoader(
    target_train_data, batch_size=256, shuffle=True, num_workers=2
)

# Test loaders (only weak augmentation, similar to self.weak_transform)
source_test_data = StrongWeakAugDataset(
    dataset_name="mnist", root="./data", img_size=224, train=False
)
target_test_data = StrongWeakAugDataset(
    dataset_name="usps", root="./data", img_size=224, train=False
)
source_test_loader = DataLoader(
    source_test_data, batch_size=256, shuffle=False, num_workers=2
)
target_test_loader = DataLoader(
    target_test_data, batch_size=256, shuffle=False, num_workers=2
)

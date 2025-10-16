import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from data_configs import DATASET_CONFIGS
import math

import torch.nn.functional as F
import torchvision.transforms.functional as TF

class StrongWeakAugDataset(Dataset):
    def __init__(self, dataset_name, root, imgsize=224, train=True, download=True):
        self.train = train
        self.imgsize = imgsize
        ds_name = dataset_name.lower()
        if ds_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        cfg = DATASET_CONFIGS[ds_name]
        split = "train" if self.train else "test"
        args = cfg["args_fn"](train, root, download, split)
        self.dataset = cfg["cls"](**args)

        self.base_transform = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(lambda img: img.convert("RGB"))
        ])
        self.affine_param_ranges = cfg["strong_affine"]

        self.color_strong_transform = transforms.Compose([
            transforms.ColorJitter(**cfg["jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

        self.weak_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.base_transform(img)
        weak_img = self.weak_transform(img)

        if self.train:
            affine_params_tuple = transforms.RandomAffine.get_params(
                degrees=[self.affine_param_ranges["degrees"], self.affine_param_ranges["degrees"]],
                translate=self.affine_param_ranges["translate"],
                scale_ranges=self.affine_param_ranges["scale"],
                shears=[self.affine_param_ranges["shear"], self.affine_param_ranges["shear"]],
                img_size=[self.imgsize, self.imgsize]
            )   
            affine_params = {
                'angle': affine_params_tuple[0],
                'translate': list(affine_params_tuple[1]),
                'scale': affine_params_tuple[2],
                'shear': list(affine_params_tuple[3])
            }
            spatial_strong_transform = TF.affine(img, **affine_params, fill=0)
            strong_img = self.color_strong_transform(spatial_strong_transform)
            return weak_img, strong_img, label, affine_params
        else:
            return weak_img, label


def make_dataset(
    source_dataset,
    target_dataset,
    imgsize,
    train_bs,
    eval_bs,
    num_workers=4,
):
    source_train_data = StrongWeakAugDataset(
        dataset_name=source_dataset,
        root="./data",
        imgsize=imgsize,
        train=True,
        download=False,
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset,
        root="./data",
        imgsize=imgsize,
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
        dataset_name=source_dataset, root="./data", imgsize=imgsize, train=False
    )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root="./data", imgsize=imgsize, train=False
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

def transform_map(salience_map, affine_params, transform_params=[0.0, 1.0], imgsize=224):
    bg_w, fg_w = transform_params
    device = salience_map.device if torch.is_tensor(salience_map) else 'cpu'
    
    if not torch.is_tensor(salience_map):
        salience_map = torch.from_numpy(salience_map).to(device)
    
    salience_map = bg_w * (1 - salience_map) + fg_w * salience_map # [B, H, W]
    salience_map = salience_map.unsqueeze(1) # [B, 1, H, W]
            
    angles = torch.as_tensor(affine_params['angle'], dtype=torch.float32, device=device).view(-1)
    scales = torch.as_tensor(affine_params['scale'], dtype=torch.float32, device=device).view(-1)
    translates = torch.stack(affine_params['translate'], dim=1).to(dtype=torch.float32)
    shears = torch.stack(affine_params['shear'], dim=1).to(dtype=torch.float32)
    
    affine_matrices = _get_affine_matrix_vectorized(
        angles, translates, scales, shears, salience_map.shape[-2:], device
    )
    
    grid = F.affine_grid(
        affine_matrices, 
        salience_map.shape, 
        align_corners=False
    )
    
    transformed_maps = F.grid_sample(
        salience_map, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=False
    )
    
    if transformed_maps.shape[-1] != imgsize or transformed_maps.shape[-2] != imgsize:
        transformed_maps = F.interpolate(
            transformed_maps, 
            size=(imgsize, imgsize), 
            mode='bilinear', 
            align_corners=False
        )
    
    return transformed_maps

def _get_affine_matrix_vectorized(angles, translates, scales, shears, img_size, device):
    angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
    translates = torch.as_tensor(translates, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
    shears = torch.as_tensor(shears, dtype=torch.float32, device=device)
    
    batch_size = angles.shape[0]

    angle_rad = torch.deg2rad(angles)
    shear_x_rad = torch.deg2rad(shears[:, 0])
    shear_y_rad = torch.deg2rad(shears[:, 1])

    height, width = img_size
    center = torch.tensor([width / 2.0, height / 2.0], device=device).view(1, 2)

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, 0, 0] = cos_a
    rotation_matrix[:, 0, 1] = -sin_a
    rotation_matrix[:, 1, 0] = sin_a
    rotation_matrix[:, 1, 1] = cos_a

    scale_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    scale_matrix[:, 0, 0] = scales
    scale_matrix[:, 1, 1] = scales

    tan_sx = torch.tan(shear_x_rad)
    tan_sy = torch.tan(shear_y_rad)
    shear_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    shear_matrix[:, 0, 1] = -tan_sx
    shear_matrix[:, 1, 0] = -tan_sy
    
    matrix = torch.bmm(shear_matrix, rotation_matrix)
    matrix = torch.bmm(scale_matrix, matrix)
    
    c = translates[:, 0] + center[:, 0] - matrix[:, 0, 0] * center[:, 0] - matrix[:, 0, 1] * center[:, 1]
    f = translates[:, 1] + center[:, 1] - matrix[:, 1, 0] * center[:, 0] - matrix[:, 1, 1] * center[:, 1]

    final_matrix = torch.zeros(batch_size, 2, 3, device=device)
    final_matrix[:, :, :2] = matrix[:, :2, :2]
    final_matrix[:, 0, 2] = c
    final_matrix[:, 1, 2] = f

    return final_matrix
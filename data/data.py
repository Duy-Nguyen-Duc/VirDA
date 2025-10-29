import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from .data_configs import DATASET_CONFIGS


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
    root,
    source_dataset,
    target_dataset,
    imgsize,
    train_bs,
    eval_bs,
    num_workers=4,
):
    source_train_data = StrongWeakAugDataset(
        dataset_name=source_dataset,
        root=root,
        imgsize=imgsize,
        train=True,
        download=False,
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset,
        root=root,
        imgsize=imgsize,
        train=True,
        download=False,
    )

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
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    source_test_data = StrongWeakAugDataset(
        dataset_name=source_dataset, root=root, imgsize=imgsize, train=False
    )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root=root, imgsize=imgsize, train=False
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

def transform_map(salience_map, affine_params=None, transform_params=[0.0, 1.0], imgsize=224):
    bg_w, fg_w = transform_params
    device = salience_map.device if torch.is_tensor(salience_map) else 'cpu'
    
    if not torch.is_tensor(salience_map):
        salience_map = torch.from_numpy(salience_map).to(device)
    
    salience_map = bg_w * (1 - salience_map) + fg_w * salience_map
    if salience_map.ndim == 3:
        salience_map = salience_map.unsqueeze(1)
    
    salience_map = F.interpolate(salience_map, size=(imgsize, imgsize), mode='bilinear', align_corners=False) 
    if affine_params is not None:
        batch_size = salience_map.shape[0]
        transformed_batch = []
        
        for i in range(batch_size):
            single_map = salience_map[i:i+1]
            
            angle = affine_params['angle'][i] if isinstance(affine_params['angle'], (list, tuple)) else affine_params['angle'][i].item()
            translate = affine_params['translate'][i] if isinstance(affine_params['translate'][0], (list, tuple)) else [affine_params['translate'][0][i].item(), affine_params['translate'][1][i].item()]
            scale = affine_params['scale'][i] if isinstance(affine_params['scale'], (list, tuple)) else affine_params['scale'][i].item()
            shear = affine_params['shear'][i] if isinstance(affine_params['shear'][0], (list, tuple)) else [affine_params['shear'][0][i].item(), affine_params['shear'][1][i].item()]
            
            transformed = TF.affine(single_map, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
            transformed_batch.append(transformed)
        
        return torch.cat(transformed_batch, dim=0)
    else:
        return salience_map
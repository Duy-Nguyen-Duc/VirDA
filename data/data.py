import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from torch import nn
from torch.utils.data import DataLoader, Dataset
from .data_configs import DATASET_CONFIGS


class MultiRandomErasing(nn.Module):
    """
    Apply torchvision RandomErasing k times (k ~ Uniform[n_min, n_max]).
    Expects a tensor image in [0,1] BEFORE Normalize.
    """
    def __init__(self, n_range=(1, 2), p=0.75, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0):
        super().__init__()
        self.n_range = n_range
        self.p = p
        self.erase = T.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=value, inplace=True)

    def forward(self, img: torch.Tensor):
        if random.random() > self.p:
            return img
        k = random.randint(self.n_range[0], self.n_range[1])
        for _ in range(k):
            img = self.erase(img)
        return img


class StrongWeakAugDataset(Dataset):
    """
    Weak:  flip -> ToTensor -> Normalize
    Strong: flip -> hue-only jitter -> ToTensor -> blackout (cutout) -> Normalize
    No affine transforms (no black corners).
    """
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

        # config defaults
        self.mean = cfg["mean"]
        self.std = cfg["std"]
        self.flip_p = cfg.get("flip_p", 0.5)
        hue = cfg.get("jitter", {}).get("hue", 0.10)

        bo = cfg.get("blackout", {"p": 0.75, "scale": (0.02, 0.20), "ratio": (0.3, 3.3), "times": (1, 2)})
        self.blackout = MultiRandomErasing(
            n_range=bo.get("times", (1, 2)),
            p=bo.get("p", 0.75),
            scale=bo.get("scale", (0.02, 0.20)),
            ratio=bo.get("ratio", (0.3, 3.3)),
            value=0.0
        )

        self.base_resize = T.Resize((imgsize, imgsize))
        self.to_rgb = T.Lambda(lambda img: img.convert("RGB"))
        self.hue_jitter = T.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=hue)

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.dataset)

    def _apply_flip(self, img, do_flip: bool):
        return TF.hflip(img) if do_flip else img

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.to_rgb(self.base_resize(img))

        if self.train:
            flip_w = random.random() < self.flip_p
            flip_s = random.random() < self.flip_p

            # ----- weak view -----
            img_w = self._apply_flip(img, flip_w)
            weak_img = self.normalize(self.to_tensor(img_w))

            # ----- strong view -----
            img_s = self._apply_flip(img, flip_s)         # PIL
            img_s = self.hue_jitter(img_s)                # hue-only jitter
            strong_img = self.to_tensor(img_s)            # to tensor [0,1]
            strong_img = self.blackout(strong_img)        # blackout before Normalize
            strong_img = self.normalize(strong_img)

            # metadata so you can warp maps if needed
            aug_params = {
                "hflip_weak": flip_w,
                "hflip_strong": flip_s,
            }
            return weak_img, strong_img, label, aug_params

        else:
            # eval: no flip/jitter/blackout
            weak_img = self.normalize(self.to_tensor(img))
            return weak_img, label


# ----------------------- dataloaders -----------------------

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
        dataset_name=source_dataset, root=root, imgsize=imgsize, train=True, download=False
    )
    target_train_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root=root, imgsize=imgsize, train=True, download=False
    )

    source_train_loader = DataLoader(
        source_train_data, batch_size=train_bs, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )
    target_train_loader = DataLoader(
        target_train_data, batch_size=train_bs, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )

    source_test_data = StrongWeakAugDataset(
        dataset_name=source_dataset, root=root, imgsize=imgsize, train=False
    )
    target_test_data = StrongWeakAugDataset(
        dataset_name=target_dataset, root=root, imgsize=imgsize, train=False
    )
    source_test_loader = DataLoader(
        source_test_data, batch_size=eval_bs, shuffle=False, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )
    target_test_loader = DataLoader(
        target_test_data, batch_size=eval_bs, shuffle=False, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )

    return (
        source_train_loader,
        target_train_loader,
        source_test_loader,
        target_test_loader,
    )


# ----------------------- map warper -----------------------

def transform_map(salience_map, aug_params=None, transform_params=[0.0, 1.0], imgsize=224):
    """
    Rescales a [B,1,h,w] (or [B,h,w]) map to imgsize and applies the *relative*
    weak→strong flip if provided via aug_params.

    Supported aug_params:
      - None: no-op
      - {"hflip": bool or [bool]*B}: apply that flip to every sample
      - {"hflip_weak": bool, "hflip_strong": bool} (scalars or lists/tensors):
          apply flip if hflip_weak != hflip_strong per sample
      - (legacy) affine dict with angle/translate/scale/shear: ignored here
        because we removed affine from the pipeline.
    """
    bg_w, fg_w = transform_params
    device = salience_map.device if torch.is_tensor(salience_map) else 'cpu'

    if not torch.is_tensor(salience_map):
        salience_map = torch.from_numpy(salience_map).to(device)

    # [0,1] blend between bg and fg
    salience_map = bg_w * (1 - salience_map) + fg_w * salience_map

    if salience_map.ndim == 3:
        salience_map = salience_map.unsqueeze(1)  # [B,1,h,w]

    salience_map = F.interpolate(
        salience_map, size=(imgsize, imgsize),
        mode='bilinear', align_corners=False
    )

    if aug_params is None:
        return salience_map

    # --- flip support ---
    def to_bool_list(x, B):
        if isinstance(x, (list, tuple)):
            return [bool(v) for v in x]
        if torch.is_tensor(x):
            return [bool(v.item()) for v in x.view(-1)]
        return [bool(x)] * B

    B = salience_map.shape[0]

    if "hflip" in aug_params:
        flips = to_bool_list(aug_params["hflip"], B)
        out = []
        for i in range(B):
            out.append(TF.hflip(salience_map[i]) if flips[i] else salience_map[i])
        return torch.stack(out, dim=0)

    if "hflip_weak" in aug_params and "hflip_strong" in aug_params:
        fw = to_bool_list(aug_params["hflip_weak"], B)
        fs = to_bool_list(aug_params["hflip_strong"], B)
        out = []
        for i in range(B):
            do_flip = (fw[i] != fs[i])  # flip if weak/strong differ
            out.append(TF.hflip(salience_map[i]) if do_flip else salience_map[i])
        return torch.stack(out, dim=0)

    # Legacy affine keys: safely ignore (no-op) since we removed affine.
    return salience_map

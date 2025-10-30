import torch
import cv2
from torch_nn.model import UModel, EigenCAM
from torch_nn.utils import visualize_salience_map, visualize_multiple_images
from yacs.config import CfgNode as CN
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from data.data import transform_map
import torch.nn.functional as F
import matplotlib.pyplot as plt

cfg = CN(new_allowed=True)
cfg.merge_from_file("config.yaml")

model = UModel(
    backbone=cfg.model.backbone.type,
    hidden_dim=cfg.model.backbone.hidden_dim,
    out_dim=cfg.dataset.num_classes,
    imgsize=cfg.img_size,
    freeze_backbone=cfg.model.backbone.freeze,
)

device = torch.device(cfg.device)
ckpt = torch.load("runs/new_30_10/GuMjEY/da_best_62.91.pth")
model.load_state_dict(ckpt['model_state_dict'])
cam = EigenCAM(model.eval(), model.backbone.transformer.blocks[-1].norm2)
cam.register_hook()
model = model.to(device)

input_paths_target = [
    "dataset/OfficeHome/Clipart/Computer/00027.jpg", 
    "dataset/OfficeHome/Clipart/Computer/00028.jpg",
    "dataset/OfficeHome/Clipart/Mouse/00006.jpg",
    "dataset/OfficeHome/Clipart/Printer/00005.jpg",
    "dataset/OfficeHome/Clipart/Chair/00003.jpg",
    "dataset/OfficeHome/Clipart/Desk_Lamp/00001.jpg",
]

input_paths_source = [
    "dataset/OfficeHome/Art/Computer/00001.jpg", 
    "dataset/OfficeHome/Art/Computer/00002.jpg",
    "dataset/OfficeHome/Art/Mouse/00006.jpg",
    "dataset/OfficeHome/Art/Printer/00005.jpg",
    "dataset/OfficeHome/Art/Chair/00003.jpg",
    "dataset/OfficeHome/Art/Desk_Lamp/00001.jpg",
]

visualize_multiple_images(input_paths_source, model, cam, "src", "src", device, "src_branch_src_data_3.png")
visualize_multiple_images(input_paths_target, model, cam, "tgt", "tgt", device, "tgt_branch_tgt_data_3.png")
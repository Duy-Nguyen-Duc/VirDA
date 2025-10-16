import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

class AttributeNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, channels=3, dropout_p=0.5):
        """
        Paper: https://arxiv.org/abs/2406.03150
        """

        super(AttributeNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.channels = channels

        self.pooling = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(p=dropout_p)
        if self.layers == 5 and self.channels == 3:
            self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)
            self.dropout5 = nn.Dropout2d(p=dropout_p)

            if self.channels == 3:
                self.conv6 = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        if self.patch_size in [2, 4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        if self.patch_size in [4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        if self.patch_size in [8, 16, 32]:
            y = self.pooling(y)
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = self.dropout4(y)
        if self.patch_size in [16, 32]:
            y = self.pooling(y)
        if self.layers == 6:
            y = self.conv5(y)
            y = self.bn5(y)
            y = self.relu5(y)
            y = self.dropout5(y)
            if self.patch_size == 32:
                y = self.pooling(y)

        if self.channels == 3:
            y = self.conv6(y)
        elif self.channels == 1:
            y = torch.mean(y, dim=1)
        return y



class CoordAtt(nn.Module):
    """
    Hou et al., 2021 (Coordinate Attention).
    Keeps [B,C,H,W]. Adds positional info via separate H/W encodings.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        m = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, m, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(m, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(m, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # split and project back
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w

class InstancewiseVisualPromptCoordNet(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3, dropout_p=0.3):
        """
        Paper: https://arxiv.org/abs/2406.03150
        Args:
            size: input image size
            layers: the number of layers of mask-training CNN
            patch_size: the size of patches with the same mask value
            channels: 3 means that the mask value for RGB channels are different, 1 means the same
            keep_watermark: whether to keep the reprogram (delta) in the model
        """
        super(InstancewiseVisualPromptCoordNet, self).__init__()
        if layers not in [5, 6]:
            raise ValueError("Input layer number is not supported")
        if patch_size not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Input patch size is not supported")
        if channels not in [1, 3]:
            raise ValueError("Input channel number is not supported")
        if patch_size == 32 and layers != 6:
            raise ValueError(
                "Input layer number and patch size are conflict with each other"
            )

        # Set the attribute mask CNN
        self.patch_num = int(size / patch_size)
        self.imgsize = size
        self.patch_size = patch_size
        self.channels = channels
        self.priority = AttributeNet(layers, patch_size, channels, dropout_p)

        # Set reprogram (delta) according to the image size
        self.size = size
        # self.program = torch.nn.Parameter(data=torch.zeros(3, size, size))
        self.program = nn.Parameter(0.001 * torch.randn(3, size, size))

        self.coord_att = CoordAtt(3)
    
    def forward(self, x):
        x = self.coord_att(x)
        att = self.priority(x)
        attention = (
            att.view(-1, self.channels, self.patch_num * self.patch_num, 1)
               .expand(-1, 3, -1, self.patch_size * self.patch_size)
               .view(-1, 3, self.patch_num, self.patch_num, self.patch_size, self.patch_size)
               .transpose(3, 4)
               .reshape(-1, 3, self.imgsize, self.imgsize)
        )
        x = x + self.program * attention
        return x
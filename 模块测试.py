import torch
import torch.nn as nn
from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class ema_1(nn.Module):
    def __init__(self,c1, c2, k ,s,factor = 32):
        super().__init__()
        self.k = k
        self.groups = factor
        # assert c2 // self.groups > 0
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        self.conv1x1 = nn.Conv2d(c1 // self.groups, c1// self.groups, kernel_size=1, stride=1, padding=0)
        self.at = SELayer(c1// self.groups,1)
        self.conv3x3 = Conv(c1 // self.groups, c1// self.groups, k=3, s=1,g=c1 // self.groups)
        self.conv5x5 = Conv(c1 // self.groups, c1// self.groups, k=5, s=1,g=c1 // self.groups)
        self.conv = Conv(c1, c2, k, s)
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x2 = self.at(self.conv3x3(group_x) + self.conv5x5(group_x))
        if self.k ==1:
            return (group_x * (x1+x2).sigmoid()).reshape(b, c, h, w)
        else:
            return self.conv((group_x * (x1 + x2).sigmoid()).reshape(b, c, h, w))
class LightweightSPPFA(nn.Module):
    def __init__(self, c1, c2, k=3, reduction=16):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c1//reduction, 1, 1)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m = nn.Conv2d(c1//reduction, c1//reduction, kernel_size=k, stride=1, padding=k // 2,groups=16)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1//reduction * 4, c1 // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c1 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        out = torch.cat((x1, y1, y2, self.m(y2)), 1)
        out = self.avg_pool1(out).reshape((b, -1))
        b = self.fc(out).view(b, c, 1, 1)
        return x * b
m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = LightweightSPPFA(512, 512)
results = profile(input=torch.randn(16, 512, 20, 20), ops=[m1, m2], n=3)

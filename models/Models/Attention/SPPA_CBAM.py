import torch
from torch.nn import functional
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam

class SPA(nn.Module):
    #多尺度通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool1 =    nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 =    nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 =    nn.AdaptiveAvgPool2d(4)

        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # 设置可学习权值
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.5)
    def forward(self, x):
        b, c, _, _ = x.shape
        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y3 = self.avg_pool4(x).reshape((b, -1))
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).reshape((b, c, 1, 1))
        # y = self.fc(y).reshape((b, c, 1, 1)) *self.w1 #添加自适应学习权值，对通道信息增加自适应特征学习
        return x * y
class SPAF(nn.Module):
    #多尺度通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool1 =    nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 =    nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 =    nn.AdaptiveAvgPool2d(4)

        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # 设置可学习权值
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.5)
    def forward(self, x):
        b, c, _, _ = x.shape
        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y3 = self.avg_pool4(x).reshape((b, -1))
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).reshape((b, c, 1, 1))
        # y = self.fc(y).reshape((b, c, 1, 1)) *self.w1 #添加自适应学习权值，对通道信息增加自适应特征学习
        return x * y

class SpatialAttention(nn.Module):
    # CBAM的空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
class SPPA_CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SPPA_CBAM, self).__init__()
        self.channel_attention = SPA(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out
class SPPFA(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5,reduction=16):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.fc = nn.Sequential(
            nn.Linear(c2 , c2 // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        b, c, _, _ = x.shape
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        out = self.cv2(torch.cat((x1, y1, y2, self.m(y2)), 1))
        out = self.avg_pool1(out).reshape((b, -1))
        b = self.fc(out).view(b, c, 1, 1)
        return x*b

if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = SPPFA(c1=64,c2=64)
    y = net(x)
    print(net)
    print(y.size())
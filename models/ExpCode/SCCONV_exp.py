import torch

from utils.torch_utils import profile
# from models.experimental import MixConv2d

# from models.common import Conv
import torch.nn as nn
import torch
import torch.nn.functional as F
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class SPA(nn.Module):
    #多尺度通道注意力
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)

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
        return y
class SCConv_exp1(nn.Module):
    def __init__(self, inplanes, planes, k, s, p=1, d=1, groups=16, pooling_r=1, norm_layer=nn.BatchNorm2d):
        super(SCConv_exp1, self).__init__()
        self.spa = SPA(inplanes)  # Add SPA layer
        padding = k//2 *2
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                                padding=autopad(k, p, d), dilation=d,
                                groups=groups, bias=False),
                    norm_layer(inplanes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                                padding=autopad(k, p, d), dilation=d,
                                groups=groups, bias=False),
                    norm_layer(inplanes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=k, stride=s,
                                padding=autopad(k, p, d), dilation=d,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = self.spa(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # SPA(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * SPA(identity + k2)
        out = self.k4(out) # k4

        return out
class SCConv_exp2(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride=1, padding=3//2, dilation=1, groups=1, pooling_r=3, norm_layer=nn.BatchNorm2d):
        super(SCConv_exp2, self).__init__()
        self.c = int(inplanes * 0.5)  # hidden channels

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(self.c, self.c, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(self.c),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(self.c, self.c, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(self.c),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x
        y = list(x.split((self.c, self.c), 1))
        out = torch.sigmoid(torch.add(y[-1], F.interpolate(self.k2(y[-1]), y[-1].size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(y[0]), out) # k3 * sigmoid(identity + k2)
        out = self.k4(torch.cat((out, y[0]), dim=1)) # k4
        return out
class SCConv_exp3(nn.Module):
    def __init__(self, inplanes, planes, kernel, stride=1, padding=3//2, dilation=1, groups=2, pooling_r=3, norm_layer=nn.BatchNorm2d):
        super(SCConv_exp3, self).__init__()
        self.c = int(inplanes * 0.5)  # hidden channels
        self.spa = SPA(self.c)  # Add SPA layer
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(self.c, self.c, kernel_size=3, stride=1,
                                padding=autopad(kernel, padding, dilation), dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(self.c),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(self.c, self.c, kernel_size=3, stride=1,
                                padding=autopad(kernel, padding, dilation), dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(self.c),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride,
                                padding=autopad(kernel, padding, dilation), dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x
        y = list(x.split((self.c, self.c), 1))
        out = self.spa(torch.add(y[-1], F.interpolate(self.k2(y[-1]), y[-1].size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(y[0]), out) # k3 * sigmoid(identity + k2)
        out = self.k4(torch.cat((out, y[0]), dim=1)) # k4
        return out
class SEAttention(nn.Module):

    def __init__(self, channel=512,c2=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
if __name__ =='__main__':
    # x = torch.randn(1, 64, 80, 80)  # 用来生成随机数字的tensor 输出一个64*32*48的张量
    # b, c, h, w = x.shape
    # net1 = SCConv2(64, 64, 1)
    # # net2 = SELayer(c, c)
    # y1 = net1(x)
    # print(y1.shape)

    # m1 = Conv(128, 128, 3, 2)
    m2 = SCConv_exp3(128, 128, 3,2)
    results = profile(input=torch.randn(1, 128, 80, 80), ops=[ m2], n=3)

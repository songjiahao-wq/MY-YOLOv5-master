# https://zhuanlan.zhihu.com/p/548365972
# article ES-Net: Efficient Scale-Aware Network for Tiny Defect Detection
"""
论文：ES-Net: Efficient Scale-Aware Network for Tiny Defect Detection
时间：2022年
发表于：IEEE Transactions on Instrumentation and Measurement
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, custom_conv=None):
        super().__init__()
        if custom_conv is None:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        else:
            self.conv = custom_conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PoolingUpsampleModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PoolingUpsampleModel, self).__init__()

        # 创建三个不同尺度的自适应平均池化层
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)

        # 创建三个上采样层，分别将各自对应的池化结果上采样到原始尺度
        # self.up1 = F.interpolate(scale_factor=2,size=[], mode='bilinear', align_corners=True)
        # self.up2 = F.interpolate(scale_factor=4, mode='bilinear', align_corners=True)
        # self.up3 = F.interpolate(scale_factor=6, mode='bilinear', align_corners=True)

        # 创建一个卷积层，它将池化和上采样的结果连接在一起
        self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

        # 创建一个通道注意力模型，它将上面卷积的结果进行注意力特征提取
        self.attention = ChannelAttention(out_channels)

    def forward(self, x):
        # 进行池化操作
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)

        # 进行上采样操作
        up1 = F.interpolate(pool1, size=[x.size(2), x.size(3)], mode="nearest")
        up2 = F.interpolate(pool2, size=[x.size(2), x.size(3)], mode="nearest")
        up3 = F.interpolate(pool3, size=[x.size(2), x.size(3)], mode="nearest")


        # 将上采样的结果融合
        fused = torch.cat((up1, up2, up3), dim=1)

        # 对融合的结果进行卷积操作
        output = self.conv(fused)

        # 对卷积结果进行注意力特征提取
        attention_output = self.attention(output)

        return attention_output

class MMF(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(MMF, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(c1 *4, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1 , bias=False),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(c1 * 16, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = Conv(c1 * 3, c1, 1)
        self.conv = nn.Sequential(
            DWConv(c1 , c1, 3),
            nn.Sigmoid() #or nn.Sigmoid()
        )
        self.conv1_1 = Conv(c1, c1, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y3 = self.avg_pool4(x).reshape((b, -1))
        y1 = self.fc1(y1).reshape((b, c, 1, 1)) * x
        y2 = self.fc2(y2).reshape((b, c, 1, 1)) * x
        y3 = self.fc3(y3).reshape((b, c, 1, 1)) * x
        y = self.conv1(torch.cat((y1,y2,y3), dim=1))
        out = self.conv(y)* y
        return out + self.conv1_1(x)

if __name__ =="__main__":
    # 创建模型实例
    Model = PoolingUpsampleModel(64, 64)

    # 随机输入数据
    input_data = torch.randn(1, 64, 20, 20)

    # 前向传播
    output = Model(input_data)

    print("模型输出:", output.shape)

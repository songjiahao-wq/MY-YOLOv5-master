# https://zhuanlan.zhihu.com/p/548365972
# article ES-Net: Efficient Scale-Aware Network for Tiny Defect Detection
"""
论文：ES-Net: Efficient Scale-Aware Network for Tiny Defect Detection
时间：2022年
发表于：IEEE Transactions on Instrumentation and Measurement
"""
from models.common import Conv, DWConv
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)

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
        up1 = self.up1(pool1)
        up2 = self.up2(pool2)
        up3 = self.up3(pool3)

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
        return out

if __name__ =="__main__":
    # 创建模型实例
    Model = MMF(c1=64, c2=64)

    # 随机输入数据
    input_data = torch.randn(1, 64, 64, 64)

    # 前向传播
    output = Model(input_data)

    print("模型输出:", output.shape)

# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 10:28
# @Author  : sjh
# @Site    : 
# @File    : DPAG.py
# @Comment :
'https://www.mdpi.com/2073-4395/13/7/1824'
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPAG(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.repconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 逐元素相加的分支
        identity = x

        # 通过3x3和5x5的卷积核处理特征图
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)

        # 将两者逐元素相加
        combined = conv3_out + conv5_out

        # 通过全局平均池化和激活函数处理
        gap = self.global_avg_pool(combined)
        channel_att_raw = self.relu(gap)

        # 学习通道权重
        channel_att = self.sigmoid(channel_att_raw)

        # 应用通道注意力
        channel_att_applied = self.conv(x) * channel_att

        # 通过标准卷积和RepVGG块
        out = self.repconv(channel_att_applied)

        # 空间注意力分支
        avg_pool = F.avg_pool2d(out, out.size()[2:])
        max_pool = F.max_pool2d(out, out.size()[2:])
        concat = torch.cat((avg_pool, max_pool), 1)
        spatial_att_raw = self.conv1x1(concat)
        spatial_att = self.sigmoid(spatial_att_raw)

        # 应用空间注意力
        spatial_att_applied = out * spatial_att

        # 将通道注意力和空间注意力的结果相加
        out = spatial_att_applied + identity

        return out

class Feature_Enhancement_Module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=2):
        super(Feature_Enhancement_Module, self).__init__()
        self.paths1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate),
                    nn.ReLU()
                )
        self.paths2 = nn.ModuleList()
        self.paths3 = nn.ModuleList()
        self.paths1.append()

        self.conv1x1 = nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1)

    def forward(self, x):
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x))
        # Concatenate along the channel dimension
        out = torch.cat(path_outputs, dim=1)
        # Reduce the channel dimension with a 1x1 convolution
        out = self.conv1x1(out)
        # Element-wise sum with the input feature map
        out = out + x
        return out

if __name__ =="__main__":
    input = torch.randn(1,256,20,20)
    model1 = DPAG(256)
    model2 = Feature_Enhancement_Module(256)
    print(model(input).shape)
    print(model2(input).shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from models.experimental import MixConv2d
"""
#num_scales参数决定了注意力机制将考虑的不同尺度的数量。这个值取决于你想在你的模型中捕获的多尺度背景的水平和输入图像的大小。
#没有一个放之四海而皆准的数值，因为它取决于你正在处理的具体问题和数据集。
#对于CIFAR-10数据集，它有32x32的图像，你可以从num_scales的值开始，等于3。
#这将允许注意力机制使用扩张的卷积在三种不同的感受野尺寸（3x3，5x5和7x7）上捕捉上下文。
"""
class SpatialMultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_scales,groups= [4, 8, 16]):
        super(SpatialMultiScaleAttention, self).__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales

        # Define the attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(num_scales): #MixConv2d(128, 256, (3, 5), 1)
            # self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1))
            self.attention_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=i+1, dilation=i+1, groups=groups[i]))
            # self.attention_layers.append(MixConv2d(in_channels, in_channels, k=(3, 5), s=1))

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Calculate attention maps for each scale
        attention_maps = []
        for i in range(self.num_scales):
            attention_map = self.attention_layers[i](x)
            attention_map = F.softmax(attention_map.view(batch_size, -1), dim=1)
            attention_map = attention_map.view(batch_size, self.in_channels, height, width)
            attention_maps.append(attention_map)

        # Sum attention maps
        attention_maps = sum(attention_maps)

        # Apply attention to the input
        x = x * attention_maps
        return x
class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModified, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / reduction_ratio), in_channels),
        )
    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        channel_att_avg = self.relu(self.fc(avg_pool)).view(batch_size, channels, 1, 1)
        channel_att_max = self.relu(self.fc(max_pool)).view(batch_size, channels, 1, 1)
        # channel_att_avg = self.relu(self.fc(avg_pool))
        # channel_att_max = self.relu(self.fc(max_pool))
        #
        # if batch_size > 1:
        #     y = channel_att_avg.view(batch_size, channels, 1, 1)
        #     channel_att_avg = channel_att_avg.view(batch_size, channels, 1, 1)
        #     channel_att_max = channel_att_max.view(batch_size, channels, 1, 1)
        # else:
        #     channel_att_avg = channel_att_avg.view(channels, 1, 1)
        #     channel_att_max = channel_att_max.view(channels, 1, 1)
        channel_att = channel_att_avg + channel_att_max
        return x * channel_att

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels,  num_scales):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_attention = SpatialMultiScaleAttention(in_channels, num_scales)
        self.channel_attention = ChannelAttentionModified(in_channels)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x = F.relu(self.conv1(x))

        weight = self.spatial_attention(x) * self.channel_attention(x)
        # weight =  self.channel_attention(x)
        out_feature = weight * x
        # out_feature = weight + x
        # out_feature = out_feature + x #加残差
        return out_feature

    """
    #在这个特定的例子中，空间注意力机制 (self.spatial_attention(x)) 和通道注意力机制 (self.channel_attention(x))
    #都输出一个与输入特征图 (x) 形状相同的张量。这两个输出张量中的每个元素都是一个注意力权重，它们分别表示空间和通道维度上的注意力分布。
    #逐元相乘操作将这两个输出张量相乘，从而将空间和通道注意力权重结合在一起。这样，我们可以在一个操作中同时捕捉空间和通道维度上的上下文信息。
    #最后，将注意力权重逐元素地应用于输入特征图 x。这意味着每个特征图的值都会根据空间和通道注意力权重进行调整。
    #这样，经过注意力机制调整的特征图能更好地捕捉输入数据中的重要信息。
    """
if __name__ == "__main__":
    input = torch.randn(1, 128, 20, 20)
    models = MultiScaleAttention(128, 128, 1)
    print(models(input).shape)
"""
1. 空间多尺度注意力机制 (Spatial Multi-Scale Attention):

空间注意力机制关注的是输入特征图中每个位置的重要性。在这个例子中，我们实现了一个空间多尺度注意力机制，它可以捕获不同尺度下的上下文信息。这种注意力机制通过在不同空间范围内进行局部邻域操作来实现多尺度特性。在这个实现中，我们使用了具有不同 dilation 参数的卷积层。这样可以在不增加计算复杂度的情况下，扩大感受野。

对于每个尺度，我们计算一个注意力图。这些注意力图随后相加，从而形成一个综合的空间注意力图。通过逐元素乘法，将空间注意力图应用于输入特征图，从而突出显示在不同尺度下具有较高权重的区域。

2. 通道注意力机制 (Channel Attention):

通道注意力机制关注的是输入特征图中每个通道的重要性。在上述实现中，我们提出了一种修改后的通道注意力机制，它基于全局平均池化和全局最大池化，以及一个全连接层。与 Squeeze-and-Excitation (SE) 块相比，它减少了参数数量，从而实现了轻量化。

这个通道注意力机制首先计算输入特征图的全局平均池化和全局最大池化，然后将它们分别输入一个全连接层。这个全连接层将通道数量降低为原始通道数量除以一个固定的缩减比例（例如 16）。将平均池化和最大池化得到的两个注意力向量相加，从而形成一个综合的通道注意力向量。将通道注意力向量应用于输入特征图的通道维度，突出显示具有较高权重的通道。

3. 注意力机制的结合：

在上述实现的神经网络模型中，我们将空间多尺度注意力机制和修改后的通道注意力机制结合在一起。这是通过将两个注意力机制的输出逐元素相乘来实现的。这样，我们可以同时捕捉空间和通道维度上的上下文信息，从而使模型更好地关注输入数据中的重要特征。这种组合在某些情况下可能比单独使用空间或通道注意力机制产生更好的性能。
"""
"""
以下是上述注意力机制的理论公式：
空间多尺度注意力机制 (Spatial Multi-Scale Attention):
对于每个尺度 i (i = 1, 2, ..., num_scales)，我们计算空间注意力图 A_s_i 如下：
A_s_i = f_s_i(x)
其中 f_s_i(x) 是具有不同 dilation 参数的卷积运算，用于扩展感受野以捕获第 i 个尺度的上下文信息。
接下来，将所有尺度的空间注意力图相加，得到综合空间注意力图 A_s：
A_s = Σ A_s_i
通道注意力机制 (Channel Attention):
计算全局平均池化 (GAP) 和全局最大池化 (GMP)：
GAP_x = GAP(x)
GMP_x = GMP(x)
将 GAP_x 和 GMP_x 分别输入一个全连接层 f_c(x)，得到通道注意力向量 A_c_avg 和 A_c_max：
A_c_avg = f_c(GAP_x)
A_c_max = f_c(GMP_x)
将 A_c_avg 和 A_c_max 相加，得到综合通道注意力向量 A_c：
A_c = A_c_avg + A_c_max
注意力机制的结合：
将空间多尺度注意力图 A_s 和通道注意力向量 A_c 结合起来，形成一个综合注意力图 A：
A = A_s ⊙ A_c
其中 ⊙ 表示逐元素乘法。这个综合注意力图 A 既包含空间维度上的信息，也包含通道维度上的信息。
将注意力图 A 应用于输入特征图 x，得到输出特征图 y：
y = x ⊙ A
在这个实现中，我们直接输出经过注意力调整的特征图 y。
"""
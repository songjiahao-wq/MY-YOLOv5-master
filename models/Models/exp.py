import torch
import torch.nn as nn
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class LightweightMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, scale_factor=1.0):
        super(LightweightMLP, self).__init__()

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Channel scaling layer
        self.channel_scale = nn.Conv2d(in_channels, int(in_channels * scale_factor), 1)

        # MLP with one hidden layer
        self.fc1 = nn.Linear(int(in_channels * scale_factor), hidden_channels)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        # Apply global average pooling
        redual = x
        x = self.global_avg_pool(x)

        # Apply channel scaling
        x = self.channel_scale(x)

        x = x.view(x.size(0), -1)

        # Apply MLP
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = x.view(x.size(0), -1, 1, 1)  # Reshape x to match input feature map dimensions
        # x = x + redual  # Add the input feature map to the enhanced feature map

        return x
#Concat_bifpn系列*************************************************************************
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
#1
class Concat_bifpn_cat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
       # self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1 * 3, c2, 1 ,1 ,0 )
        self.act = MemoryEfficientSwish()
    def forward(self, x): # mutil-layer 1-3 layers #ADD or Concat
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(weight[0] * x[0] + weight[1] * x[1])
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(torch.cat(weight[0] * x[0] , weight[1] * x[1] , weight[2] * x[2]),dim=1)
        return x
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))

        max_out = self.max_pool(x)
        max_out = self.fc2(self.relu1(self.fc1(max_out)))

        out = torch.sigmoid(avg_out + max_out)
        return out

class Concat_bifpn(nn.Module):
    def __init__(self, c1, c2):
        super(Concat_bifpn, self).__init__()
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.channel_attention = ChannelAttention(c1)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x) == 2:
            attention_weights = self.channel_attention(torch.cat(x, dim=1))
            weight = attention_weights / (torch.sum(attention_weights, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            attention_weights = self.channel_attention(torch.cat(x, dim=1))
            weight = attention_weights / (torch.sum(attention_weights, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

        return x


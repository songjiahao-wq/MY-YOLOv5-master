import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv,FEM
from models.Models.Attention.my_attention import *
from models.Models.Attention.MultiScaleAttention import *
# from models.ExpCode.FEM import *
from models.Models.fighting_model.backbone.inceptionNext import inceptionnext_tiny

class CustomNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(CustomNetwork, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.group_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
        self.softmax = nn.Softmax(dim=1)

        self.group_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), groups=groups)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out1 = self.avgpool(residual)
        out1 = self.group_conv1(out1)
        out1 = self.softmax(out1)

        out2 = self.group_conv2(residual)
        out2 = self.norm(out2)
        out2 = self.relu(out2)

        out = out1 * out2

        return out
m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = CustomNetwork(256,256 ,256)
results = profile(input=torch.randn(1, 256, 40, 40), ops=[m1, m2], n=1)




# Example usage
# input_tensor = torch.randn(8, 256, 32, 32)  # batch_size=8, in_channels=256, height=32, width=32
# model = FeatureEnhancementModule(in_channels=256)
# output_tensor = model(input_tensor)


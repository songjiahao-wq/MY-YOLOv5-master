import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv,FEM
from models.Models.Attention.my_attention import *
from models.Models.Attention.MultiScaleAttention import *
# from models.ExpCode.FEM import *
from models.Models.fighting_model.backbone.inceptionNext import inceptionnext_tiny
class ConvStackSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_layers=3):
        super(ConvStackSpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)
        ])
        self.conv_final = nn.Conv2d(in_channels, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_final(x)
        return self.sigmoid(x)
class DilatedConvSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation_rate=2):
        super(DilatedConvSpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation_rate
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size, padding=padding, dilation=dilation_rate, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        return self.sigmoid(x)

m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = DilatedConvSpatialAttention(256)
results = profile(input=torch.randn(1, 256, 40, 40), ops=[m1, m2], n=1)




# Example usage
# input_tensor = torch.randn(8, 256, 32, 32)  # batch_size=8, in_channels=256, height=32, width=32
# model = FeatureEnhancementModule(in_channels=256)
# output_tensor = model(input_tensor)


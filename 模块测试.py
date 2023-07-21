import torch
import torch.nn as nn
from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv

class half_conv(nn.Module):
    def __init__(self,c1, c2, k=1, s=1):
        super(half_conv, self).__init__()
        self.c_ = c1 // 2
        self.conv3x3 = nn.Conv2d(self.c_, self.c_, 3, 1, 3//2)
        self.conv1x1 = nn.Conv2d(c1, self.c_, 1, 1)
    def forward(self, x):
        y = torch.split(x, [self.c_, self.c_], dim=1)
        identity1 = y[1]
        identity2 = y[0]
        y1 =self.conv3x3(y[0])
        out = torch.cat((y1, identity1), dim=1)
        halfout = self.conv1x1(out)

        return torch.cat((halfout, identity2), 1)

m1 = Conv(512, 512, 3, 1)
m2 = half_conv(512, 512, 3, 1)
results = profile(input=torch.randn(16, 512, 20, 20), ops=[m1, m2], n=3)

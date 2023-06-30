import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv

m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = Conv(128, 256, 3, 1)
results = profile(input=torch.randn(16, 128, 80, 80), ops=[m1, m2], n=3)

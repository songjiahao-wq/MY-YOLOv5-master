import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv
from models.Models.Attention.my_attention import *
m1 = MixConv2d(128, 256, (3, 5), 1)
m2 = PSAMix(128, 128)
results = profile(input=torch.randn(16, 128, 20, 20), ops=[m1, m2], n=3)


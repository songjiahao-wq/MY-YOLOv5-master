import torch

from utils.torch_utils import profile
from models.experimental import MixConv2d
from models.common import Conv
from models.Models.Attention.my_attention import *
from models.Models.Attention.MultiScaleAttention import *

m1 = MixConv2d(256, 256, (3, 5), 1)
m2 = LightweightSPPFA(256, 256)
results = profile(input=torch.randn(1, 256, 20, 20), ops=[m1, m2], n=3)


import torch
import numpy as np
import torchvision
import torch

# print(torch.__version__)
#
# print(torch.version.cuda)

print(torch.backends.cudnn.version())
# device = torch.device("cuda")
# model = torchvision.models.__dict__['efficientnet_b0'](pretrained=False).to(device).eval()
# print(model)
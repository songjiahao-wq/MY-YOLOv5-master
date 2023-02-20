import torch
import numpy as np
import torchvision

device = torch.device("cuda")
model = torchvision.models.__dict__['efficientnet_b0'](pretrained=False).to(device).eval()
print(model)
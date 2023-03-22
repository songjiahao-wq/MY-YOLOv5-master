import torch


input1 = torch.randn(1,3,4,4)
print(input1)
# input2 = torch.randn(3,64,32,32)
out = input1.reshape([1,3,2,2])
print(out.shape)
# import torch
# import torch.nn as nn
#
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, k=3, s=1, p=None):
#         super(Conv, self).__init__()
#         if p is None:
#             p = k // 2
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
# class FEM(nn.Module):
#     def __init__(self, in_channels, out_channels, d=[1, 3, 5], kernel_sizes=[1, 3, 5]):
#         super(FEM, self).__init__()
#         self.cv1 = Conv(in_channels, out_channels)
#         self.branches = nn.ModuleList()
#         for i in range(len(kernel_sizes)):
#             padding = kernel_sizes[i] // 2
#             branch = nn.Sequential(
#                 nn.Conv2d(out_channels, out_channels, kernel_size=kernel_sizes[i], stride=1, padding=padding + d[i], dilation=d[i]  , groups=16),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )
#             self.branches.append(branch)
#         self.cv2 = Conv(out_channels * len(kernel_sizes), in_channels, k=1)
#
#     def forward(self, x):
#         # x: input tensor of shape (batch_size, in_channels, height, width)
#         residual = x
#         x = self.cv1(x)
#         # Apply different kernel sizes
#         outputs = []
#         for branch in self.branches:
#             out = branch(x)
#             outputs.append(out)
#         # Concatenate the outputs along the channel dimension
#         out = torch.cat(outputs, dim=1)
#         out = self.cv2(out)
#         return out
#
# # Example usage:
# in_channels = 256
# out_channels = 128
# fem = FEM(in_channels, out_channels)
#
# input_tensor = torch.randn(1, in_channels, 32, 32)
# output_tensor = fem(input_tensor)
# print(output_tensor.shape)  # torch.Size([1, 256, 32, 32])

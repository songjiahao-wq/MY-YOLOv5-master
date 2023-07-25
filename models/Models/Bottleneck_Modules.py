from models.module import SEAttention
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, custom_conv=None):
        super().__init__()
        if custom_conv is None:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        else:
            self.conv = custom_conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.act = nn.Identity() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Tanh() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Sigmoid() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = AconC(c2) iMemoryEfficientMisf act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = MetaAconC(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = SiLU_beta(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU_noBN_biasFalse(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU_noBN_biasTrue(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = DyReLUA(c2, conv_type='2d') if act is True else (act if isinstance(act, nn.Module) else nn.Identity())#参数量大一些
        # self.act = DyReLUB(c2, conv_type='2d') if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # 比较看好的激活函数是DyReLU和meta - AconC这两个激活函数
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
"""
# RepViT 块设计的优化 https://mp.weixin.qq.com/s/rk9GHuhyXfsbOgMRxefz_Q
"""


class Bottleneck_1(nn.Module):
    def __init__(self, inp, oup, shortcut, e, kernel_size, stride, use_se, use_hs):
        super(Bottleneck_1, self).__init__()
        assert stride in [1, 2]

        self.identity = shortcut
        hidden_dim = int(inp * e)  # hidden channels
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if use_hs else nn.SiLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # Squeeze-and-Excite
            SEAttention(hidden_dim) if use_se else nn.Sequential(),
            nn.Hardswish() if use_hs else nn.SiLU(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class Bottleneck_2(nn.Module):
    def __init__(self, inp, oup, shortcut, e, kernel_size, stride, use_se, use_hs):
        super(Bottleneck_2, self).__init__()
        assert stride in [1, 2]

        self.identity = shortcut
        hidden_dim = int(inp * e)  # hidden channels
        self.DWconv1x1 = DWConv(inp, oup, kernel_size, stride)
        self.DWconv3x3 = DWConv(inp, oup, kernel_size, stride)
        # Squeeze-and-Excite
        self.ATT = SEAttention(inp) if use_se else nn.Identity
        self.conv2 = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if use_hs else nn.SiLU(),

            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        y = self.DWconv1x1(x) + self.DWconv3x3(x) + x
        yatt = self.ATT(y)
        out = self.conv2(yatt)
        return out

class Bottleneck_3(nn.Module):
    def __init__(self, inp, oup, shortcut, e, kernel_size, stride, use_se, use_hs):
        super(Bottleneck_3, self).__init__()
        assert stride in [1, 2]

        self.identity = shortcut
        hidden_dim = int(inp * e)  # hidden channels
        self.DWconv3x3 = DWConv(inp, oup, kernel_size, stride)
        # Squeeze-and-Excite
        self.ATT = SEAttention(inp) if use_se else nn.Identity
        self.conv2 = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if use_hs else nn.SiLU(),

            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
       y = self.DWconv3x3(x)
       out = self.conv2(y) + y
       return out

class C3_RepViT1(C3):
    def __init__(self, c1, c2, use_se=False, use_hs=False, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2,  n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_1(c_, c_, shortcut, e=1.0, kernel_size=3,
                                              stride=1, use_se=use_se, use_hs=use_hs) for _ in range(n)))
class C3_RepViT2(C3):
    def __init__(self, c1, c2, use_se=False, use_hs=False, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2,  n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_2(c_, c_, shortcut, e=1.0, kernel_size=3,
                                              stride=1, use_se=use_se, use_hs=use_hs) for _ in range(n)))
class C3_RepViT3(C3):
    def __init__(self, c1, c2, use_se=False, use_hs=False, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2,  n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_3(c_, c_, shortcut, e=1.0, kernel_size=3,
                                              stride=1, use_se=use_se, use_hs=use_hs) for _ in range(n)))
class C3_Expand(C3):
    # C3 module with vanillaBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

"""
# CVPR2023 ScSCConv https://mp.weixin.qq.com/s/KQV_gfU5yQVa33sXIVueMA
"""
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = F.softmax(self.gn.gamma, dim=0)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = w_gamma > self.gate_treshold
        noninfo_mask = w_gamma <= self.gate_treshold
        x_1 = info_mask * reweigts * x
        x_2 = noninfo_mask * reweigts * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self,
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up, low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up, low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        # 先mul后split
        # out     = F.softmax( self.advavg(out), dim=1 ) * out
        # out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        # 先split后mul
        out1, out2 = torch.split(out, out.size(1)//2, dim=1)
        out, out2 = out1 * Y1, out2 * Y2
        return out1+out2

class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class Bottleneck_ScSCConv(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ScConv(c_)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C3_ScSCConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2,  n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_ScSCConv(c_, c_, shortcut, e=1.0) for _ in range(n)))

# Conv------------------------------------------------------------------------------------------------------------------
class slim_conv_3x3(nn.Module):

    def __init__(self, in_planes, stride, groups, dilation):
        super(slim_conv_3x3, self).__init__()
        self.stride = stride

        reduce_1 = 2
        reduce_2 = 4

        self.conv2_2 = nn.Sequential(nn.Conv2d(in_planes // reduce_1, in_planes // reduce_2, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes // reduce_2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_planes // reduce_2, in_planes // reduce_2, kernel_size=3,
                                               stride=stride, groups=groups, padding=dilation, bias=False,
                                               dilation=dilation),
                                     nn.BatchNorm2d(in_planes // reduce_2))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_planes // reduce_1, in_planes // reduce_1, kernel_size=3, stride=stride, groups=groups,
                      padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(in_planes // reduce_1))

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 32, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_planes // 32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // 32, in_planes, kernel_size=1),
                                nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x
        b, c, h, _ = out.size()

        w = self.pool(out)
        w = self.fc(w)
        w_f = torch.flip(w, [1])

        out1 = w * out
        out2 = w_f * out
        fs1 = torch.split(out1, c // 2, 1)
        fs2 = torch.split(out2, c // 2, 1)

        ft1 = fs1[0] + fs1[1]
        ft2 = fs2[0] + fs2[1]

        out2_1 = self.conv2_1(ft1)
        out2_2 = self.conv2_2(ft2)

        out = torch.cat((out2_1, out2_2), 1)
        return out


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V

class slim_conv_SK(nn.Module):

    def __init__(self, in_planes, out_planes,k, stride=1, groups=16, dilation=1, L=32, r=16):
        super(slim_conv_SK, self).__init__()
        self.stride = stride
        self.features = in_planes
        reduce_1 = 2
        reduce_2 = 4
        d = max(int(in_planes / r), L)
        if in_planes ==out_planes:
            out_planes =  in_planes // reduce_1
        else:
            out_planes = out_planes //2
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_planes // reduce_1, in_planes // reduce_2, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes // reduce_2),
                                     nn.SiLU(inplace=True),
                                     nn.Conv2d(in_planes // reduce_2, out_planes, kernel_size=3,
                                               stride=stride, groups=groups, padding=dilation, bias=False,
                                               dilation=dilation),
                                     nn.BatchNorm2d(out_planes))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_planes // reduce_1, out_planes, kernel_size=3, stride=stride, groups=groups,
                      padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_planes))

        self.fc = nn.Sequential(nn.Conv2d(in_planes, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.SiLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Conv2d(d, in_planes, kernel_size=1, stride=1)
            )

        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        out = x
        b, c, h, _ = out.size()

        fea_s = self.gap(out)
        fea_z = self.fc(fea_s)
        attention_vectors = [fc(fea_z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(b, 2, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats = torch.cat((x, x), dim=1)
        feats = feats.view(b, 2, self.features, feats.shape[2], feats.shape[3])

        out = attention_vectors * feats
        out1, out2 = torch.split(out, 1, 1)
        fs1 = torch.split(out1.squeeze(1), c // 2, 1)
        fs2 = torch.split(out2.squeeze(1), c // 2, 1)

        ft1 = fs1[0] + fs1[1]
        ft2 = fs2[0] + fs2[1]

        out2_1 = self.conv2_1(ft1)
        out2_2 = self.conv2_2(ft2)

        out = torch.cat((out2_1, out2_2), 1)
        return out




if __name__ == '__main__':
    x = torch.randn(2, 512, 20, 20)
    model = slim_conv_SK(512, 1024, 2)
    print(model(x).shape)
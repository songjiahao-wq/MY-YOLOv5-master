from models.Models.yolovii import Conv, RobustConv, RobustConv2, GhostConv, RepConv, RepConv_OREPA, DownC,  \
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, Focus, Stem, GhostStem,  \
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,  \
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,   \
                 Res, Swin_v2_A, Swin_v2_B, Swin_v2_C, Swin_Transformer_A, Swin_Transformer_B, Swin_Transformer_C, ResCSPA, ResCSPB, ResCSPC,  \
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,  \
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,  \
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,  \
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC, \
                 SwinTransformerBlock, \
                 SwinTransformer2Block, \
                SCConv,Bottleneck_DCN, FusionModel
from models.Models.fighting_model.backbone.inceptionNext import InceptionNext_small
from models.Models.fighting_model.backbone.convnextv2 import convnextv2_att

from models.Models.Attention.SPPA_CBAM import SPPA_CBAM
from models.Models.Attention.SPPAF import C3SPPAF
from models.Models.Litemodel import ADD
from models.Models.research import Concat_bifpn
from models.Models.Litemodel import ADD,ConvBNReLU, InvertedResidual

from models.Models.Attention.SoftThresholdAttention import *
from models.Models.Attention.my_attention import PSAModule_s, SPPFC,PSAMix
from models.Models.Attention.MultiScaleAttention import MultiScaleAttention,MultiScaleAttentionCSPA,MultiScaleAttentionCSPB,MultiScaleFrequencyAttention,\
                    FrequencyAttention, MultiScaleFreqAttention,SoftThresholdAttentionResidual

from add.cv_attention.Acmix import ACmix
from add.cv_attention.CBAM import CBAMBlock as CBAM, CBAMBlock2
from models.Models.fighting_model.attention.SEAttention import SEAttention #se = SEAttention(channel=512,reduction=8)
from models.Models.fighting_model.attention.SKAttention import SKAttention #se = SKAttention(channel=512,reduction=8)
from models.Models.fighting_model.attention.BAM import BAMBlock #bam = BAMBlock(channel=512,reduction=16,dia_val=2)
from models.Models.fighting_model.attention.ECAAttention import ECAAttention #eca = ECAAttention(kernel_size=3)
from models.Models.fighting_model.attention.DANet import DAModule #danet=DAModule(d_models.Models.fighting_model=512,kernel_size=3,H=7,W=7)
from models.Models.fighting_model.attention.PSA import PSA #psa = PSA(channel=512,reduction=8)
from models.Models.fighting_model.attention.EMSA import EMSA #emsa = EMSA(d_models.Models.fighting_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
from models.Models.fighting_model.attention.ShuffleAttention import ShuffleAttention # se = ShuffleAttention(channel=512,G=8)
from models.Models.fighting_model.attention.SGE import SpatialGroupEnhance #sge = SpatialGroupEnhance(groups=8)
from models.Models.fighting_model.attention.A2Atttention import DoubleAttention #a2 = DoubleAttention(512,128,128,True)
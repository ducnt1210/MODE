from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .cpa_arch import CPA_arch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class PromptLangRestormer(BaseModule):
    def __init__(self,c_in=3,c_out=3,dim=32):
        super(PromptLangRestormer, self).__init__()
        self.prompt_unet_arch = CPA_arch(c_in, c_out, dim)
    def forward(self,x: Tensor, l: Tensor, l_mask: Tensor): # (N, C, H, W), (N, 1, seq_length, feature dimension), (N, 1, seq_length)
        l = l.squeeze(1).permute(0, 2, 1)
        l_mask = l_mask.permute(0, 2, 1)
        x_=x
        x=self.prompt_unet_arch(x, l, l_mask)
        x=x+x_
        return x

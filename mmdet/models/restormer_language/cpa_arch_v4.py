import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from calflops import calculate_flops
from .attention import VisionTextAttention
from .InstructIR import masked_mean_pooling, ICB
from .mask_block import RGBD, ConvBnLeakyRelu2d

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):  # (b,c,h,w)
        return self.body(x)  # (b,c/2,h*2,w*2)

class InteractionBlock(nn.Module):
    def __init__(self, v_in_channels, l_in_channels):
        super(InteractionBlock, self).__init__()

        self.vis_project = nn.Sequential(nn.Conv1d(v_in_channels, v_in_channels, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                        )

        self.fusion = VisionTextAttention(v_in_channels,
                                          l_in_channels,
                                          v_in_channels,
                                          v_in_channels,
                                          out_channels=v_in_channels,
                                          num_heads=1)
        
        self.project_mm = nn.Sequential(nn.Conv1d(v_in_channels, v_in_channels, 1, 1),
                                        nn.GELU(),
                                        )
        
        self.res_gate = nn.Sequential(
            nn.Linear(v_in_channels, v_in_channels, bias=False),
            nn.ReLU(),
            nn.Linear(v_in_channels, v_in_channels, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x, l, l_mask): # (b,c,h,w), (b, f_dim, seq_len), (b, seq_len, 1)
        B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.flatten(2) # B, C, H*W
        vis = self.vis_project(x) # (b,c,h*w)
        x = x.permute(0, 2, 1) # B, H*W, C
        lang = self.fusion(x, l, l_mask) # (B, HW, C)
        lang = lang.permute(0, 2, 1)

        vis = torch.mul(vis, lang) # (B, C, H*W)
        vis = self.project_mm(vis) # (B, C, H*W)

        vis = vis.permute(0,2,1) # B, H*W, C

        x = x + (self.res_gate(vis) * vis) # B, H*W, C
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x

#########################################################################
# CPA_Enhancer
class CPA_arch(nn.Module):
    def __init__(self, c_in, c_out, dim, m_dim = [1, 24, 48, 96, 192]):
        super(CPA_arch, self).__init__()
        self.conv0 = RFAConv(c_in, dim)
        self.conv1 = RFAConv(dim, dim)
        self.conv2 = RFAConv(dim * 2, dim * 2)
        self.conv3 = RFAConv(dim * 4, dim * 4)
        self.conv4 = RFAConv(dim * 8, dim * 8)
        self.conv5 = RFAConv(dim * 8, dim * 4)
        self.conv6 = RFAConv(dim * 4, dim * 2)
        self.conv7 = RFAConv(dim * 2, c_out)

        self.mask0 = RGBD(m_dim[0], m_dim[1])
        self.mask05 = RGBD(m_dim[1], m_dim[1])
        self.mask1 = RGBD(m_dim[2], m_dim[2])
        self.mask2 = RGBD(m_dim[3], m_dim[3])
        self.mask3 = RGBD(m_dim[4], m_dim[4])

        self.fused0 = ConvBnLeakyRelu2d(dim + m_dim[1], dim)
        self.fused1 = ConvBnLeakyRelu2d(dim*2 + m_dim[2], dim*2)
        self.fused2 = ConvBnLeakyRelu2d(dim*4 + m_dim[3], dim*4)
        self.fused3 = ConvBnLeakyRelu2d(dim*8 + m_dim[4], dim*8)

        self.m_down1 = Downsample(m_dim[1])
        self.m_down2 = Downsample(m_dim[2])
        self.m_down3 = Downsample(m_dim[3])

        self.down1 = Downsample(dim)
        self.down2 = Downsample(dim * 2)
        self.down3 = Downsample(dim * 4)

        self.icb1 = ICB(dim, 768)
        self.icb2 = ICB(dim * 2, 768)
        self.icb3 = ICB(dim * 4, 768)
        self.icb4 = ICB(dim * 8, 768)

        self.interaction5 = InteractionBlock(dim * 8, 768)
        self.interaction6 = InteractionBlock(dim * 4, 768)
        self.interaction7 = InteractionBlock(dim * 2, 768)

        self.up3 = Upsample(dim * 8)
        self.up2 = Upsample(dim * 4)
        self.up1 = Upsample(dim * 2)

    def forward(self, x, l, l_mask, obj_mask):  # (b,c_in,h,w), (b, f_dim, seq_len), (b, seq_len, 1)
        l_pool = masked_mean_pooling(l, l_mask)

        x0 = self.conv0(x)  # (b,dim,h,w)
        x1 = self.conv1(x0)  # (b,dim,h,w)
        obj_mask = self.mask0(obj_mask)
        obj_mask = self.mask05(obj_mask)
        x1 = self.icb1(x1, l_pool)
        x1 = self.fused0(torch.cat([x1, obj_mask], dim=1))
        x1_down = self.down1(x1)  # (b,dim,h/2,w/2)
        obj_mask = self.m_down1(obj_mask)

        x2 = self.conv2(x1_down)  # (b,dim,h/2,w/2)
        obj_mask = self.mask1(obj_mask)
        x2 = self.icb2(x2, l_pool)
        x2 = self.fused1(torch.cat([x2, obj_mask], dim=1))
        x2_down = self.down2(x2)
        obj_mask = self.m_down2(obj_mask)

        x3 = self.conv3(x2_down)
        obj_mask = self.mask2(obj_mask)
        x3 = self.icb3(x3, l_pool)
        x3 = self.fused2(torch.cat([x3, obj_mask], dim=1))
        x3_down = self.down3(x3)
        obj_mask = self.m_down3(obj_mask)

        x4 = self.conv4(x3_down)
        obj_mask = self.mask3(obj_mask)
        x4 = self.icb4(x4, l_pool)
        x4 = self.fused3(torch.cat([x4, obj_mask], dim=1))
        
        x4 = self.interaction5(x4, l, l_mask)
        x4 = self.up3(x4)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x5 = self.interaction6(x5, l, l_mask)
        x5 = self.up2(x5)
        x6 = self.conv6(torch.cat([x5, x2], 1))
        x6 = self.interaction7(x6, l, l_mask)
        x6 = self.up1(x6)
        x7 = self.conv7(torch.cat([x6, x1], 1))
        return x7

class RFAConv(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w 
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

def main():
    model = CPA_arch(c_in=3, c_out=3, dim=24)
    x = torch.rand(1, 3, 320, 640)
    l = torch.rand(1, 768, 25)
    l_mask = torch.cat([torch.ones(1, 5, 1), torch.zeros(1, 5, 1), torch.ones(1, 10, 1), torch.zeros(1, 5, 1)], dim=1)
    obj_mask = torch.rand(1, 1, 320, 640)
    # l_mask = torch.cat([torch.ones(1, 100, 1), torch.zeros(1, 200, 1)], dim=1)

    inputs = {}
    inputs['x'] = x
    inputs['l'] = l
    inputs['l_mask'] = l_mask
    inputs['obj_mask'] = obj_mask

    flops, macs, params = calculate_flops(model=model,
                                      kwargs = inputs,
                                      print_results=True, print_detailed=False)
    print("CPASLViT FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    x = model(x, l, l_mask, obj_mask)
    # print(outs)
    print(x.shape)
if __name__ == "__main__":
    main()
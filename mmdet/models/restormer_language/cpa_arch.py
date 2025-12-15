import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
from .SLViT_encoder import SLViT_Encoder

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

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):  # x:[b,c,h,w]
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (b,1,h,w)
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # (b,1,h,w)
        x2 = torch.concat([x_avg, x_max], dim=1)  # (b,2,h,w)
        sattn = self.sa(x2)  # 7x7conv (b,1,h,w)
        return sattn * x

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),  # Relu
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):  # x:[b,c,h,w]
        x_gap = self.gap(x)  #  [b,c,1,1]
        cattn = self.ca(x_gap)  # [b,c,1,1]
        return cattn * x

class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        result = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return result

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res

#########################################################################
# Chain-of-Thought Prompt Generation Module (CGM)
class CotPromptParaGen(nn.Module):
    def __init__(self,prompt_inch,prompt_size, num_path=2):
        super(CotPromptParaGen, self).__init__()

        # (128,32,32)->(64,64,64)->(32,128,128)
        self.chain_prompts=nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=prompt_inch if idx==0 else prompt_inch//(2**idx),
                out_channels=prompt_inch//(2**(idx+1)),
                kernel_size=3, stride=2, padding=1
            ) for idx in range(num_path)
        ])
    def forward(self,x):
        prompt_params = []
        prompt_params.append(x)
        for pe in self.chain_prompts:
            x=pe(x)
            prompt_params.append(x)
        return prompt_params

#########################################################################
# Content-driven Prompt Block (CPB)
class ContentDrivenPromptBlock(nn.Module):
    def __init__(self, dim, prompt_dim, reduction=8, num_splits=4):
        super(ContentDrivenPromptBlock, self).__init__()
        self.dim = dim
        self.num_splits = num_splits
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(dim, prompt_dim, kernel_size=1, stride=1, bias=False)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.myshuffle = Channel_Shuffle(2)
        self.out_conv1 = nn.Conv2d(prompt_dim + dim, dim, kernel_size=1, stride=1, bias=False)
        self.transformer_block = nn.Sequential(*[
            TransformerBlock(dim=dim // num_splits, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for _ in range(num_splits)])

    def forward(self, x, prompt_param):  
        # latent: (b,dim*8,h/8,w/8)  prompt_param3: (1, 256, 16, 16)
        x_ = x
        B, C, H, W = x.shape
        cattn = self.ca(x)  # channel-wise attn
        sattn = self.sa(x)  # spatial-wise attn
        pattn1 = sattn + cattn  
        pattn1 = pattn1.unsqueeze(dim=2)  # [b,c,1,h,w]
        x = x.unsqueeze(dim=2)  # [b,c,1,h,w]
        x2 = torch.cat([x, pattn1], dim=2)  #  [b,c,2,h,w]
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)  # [b,c*2,h,w]
        x2 = self.myshuffle(x2)  # [c1,c1_att,c2,c2_att,...]
        pattn2 = self.pa2(x2)  
        pattn2 = self.conv1x1(pattn2)  # [b,prompt_dim,h,w]
        prompt_weight = self.sigmoid(pattn2)  # Sigmod

        prompt_param = F.interpolate(prompt_param, (H, W), mode="bilinear")
        # (b,prompt_dim,prompt_size,prompt_size) -> (b,prompt_dim,h,w)
        prompt = prompt_weight * prompt_param
        prompt = self.conv3x3(prompt)  # (b,prompt_dim,h,w)

        inter_x = torch.cat([x_, prompt], dim=1)  # (b,prompt_dim+dim,h,w)
        inter_x = self.out_conv1(inter_x)  # (b,dim,h,w) dim=64
        splits = torch.split(inter_x, self.dim // self.num_splits, dim=1)

        transformered_splits = []
        for i, split in enumerate(splits):
            transformered_split = self.transformer_block[i](split)
            transformered_splits.append(transformered_split)
        result = torch.cat(transformered_splits, dim=1)
        return result

#########################################################################
# CPA_Enhancer
class CPA_arch(nn.Module):
    def __init__(self, c_in, c_out, dim, prompt_inch=128, prompt_size=32):
        super(CPA_arch, self).__init__()
        # self.conv0 = RFAConv(c_in, dim)
        # self.conv1 = RFAConv(dim, dim)
        # self.conv2 = RFAConv(dim * 2, dim * 2)
        # self.conv3 = RFAConv(dim * 4, dim * 4)
        # self.conv4 = RFAConv(dim * 8, dim * 8)
        self.encoder = SLViT_Encoder(in_chans=c_in, embed_dims=[dim, dim * 2, dim * 4, dim * 8])
        self.conv5 = RFAConv(dim * 8, dim * 4)
        self.conv6 = RFAConv(dim * 4, dim * 2)
        self.conv7 = RFAConv(dim * 2, c_out)

        # self.down1 = Downsample(dim)
        # self.down2 = Downsample(dim * 2)
        # self.down3 = Downsample(dim * 4)

        self.prompt_param_ini = nn.Parameter(torch.rand(1, prompt_inch, prompt_size, prompt_size)) # (b,c,h,w)
        self.myPromptParamGen = CotPromptParaGen(prompt_inch=prompt_inch,prompt_size=prompt_size)
        self.prompt1 = ContentDrivenPromptBlock(dim=dim * 2 ** 1, prompt_dim=prompt_inch // 4, reduction=8)  # !!!!
        self.prompt2 = ContentDrivenPromptBlock(dim=dim * 2 ** 2, prompt_dim=prompt_inch // 2, reduction=8)
        self.prompt3 = ContentDrivenPromptBlock(dim=dim * 2 ** 3, prompt_dim=prompt_inch , reduction=8)

        self.up3 = Upsample(dim * 8)
        self.up2 = Upsample(dim * 4)
        self.up1 = Upsample(dim * 2)

    def forward(self, x, l, l_mask):  # (b,c_in,h,w), (b, f_dim, seq_len), (b, seq_len, 1)
        prompt_params = self.myPromptParamGen(self.prompt_param_ini)
        prompt_param1 = prompt_params[2] # [1, 64, 64, 64]
        prompt_param2 = prompt_params[1]  # [1, 128, 32, 32]
        prompt_param3 = prompt_params[0]  # [1, 256, 16, 16]
        # print(prompt_param3.shape,prompt_param2.shape,prompt_param1.shape)
        # x0 = self.conv0(x)  # (b,dim,h,w)
        # x1 = self.conv1(x0)  # (b,dim,h,w)
        # x1_down = self.down1(x1)  # (b,dim,h/2,w/2)
        # x2 = self.conv2(x1_down)  # (b,dim,h/2,w/2)
        # x2_down = self.down2(x2)
        # x3 = self.conv3(x2_down)
        # x3_down = self.down3(x3)
        # x4 = self.conv4(x3_down)
        outs, _ = self.encoder(x, l, l_mask)
        x1, x2, x3, x4 = outs
        x4_prompt = self.prompt3(x4, prompt_param3)
        x3_up = self.up3(x4_prompt)
        x5 = self.conv5(torch.cat([x3_up, x3], 1))
        x5_prompt = self.prompt2(x5, prompt_param2)
        x2_up = self.up2(x5_prompt)
        x2_cat = torch.cat([x2_up, x2], 1)
        x6 = self.conv6(x2_cat)
        x6_prompt = self.prompt1(x6, prompt_param1)
        x1_up = self.up1(x6_prompt)
        x7 = self.conv7(torch.cat([x1_up, x1], 1))
        return  x7

class GatedConv(nn.Module):
    def __init__(self, dim, ks):
        super(GatedConv, self).__init__()

        if ks == 5:
            pad = 2
        elif ks == 3:
            pad = 1
        elif ks==7:
            pad = 3

        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=ks, padding=pad, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=ks, padding=pad, groups=dim)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        gx2 = self.act(x2)
        y = x1 * gx2
        return y

# Prompt 生成模块
class MultiscalePromptGenBlock(nn.Module):
    def __init__(self, dim, prompt_dim, reduction=8):
        super(MultiscalePromptGenBlock, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(dim, prompt_dim, kernel_size=1, stride=1, bias=False)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.myshuffle=Channel_Shuffle(2)
        self.out_conv1=nn.Conv2d(prompt_dim+dim,dim, kernel_size=1, stride=1, bias=False)
        self.dim=dim
        self.gconv7 = GatedConv(dim, 7)
        self.gconv5 = GatedConv(dim, 5)
    def forward(self, x, prompt_param):  # x: 原始图像 patt1n: 粗糙的SIM
        # latent: (b,dim*8,h/8,w/8)  prompt_param3: (1, 256, 16, 16)
        x_=x
        B, C, H, W = x.shape
        cattn = self.ca(x)  # 通道注意力
        sattn = self.sa(x)  # 空间注意力
        pattn1 = sattn + cattn  # 粗糙的SIM [b,c,h,w]

        pattn1 = pattn1.unsqueeze(dim=2)  # [b,c,1,h,w]
        x = x.unsqueeze(dim=2)  # [b,c,1,h,w]
        x2 = torch.cat([x, pattn1], dim=2)  # 连接 [b,c,2,h,w]
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)  # [b,c*2,h,w]

        # 添加Channel_Shuffle !!
        x2=self.myshuffle(x2)  # 分组为2时，suffle后的通道[c1,c1_att,c2,c2_att,...]，一共2c
        splits = torch.split(x2, split_size_or_sections=self.dim, dim=1) # 列表[b,c,h,w]
        p5 = self.gconv5(splits[0])
        p7 = self.gconv7(splits[1])
        pattn2 = p5+p7
        # pattn2 = self.pa2(x2)  # 7x7的分组卷积 [b,c,h,w]  分组卷积group=c
        # 原始图片每一组大小[b,2,h,w], 共c个组。对应组有1个（c/c=1）2x7x7的卷积核, 卷积之后[b,1,h,w] 相当于是对该通道的卷积
        # 相当于对每个通道分配唯一的SIM
        # C组卷积结果拼接起来之后，得到[b,c,h,w]
        pattn2 = self.conv1x1(pattn2)  # [b,prompt_dim,h,w]
        prompt_weight = self.sigmoid(pattn2)  # Sigmod

        prompt_param = F.interpolate(prompt_param, (H, W), mode="bilinear")
        # (b,prompt_dim,prompt_size,prompt_size) -> (b,prompt_dim,h,w)
        prompt = prompt_weight * prompt_param
        prompt = self.conv3x3(prompt)  # (b,prompt_dim,h,w)

        inter_x=torch.cat([x_,prompt],dim=1) # (b,prompt_dim+dim,h,w)
        inter_x=self.out_conv1(inter_x) # (b,dim,h,w)
        return inter_x # (b,dim,h,w)
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

class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class RFCBAMConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
            )
        self.get_weight = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.se = SE(in_channel)

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel), nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class RFCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


def main():
    model = CPA_arch(c_in=3, c_out=3, dim=32)
    x = torch.rand(1, 3, 320, 640)
    l = torch.rand(1, 768, 300)
    l_mask = torch.rand(1, 300, 1)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # If you want only trainable parameters, use:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    x = model(x, l, l_mask)
    # print(outs)
    print(x.shape)


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from calflops import calculate_flops
    
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
    
# class ConvBnLeakyRelu2d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBnLeakyRelu2d, self).__init__()
#         self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

#         self.conv0_1 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1), groups=out_channels)
#         self.conv0_2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), groups=out_channels)

#         self.conv1_1 = nn.Conv2d(out_channels, out_channels, (1, 5), padding=(0, 2), groups=out_channels)
#         self.conv1_2 = nn.Conv2d(out_channels, out_channels, (5, 1), padding=(2, 0), groups=out_channels)

#         self.conv2_1 = nn.Conv2d(out_channels, out_channels, (1, 7), padding=(0, 3), groups=out_channels)
#         self.conv2_2 = nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), groups=out_channels)

#         # self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=2)
    
#     def forward(self, x):
#         x = self.conv0(x)
        
#         # x1 = self.conv0_1(x)
#         # x1 = self.conv0_2(x1)

#         # x2 = self.conv1_1(x)
#         # x2 = self.conv1_2(x2)

#         # x3 = self.conv2_1(x)
#         # x3 = self.conv2_2(x3)

#         x = x + self.conv0_2(self.conv0_1(x)) + self.conv1_2(self.conv1_1(x)) + self.conv2_2(self.conv2_1(x))

#         return F.leaky_relu(x, negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(channels, channels)
        # self.conv1 = MultiReceptiveConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvBnLeakyRelu2d(2*channels, channels)
        # self.conv2 = MultiReceptiveConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, gelu=False, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, gelu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale

class SpatialInteraction(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialInteraction, self).__init__()
        self.gate_vis = SpatialGate(kernel_size)
        self.gate_ir = SpatialGate(kernel_size)

    def forward(self, vis, ir):
        vis_gate = 1 - self.gate_vis(vis)
        ir_gate = 1 - self.gate_ir(ir)

        return vis * ir_gate, ir * vis_gate

def main():
    input_dim = 256
    # model = RegionAttention(n_feat=input_dim, num_heads=1)
    # model = TransformerBlock(input_dim, 1, ffn_expansion_factor=2, bias=True)
    # model = SpatialRouting(input_dim)
    x = torch.rand(1, input_dim, 40, 80)
    mask = torch.rand(1, 1, 40, 80)

    inputs = {}
    inputs['mask'] = mask
    inputs['x'] = x


    # flops, macs, params = calculate_flops(model=model,
    #                                   kwargs = inputs,
    #                                   print_results=True, print_detailed=False)
    # print("CPASLViT FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    # x = model(x, mask)
    print(x.shape)


if __name__ == "__main__":
    main()
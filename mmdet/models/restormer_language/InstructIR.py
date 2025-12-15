import torch
from torch import nn

import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class ICB(nn.Module):
    """
    Instruction Condition Block (ICB)
    Paper Section 3.3
    """

    def __init__(self, feature_dim, text_dim=768):
        super(ICB, self).__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.block = NAFBlock(feature_dim)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding): # (b,dim,h,w), (b, f_dim)
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.block(f)               # 3) block feature enhancement
        return f + x
    
class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        # Define a convolution to generate attention scores across spatial dimensions
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x has shape (B, C, H, W)
        B, C, H, W = x.size()

        # Apply 1x1 convolution to get attention map (B, 1, H, W)
        attention = self.attention_conv(x)
        attention = attention.view(B, 1, H * W)  # Reshape to (B, 1, H*W)
        attention = F.softmax(attention, dim=-1)  # Softmax across spatial dimensions
        attention = attention.view(B, 1, H, W)  # Reshape back to (B, 1, H, W)

        # Apply attention to the feature map by element-wise multiplication
        weighted_x = x * attention  # Shape (B, C, H, W)

        # Sum over spatial dimensions to get (B, C)
        out = weighted_x.view(B, C, -1).sum(dim=-1)

        return out

def masked_mean_pooling(embeddings, l_mask):
    # embeddings: (B, Feature_dim, Seq_len)
    # l_mask: (B, Seq_len, 1)

    # Expand mask to match embeddings shape
    l_mask_expanded = l_mask.transpose(1, 2)  # (B, 1, Seq_len)
    
    # Element-wise multiplication to mask out padding tokens
    masked_embeddings = embeddings * l_mask_expanded.float()  # (B, Feature_dim, Seq_len)
    
    # Sum embeddings along the Seq_len dimension
    sum_embeddings = torch.sum(masked_embeddings, dim=-1)  # (B, Feature_dim)
    
    # Count the number of valid tokens in each sequence (sum of the mask)
    valid_token_counts = torch.clamp(l_mask_expanded.sum(dim=-1), min=1e-9)  # (B, 1)
    # print("This is number of valid token", valid_token_counts)
    
    # Compute mean by dividing by valid token counts
    mean_embeddings = sum_embeddings / valid_token_counts  # (B, Feature_dim)
    normalized_embeddings = F.normalize(mean_embeddings, p=2, dim=1)  # L2 normalization

    return normalized_embeddings

def mean_pooling_text(embeddings):
    # embeddings: (B, Feature_dim, Seq_len)
    
    embeddings = torch.mean(embeddings, dim=-1)  # (B, Feature_dim)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization

    return normalized_embeddings
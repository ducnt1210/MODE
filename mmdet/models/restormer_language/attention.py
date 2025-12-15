import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch.nn.modules.utils import _pair as to_2tuple
from calflops import calculate_flops

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class VisionAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1, target_patches=300):
        super(VisionAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        self.target_patches = target_patches

        if out_channels is None:
            self.out_channels = self.value_channels

        self.reduce_conv = nn.Conv1d(self.v_in_channels, self.v_in_channels, kernel_size=3, stride=2, padding=1)
        # Reduce number of patches using pooling
        self.pool = nn.AdaptiveAvgPool1d(self.target_patches)  # Pooling to reduce the number of patches

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.key_channels),
            nn.GroupNorm(self.key_channels, self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.out_channels),
            nn.GroupNorm(self.out_channels, self.out_channels),
        )
    
    def forward(self, x, l, l_mask=None):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, T)
        # l_mask shape: (B, T, 1)
        
        B, HW = x.size(0), x.size(1)
            
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        x_pool = self.pool(self.reduce_conv(x)) # (B, key_channels, target_patches)

        query = self.project_q(x)  # (B, key_channels, H*W) 
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(x_pool)  # (B, key_channels, target_patches)
        value = self.project_v(x_pool)  # (B, value_channels, target_patches)

        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, target_patches)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, target_patches)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, target_patches)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, H*W, target_patches)
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, out_channels, HW) 
        out = out.permute(0, 2, 1)  # (B, HW, out_channels)

        return out, None

class VisionTextAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(VisionTextAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.key_channels),
            nn.GroupNorm(self.key_channels, self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.out_channels),
            nn.GroupNorm(self.out_channels, self.out_channels),
        )

    def forward(self, x, l, l_mask=None):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, T)
        # l_mask shape: (B, T, 1)
        
        B, HW = x.size(0), x.size(1)
        T = l.size(2)
        if l_mask is None:
            l_mask = torch.ones(B, T, 1).to(x.device if x.is_cuda else torch.device('cpu'))
            
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)

        query = self.project_q(x)  # (B, key_channels, H*W) 
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(l)  # (B, key_channels, T)
        value = self.project_v(l)  # (B, value_channels, T)

        key = key * l_mask  # (B, key_channels, T)
        value = value * l_mask  # (B, value_channels, T)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, T)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, T)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, T)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, T)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, T)
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, out_channels, HW) 
        out = out.permute(0, 2, 1)  # (B, HW, out_channels)

        return out
    
class VisionTextAttentionNoMask(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(VisionTextAttentionNoMask, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.key_channels),
            nn.GroupNorm(self.key_channels, self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.out_channels),
            nn.GroupNorm(self.out_channels, self.out_channels),
        )

    def forward(self, x, l):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, 1)
        
        B, HW = x.size(0), x.size(1)
        T = l.size(2)
            
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)

        query = self.project_q(x)  # (B, key_channels, H*W) 
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(l)  # (B, key_channels, T)
        value = self.project_v(l)  # (B, value_channels, T)

        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, T)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, T)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, T)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, T)
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, out_channels, HW) 
        out = out.permute(0, 2, 1)  # (B, HW, out_channels)

        return out
    
class TextVisionAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(TextVisionAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: visual features: (B, v_in_channels, H*W)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: text features: (B, C_l, T)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.GroupNorm(self.key_channels, self.key_channels),
        )

        # Values: visual features: (B, v_in_channels, H*W)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection for text output
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.GroupNorm(self.out_channels, self.out_channels),
        )

    def forward(self, l, x, l_mask=None):
        # l input shape: (B, C_l, T)
        # x shape: (B, H*W, C_v)
        # l_mask shape: (B, T, 1) - used to mask out padding in text

        B, T = l.size(0), l.size(2)
        T = l.size(2)
        if l_mask is None:
            l_mask = torch.ones(B, T, 1).to(x.device if x.is_cuda else torch.device('cpu'))

        x = x.permute(0, 2, 1)  # (B, C_v, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)

        query = self.project_q(l)  # (B, key_channels, T)
        query = query.permute(0, 2, 1)  # (B, T, key_channels)
        key = self.project_k(x)  # (B, key_channels, H*W)
        value = self.project_v(x)  # (B, value_channels, H*W)

        # Mask only applied to query (text) during attention score calculation
        query = query * l_mask.permute(0, 2, 1)  # Apply mask to query (text features)

        query = query.reshape(B, self.num_heads, T, self.key_channels // self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, -1)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, -1)

        # Attention score
        sim_map = torch.matmul(query, key)  # (B, num_heads, T, H*W)
        sim_map = (self.key_channels ** -0.5) * sim_map  # Scaled dot product

        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, T, H*W)

        # Apply attention weights to values
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, T, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, self.value_channels)  # (B, T, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, T)
        out = self.W(out)  # (B, out_channels, T)
        # out = out.permute(0, 2, 1)  # (B, T, out_channels)

        return out # (B, out_channels, T)
    
class TextVisionAttentionV2(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, dim, out_channels, num_heads=1, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.lang = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.query = nn.Linear(l_in_channels, dim, bias=qkv_bias)
        self.key = nn.Linear(v_in_channels, dim, bias=qkv_bias)
        self.value = nn.Linear(v_in_channels, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.vis_project = nn.Sequential(nn.Conv1d(l_in_channels, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                        )
        self.project_mm = nn.Sequential(nn.Conv1d(dim, out_channels, 1, 1),
                                        nn.GELU(),
                                        )

    def forward(self, vfeat, tfeat):
        # vfeat (B, C_v, H, W)
        # tfeat (B, C_t, L).
        vfeat = vfeat.flatten(2).permute(0, 2, 1)
        tfeat = tfeat.permute(0, 2, 1)
        # print("This is v_feat", vfeat.shape)
        # print("This is t_feat", tfeat.shape)
        B, Nv, C = vfeat.shape
        # print(tfeat.shape)
        # print(vfeat.shape)
        Nt = tfeat.size(1)
        tf = self.vis_project(tfeat.permute(0, 2, 1))
        q = self.query(tfeat).reshape(B, Nt, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # B, L, 1, dim
        k = self.key(vfeat).reshape(B, Nv, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # B, Nv, 1, dim
        v = self.value(vfeat).reshape(B, Nv, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # B, Nv, 1, dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nt, C)
        x = self.proj(x)

        mm = torch.mul(tf, x.permute(0, 2, 1))
        mm = self.project_mm(mm)  # (B, C_t, L)

        # print("This is text attention", mm.shape)
        # raise SystemExit

        return mm

class Mobile_Attention(nn.Module):
    # Mobile Attention with head competing mechanism
    def __init__(self, d_input, d_model, d_output, n_heads, dropout=0.05, eps=1e-6):
        super(Mobile_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Mobile-Attention competing on the head dimension
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / torch.sum(
            (queries + self.eps) * (keys.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        source_outgoing = 1.0 / torch.sum(
            (keys + self.eps) * (queries.sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1, 1) + self.eps), dim=-1)
        # (2) conservation refine for source and sink
        conserved_sink = torch.sum((queries + self.eps) * (
                (keys * source_outgoing[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                        1) + self.eps), dim=-1)
        conserved_source = torch.sum((keys + self.eps) * (
                (queries * sink_incoming[:, :, :, None]).sum(dim=1, keepdim=True).repeat(1, self.n_heads, 1,
                                                                                         1) + self.eps), dim=-1)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x
    
class TextTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, output_dim, dropout=0.1, seq_len=300, seq_reduce_factor=2, attn_mode="MHA"):
        """
        Initialize a simplified Transformer block for low computational cost.

        Args:
            dim (int): Dimension of input embeddings (C_l).
            n_heads (int): Number of attention heads.
            ff_dim (int): Hidden dimension for feedforward layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attn_mode = attn_mode
        
        # Self-attention module
        if attn_mode == "MHA":
            self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        else:
            self.attention = Mobile_Attention(d_input=dim, d_model=ff_dim, d_output=dim, n_heads=1, dropout=dropout)
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

        self.output_projection = None
        if self.dim != self.output_projection:
            self.output_projection = nn.Linear(dim, output_dim)

        self.pooling = None
        if seq_reduce_factor and seq_len:
            self.pooling = nn.AdaptiveAvgPool1d(output_size=seq_len // seq_reduce_factor)
        
        # Layer normalization
        # self.attention_norm = RMSNorm(dim)
        self.attention_norm = nn.LayerNorm(dim)
        # self.ffn_norm = RMSNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer block.
        
        Args:
            x (torch.Tensor): Input text embeddings of shape (B, C_t, L).
            mask (torch.Tensor, optional): Mask of shape (B, T) for padding positions. Defaults to None.
        
        Returns:
            torch.Tensor: Output embeddings of shape (B, C_t, L).
        """
        x = x.permute(2, 0, 1)
        # print("This is transformer in:", x.shape)
        # Apply layer normalization before attention
        x_norm = self.attention_norm(x)
        
        # Self-attention with masking
        if self.attn_mode == "MHA":
            if mask is not None:
                mask = ~mask.bool() # Convert to boolean mask with True for padding
                # print("This is mask", mask)
                attn_output, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=mask)
            else:
                attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        else:
            attn_output = self.attention(x_norm, x_norm, x_norm)
        
        # Residual connection and normalization
        x = x + attn_output
        
        # Feed-forward network
        x_ffn_norm = self.ffn_norm(x)
        x_ffn = self.feed_forward(x_ffn_norm)
        
        # Final residual connection
        output = x + x_ffn

        # Project to output dimension
        if self.output_projection:
            output = self.output_projection(output)

        if self.pooling:
            output = self.pooling(output.permute(1, 2, 0)).permute(2, 0, 1)
        
        output = output.permute(1, 2, 0)
        # print("This is text transformer", output.shape)
        
        return output

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

def main():
    input_dim = 256
    model = TextVisionAttention(input_dim, 256, input_dim, input_dim, num_heads=1)
    # model = VisionTextAttention(input_dim, 768, input_dim, input_dim, num_heads=1)
    x = torch.rand(1, 320*640, input_dim)
    l = torch.rand(1, 256, 25)
    l_mask = torch.cat([torch.ones(1, 15, 1), torch.zeros(1, 10, 1)], dim=1)

    inputs = {}
    inputs['l'] = l
    inputs['x'] = x
    inputs['l_mask'] = l_mask

    # q = torch.randn([1, 320*640, 768])
    # k = torch.randn([1, 300, 768])
    # v = torch.randn([1, 300, 768])
    # mobile_attn = Mobile_Attention(768, 192, 192, 1)
    # inputs = {}
    # inputs['queries'] = q
    # inputs['keys'] = k
    # inputs['values'] = v

    flops, macs, params = calculate_flops(model=model,
                                      kwargs = inputs,
                                      print_results=True, print_detailed=False)
    print("CPASLViT FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    # x = mobile_attn(q, k, v)
    # print(x.shape)


if __name__ == "__main__":
    main()
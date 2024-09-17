# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/8/28 19:50
@desc: 
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class ViT_Backbone(nn.Module):
    def __init__(self):
        super(ViT_Backbone, self).__init__()
        self.conv = nn.Conv2d(3, 768, kernel_size=16, stride=16, padding=0)
        self.embedding_dim = 768
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.embedding_dim)
        class_token = torch.arange(self.embedding_dim, dtype=torch.int32).view(1, -1)
        x = torch.concatenate([x, class_token], dim=0)
        position_token = self.position(x.shape[0])
        x = x + position_token
        return x
    def position(self, max_len):
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_pos = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(10000) / self.embedding_dim))
        encoding = torch.zeros((max_len, self.embedding_dim))
        encoding[:, 0::2] = torch.sin(pos * div_pos)
        encoding[:, 1::2] = torch.cos(pos * div_pos)
        return encoding

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.layer = nn.LayerNorm(4 * self.dim)
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "wrong size"

        x = x.view(B, H, W, C)
        padding = (H % 2) or (W % 2)
        if padding:
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4*C)
        x = self.layer(x)
        x = self.reduction(x)
        return x

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Attention, self).__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = embedding_dim // num_heads
        self.to_qkv = nn.Linear(self.dim, self.dim*3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.dim ** -0.5

        self.layer = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout()
    def forward(self, x):
        x = self.layer(x)
        B, L, C = x.shape
        x = rearrange(x, "b n (head dim) -> b head n dim", head=self.num_heads, dim=self.dim)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, head, L, dim]
        atten = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)   # [B, head, L, L]
        atten = torch.matmul(atten, v)   # [B, head, L, dim]

        out = self.drop(atten)
        out = rearrange(out, "b head n dim -> b n (head dim)")
        return out

x = torch.randn(size=(1, 10, 16))
attention = Attention(16, 8)
out = attention(x)
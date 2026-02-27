import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=16, base_channels=64):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32
        self.enc_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.Sequential(nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1), ResBlock(base_channels), ResBlock(base_channels))
        self.down2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), ResBlock(base_channels * 2), ResBlock(base_channels * 2))
        self.down3 = nn.Sequential(nn.Conv2d(base_channels * 2, latent_channels, 4, stride=2, padding=1), ResBlock(latent_channels), ResBlock(latent_channels))
        
        # Decoder: 32 -> 64 -> 128 -> 256
        self.up1 = nn.Sequential(nn.Conv2d(latent_channels, base_channels * 2 * 4, 3, padding=1), nn.PixelShuffle(2), ResBlock(base_channels * 2), ResBlock(base_channels * 2))
        self.up2 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.PixelShuffle(2), ResBlock(base_channels), ResBlock(base_channels))
        self.up3 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 4, 3, padding=1), nn.PixelShuffle(2), ResBlock(base_channels), ResBlock(base_channels))
        
        self.dec_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def encode(self, x):
        h = self.enc_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return h

    def decode(self, z):
        h = self.up1(z)
        h = self.up2(h)
        h = self.up3(h)
        return self.dec_out(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        # Self attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross attention
        x = x + self.cross_attn(self.norm2(x), context)
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x

class FlowDenoiser(nn.Module):
    def __init__(self, latent_channels=16, context_dim=768, dim=256, depth=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = 1 # Latent is already small (16x16)
        
        self.proj_in = nn.Conv2d(latent_channels, dim, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, context_dim) for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(dim, latent_channels, 1)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 32*32, dim))

    def forward(self, x, t, context):
        # x: [b, 16, 16, 16]
        # t: [b]
        # context: [b, seq_len, context_dim]
        
        b, c, h, w = x.shape
        x = self.proj_in(x) # [b, dim, h, w]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1) # [b, 1, dim]
        
        # Inject time embedding into x
        x = x + t_emb
        
        for block in self.transformer_blocks:
            x = block(x, context)
            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x)

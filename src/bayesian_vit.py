import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit import FeedForward

def model_kl_loss(model: nn.Module):
    kl = 0.0
    for m in model.modules():
        if isinstance(m, BayesianLinear):
            if hasattr(m, '_last_kl'):
                kl = kl + m._last_kl
            else:
                # compute analytic kl even if not sampled yet
                kl = kl + m.kl_divergence()
    return kl

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class BayesianLinear(Module):
    self.init__(self, in_features, out_features, rank=4, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
    


class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_rank=4, prior_var=1.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Replace linear with BayesianLinear for Q/K/V
        self.to_qkv = BayesianLinear(in_features=dim, out_features=inner_dim * 3, rank=qkv_rank, prior_var=prior_var)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, sample=True):
        x = self.norm(x)

        # to_qkv applies linear with sampled W; returns shape (b, n, inner_dim*3)
        qkv = self.to_qkv(x, sample=sample).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qkv_rank=4, prior_var=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qkv_rank=qkv_rank, prior_var=prior_var),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, sample=True):
        for attn, ff in self.layers:
            x = attn(x, sample=sample) + x
            x = ff(x) + x

        return self.norm(x)

class BViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., qkv_rank=4, prior_var=1.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qkv_rank=qkv_rank, prior_var=prior_var)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, sample=True):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x, sample=sample)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
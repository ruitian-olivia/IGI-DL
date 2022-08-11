import torch
import torch.nn as nn
from vit_pytorch.vit import PreNorm, FeedForward, Attention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from MLP_model import MLP

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT_MLP_gene(nn.Module):
    def __init__(
        self, 
        attn_heads: Optional[int] = 8, 
        dim_head: Optional[int] = 64, 
        emb_dropout: Optional[float] = 0.,
        hidden_features: Optional[int] = 256, 
        gene_dim: Optional[int] = 215,
        mlp_hidden_list: Optional[List[int]] = [256, 256],
        vit_pool: Optional[str] ='mean', 
        vit_dropout: Optional[float] = 0., 
        input_image_channels: Optional[int] = 3, 
        patch_size: Optional[Tuple[int]] = (10,10), 
        image_size: Optional[Tuple[int]] = (200,200), 
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = input_image_channels * patch_height * patch_width
        assert vit_pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, hidden_features),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_features))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn1 = PreNorm(hidden_features, Attention(hidden_features, heads = attn_heads, dim_head = dim_head, dropout = vit_dropout))
        self.fft1 = PreNorm(hidden_features, FeedForward(hidden_features, hidden_features, dropout = vit_dropout))

        self.attn2 = PreNorm(hidden_features, Attention(hidden_features, heads = attn_heads, dim_head = dim_head, dropout = vit_dropout))
        self.fft2 = PreNorm(hidden_features, FeedForward(hidden_features, hidden_features, dropout = vit_dropout))
        
        self.attn3 = PreNorm(hidden_features, Attention(hidden_features, heads = attn_heads, dim_head = dim_head, dropout = vit_dropout))
        self.fft3 = PreNorm(hidden_features, FeedForward(hidden_features, hidden_features, dropout = vit_dropout))
        
        self.attn4 = PreNorm(hidden_features, Attention(hidden_features, heads = attn_heads, dim_head = dim_head, dropout = vit_dropout))
        self.fft4 = PreNorm(hidden_features, FeedForward(hidden_features, hidden_features, dropout = vit_dropout))
        
        self.vit_pool = vit_pool
        # self.to_latent = nn.Identity()
        
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_features),
        #     nn.Linear(hidden_features, out_features)
        # )

        mlp_block = []
        in_features = hidden_features
        for per_hidden_features in mlp_hidden_list:
            mlp_block.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    dropout_prob=0.5,
                )
            )
            in_features = per_hidden_features

        self.mlp = nn.Sequential(*mlp_block)
        self.head = nn.Linear(in_features, gene_dim)

    def forward(self, img):
              
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.attn1(x) + x 
        x = self.fft1(x) + x 
                
        x = self.attn2(x) + x 
        x = self.fft2(x) + x 
        
        x = self.attn3(x) + x 
        x = self.fft3(x) + x 
        
        x = self.attn4(x) + x 
        x = self.fft4(x) + x 
        
        x = x.mean(dim = 1) if self.vit_pool == 'mean' else x[:, 0]

        x = self.mlp(x)
        x = self.head(x)

        return x


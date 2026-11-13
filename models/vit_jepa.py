import copy

import torch
from torch import nn
import torch.nn.functional as F


def _sincos_1d(dim, positions):
    if dim % 2:
        raise ValueError("Each positional-embedding axis must have even dimension")
    omega = torch.arange(dim // 2, dtype=torch.float64)
    omega = 1.0 / (10000 ** (omega / (dim / 2)))
    phase = positions.reshape(-1, 1).double() * omega.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=1).float()


def sincos_3d(embed_dim, temporal, height, width):
    """Official V-JEPA dimension split: 1/2 temporal, 1/4 per spatial axis."""
    if embed_dim % 4:
        raise ValueError("embed_dim must be divisible by 4")
    tt, hh, ww = torch.meshgrid(
        torch.arange(temporal), torch.arange(height), torch.arange(width), indexing="ij"
    )
    return torch.cat(
        (_sincos_1d(embed_dim // 2, tt),
         _sincos_1d(embed_dim // 4, hh),
         _sincos_1d(embed_dim // 4, ww)), dim=1
    )


def gather_tokens(tokens, indices):
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class PatchEmbed3D(nn.Module):
    """Official non-overlapping Conv3d patch projection."""
    def __init__(self, in_channels, embed_dim, tubelet_size, patch_size):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        q = self.norm1(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, depth, heads, frames, grid,
                 patch_size, tubelet_size=1, dropout=0.0):
        super().__init__()
        if frames % tubelet_size or grid % patch_size:
            raise ValueError("frames/grid must be divisible by tubelet_size/patch_size")
        temporal = frames // tubelet_size
        spatial = grid // patch_size
        self.num_patches = temporal * spatial * spatial
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, tubelet_size, patch_size)
        self.register_buffer(
            "pos_embed", sincos_3d(embed_dim, temporal, spatial, spatial).unsqueeze(0),
            persistent=True,
        )
        self.blocks = nn.ModuleList([Block(embed_dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, indices=None):
        x = self.patch_embed(x) + self.pos_embed.to(dtype=x.dtype)
        if indices is not None:
            x = gather_tokens(x, indices)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionTransformerPredictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, depth, heads, pos_embed):
        super().__init__()
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.register_buffer("pos_embed", pos_embed.clone().float(), persistent=True)
        self.pos_proj = nn.Linear(encoder_dim, predictor_dim)
        self.blocks = nn.ModuleList([Block(predictor_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context, context_indices, target_indices):
        positions = self.pos_embed.expand(context.shape[0], -1, -1)
        visible = self.input_proj(context) + self.pos_proj(gather_tokens(positions, context_indices))
        masked = self.mask_token.expand(context.shape[0], target_indices.shape[1], -1)
        masked = masked + self.pos_proj(gather_tokens(positions, target_indices))
        x = torch.cat((visible, masked), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x[:, visible.shape[1]:]))


class StudentTeacherJEPA(nn.Module):
    def __init__(self, in_channels, embed_dim, enc_depth, pred_depth, heads,
                 patch_size, frames, grid, tubelet_size=1, predictor_dim=None):
        super().__init__()
        self.student = VisionTransformer(
            in_channels, embed_dim, enc_depth, heads, frames, grid, patch_size, tubelet_size
        )
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.predictor = VisionTransformerPredictor(
            embed_dim, predictor_dim or embed_dim, pred_depth, heads, self.student.pos_embed
        )

    def forward(self, x, context_indices, target_indices):
        context = self.student(x, context_indices)
        prediction = self.predictor(context, context_indices, target_indices)
        with torch.no_grad():
            target = F.layer_norm(self.teacher(x), (prediction.shape[-1],))
            target = gather_tokens(target, target_indices)
        return prediction, target, self.student(x)

    @torch.no_grad()
    def update_teacher(self, momentum):
        for student, teacher in zip(self.student.parameters(), self.teacher.parameters()):
            teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)


class FutureTransformer(nn.Module):
    """Predict future traffic patches from context tokens without a CNN decoder."""
    def __init__(self, dim, depth, heads, horizon, spatial_patches, out_channels, patch_size):
        super().__init__()
        self.horizon = horizon
        self.spatial_patches = spatial_patches
        self.patch_size = patch_size
        self.out_channels = out_channels
        count = horizon * spatial_patches * spatial_patches
        self.future_queries = nn.Parameter(torch.zeros(1, count, dim))
        self.future_pos = nn.Parameter(torch.zeros(1, count, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.reconstruct = nn.Linear(dim, out_channels * patch_size * patch_size)
        nn.init.trunc_normal_(self.future_queries, std=0.02)
        nn.init.trunc_normal_(self.future_pos, std=0.02)

    def forward(self, context_tokens):
        batch = context_tokens.shape[0]
        queries = self.future_queries.expand(batch, -1, -1) + self.future_pos
        x = torch.cat((context_tokens, queries), dim=1)
        for block in self.blocks:
            x = block(x)
        patches = self.reconstruct(self.norm(x[:, context_tokens.shape[1]:]))
        p, s = self.patch_size, self.spatial_patches
        patches = patches.reshape(batch, self.horizon, s, s, self.out_channels, p, p)
        return patches.permute(0, 4, 1, 2, 5, 3, 6).reshape(
            batch, self.out_channels, self.horizon, s * p, s * p
        )

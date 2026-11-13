import torch
from torch import nn
import torch.nn.functional as F

from .vit_jepa import FutureTransformer, StudentTeacherJEPA


class TaxiVJEPA(nn.Module):
    def __init__(self, in_channels=2, embed_dim=192, enc_depth=6, pred_depth=4,
                 heads=3, patch=4, frames=12, horizon=3, grid=32,
                 hidden=192, tubelet_size=1, flow_depth=4):
        super().__init__()
        self.temporal_tokens = frames // tubelet_size
        self.spatial_tokens = (grid // patch) ** 2
        self.jepa = StudentTeacherJEPA(
            in_channels, embed_dim, enc_depth, pred_depth, heads, patch,
            frames, grid, tubelet_size, predictor_dim=hidden,
        )
        self.flow_predictor = FutureTransformer(
            embed_dim, flow_depth, heads, horizon, grid // patch, in_channels, patch
        )

    def make_masks(self, batch, device, mask_ratio=0.5):
        """Official random-tube mask: one spatial mask repeated over all time tokens."""
        target_spatial = max(1, min(self.spatial_tokens - 1, round(self.spatial_tokens * mask_ratio)))
        context_masks, target_masks = [], []
        offsets = torch.arange(self.temporal_tokens, device=device)[:, None] * self.spatial_tokens
        for _ in range(batch):
            permutation = torch.randperm(self.spatial_tokens, device=device)
            target = (offsets + permutation[:target_spatial]).flatten()
            context = (offsets + permutation[target_spatial:]).flatten()
            context_masks.append(context)
            target_masks.append(target)
        return torch.stack(context_masks), torch.stack(target_masks)

    def forward(self, context, future=None, mask_ratio=0.5, flow_weight=1.0,
                flow_loss="l1"):
        context_idx, target_idx = self.make_masks(context.shape[0], context.device, mask_ratio)
        prediction, target, full_tokens = self.jepa(context, context_idx, target_idx)
        jepa_loss = F.l1_loss(prediction, target)
        flow = self.flow_predictor(full_tokens)
        losses = {"jepa": jepa_loss}
        if future is not None:
            if flow_loss == "l1":
                supervised = F.l1_loss(flow, future)
            elif flow_loss == "mse":
                supervised = F.mse_loss(flow, future)
            else:
                raise ValueError(f"Unsupported flow loss: {flow_loss}")
            losses["flow"] = supervised
            losses["total"] = jepa_loss + flow_weight * supervised
        else:
            losses["total"] = jepa_loss
        return {"flow": flow, "pred_tokens": prediction, "target_tokens": target,
                "enc_idx": context_idx, "pred_idx": target_idx, "losses": losses}

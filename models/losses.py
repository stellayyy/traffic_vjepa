import torch
import torch.nn.functional as F


def flatten_tokens(x):
    return x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])


def vc_loss(z, projector, std_coeff=10.0, cov_coeff=100.0):
    r = projector(flatten_tokens(z))
    r = r - r.mean(0, keepdim=True)
    std = torch.sqrt(r.var(0) + 1e-4)
    std_loss = F.relu(1.0 - std).mean()
    cov = r.T @ r / max(r.shape[0] - 1, 1)
    off = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[0] + 1)[:, 1:]
    cov_loss = off.pow(2).mean()
    return std_coeff * std_loss + cov_coeff * cov_loss, {"std": float(std_loss.detach()), "cov": float(cov_loss.detach())}


def latent_prediction_loss(target, prediction, projector):
    return F.mse_loss(projector(flatten_tokens(target)), projector(flatten_tokens(prediction)))

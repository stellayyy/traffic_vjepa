from torch import nn


class Projector(nn.Module):
    def __init__(self, latent: int, projection: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, projection), nn.LayerNorm(projection), nn.GELU(),
            nn.Linear(projection, projection, bias=False),
        )
        self.out_dim = projection

    def forward(self, x):
        return self.net(x)

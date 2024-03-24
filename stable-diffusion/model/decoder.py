import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        residue = x

        batch_size, channel, height, width = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.htm
        x = x.view(batch_size, channel, height * width)

        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        # https://pytorch.org/docs/stable/generated/torch.transpose.html#torch.transpose
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, features, height * width) -> (batch_size, features, height, width)
        x = x.view(batch_size, channel, height, width)

        x *= residue


# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        # The channels are always 32 in stable diffusion
        self.group_norm_1 =  nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 =  nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)

        residue = x

        x = self.group_norm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.group_norm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 4, width / 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 4, width / 4) -> (batch_size, 512, height / 2, width / 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height / 2, width / 2) -> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, height, width)
        return x
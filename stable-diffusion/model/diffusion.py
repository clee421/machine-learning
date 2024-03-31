import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, num_embeds: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(num_embeds, 4 * num_embeds)
        self.linear_2 = nn.Linear(4 * num_embeds, 4 * num_embeds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 1280)
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, n_time = 1280) -> None:
        super().__init__()

        self.group_norm_feature = nn.GroupNorm(32, input_channels)
        self.conv_feature = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, output_channels)

        self.group_norm_merged = nn.GroupNorm(32, output_channels)
        self.conv_merged = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature: (batch_size, input_channels, height, width)
        # time: (1, 1280)

        residue = feature

        feature = self.group_norm_feature(feature)

        # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.group_norm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, num_heads: int, num_embeds: int, d_context = 768) -> None:
        super().__init__()

        channels = num_heads * num_embeds

        # https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        # num_groups (int) – number of groups to separate the channels into
        # num_channels (int) – number of channels expected in input
        # eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
        # affine (bool) – a boolean value that when set to True, this module has learnable
        #   per-channel affine parameters initialized to ones (for weights) and zeros
        #   (for biases). Default: True.
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(num_heads, channels, input_projection_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(num_heads, channels, d_context, input_projection_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channels)

        # GLU Variants Improve Transformer
        # https://arxiv.org/abs/2002.05202v1
        #
        # There looks to be requests to add geglu to pytorch
        # https://github.com/pytorch/pytorch/issues/80168
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context):
        # x: (batch_size, features, height, width)
        # context: (batch_size, sequence_length, dim)

        residue_long = x

        x = self.group_norm(x)

        x = self.conv_input(x)

        batch_size, num_features, height, width = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(batch_size, num_features, height * width)

        # (batch_size, height * width, features)
        x = x.transpose(-1, -2)

        ## Normalization + Self Attention w/ skip connection
        residue_short = x
        x = self.layer_norm_1(x)
        self.attention_1(x)
        x += residue_short

        ## Normalization + Cross Attention w/ skip connection
        residue_short = x
        x = self.layer_norm_2(x)
        # cross attention
        self.attention_2(x, context)
        x += residue_short

        ## Normalization + Feed Forward Layer w/ GeGLU and skip connection
        residue_short = x

        x = self.layer_norm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        x = x.transpose(-1, -2)

        x = x.view(batch_size, num_features, height, width)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch_size, features, height, width) -> (batch_size, features, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (batch_size, 4, height / 8, height / 8) -> (batch_size, 320, height / 8, height / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (batch_size, 320, height / 8, height / 8) -> (batch_size, 320, height / 16, height / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, height / 16, height / 16) -> (batch_size, 640, height / 32, height / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height / 32, height / 32) -> (batch_size, 1280, height / 64, height / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (batch_size, 2560, height / 64, width / 64) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160),  Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80),  Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of
            # features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, input_channels)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch_size, 320, height / 8, width / 8)
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x

class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        """
        latent: the encoded image
        context: the prompt to provide context to the denoise
        time: provides the step in the denoise process
        """
        # latent: (batch_size, 4, height/8, width/8)
        #   The output of the encoder is (batch_size, 4, height/8, width/8), thus why it's 4
        # context: (batch_size, sequence_length, dimension)
        # time: (1, 320)
        #   The time is calculated similarly to positional encoding

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        output = self.final(output)

        return output


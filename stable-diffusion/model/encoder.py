import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

"""
Variational Autoencoder (VAE): A type of neural network architecture designed
for generative tasks. It learns to encode input data into a lower-dimensional
representation and then decode this representation back into the original data
space. The "variational" aspect comes from the way it handles the encoding
process, introducing a probabilistic approach to generate diverse outputs from
similar inputs.
"""
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        Notes for nn.Conv2d

        in_channels (int): Specifies the number of channels in the input image. For example,
        for a grayscale image, this would be 1. For an RGB image, it would be 3.

        out_channels (int): Determines the number of filters (kernels) to apply to the input
        image. This also corresponds to the number of feature maps (or output channels)
        produced by the convolution.

        kernel_size (int or tuple): Defines the size of the filter (kernel) to be used for
        convolution. If it's an integer n, it implies a (n x n) square filter. If it's a
        tuple (n, m), it specifies the filter size as (height, width).

        stride (int or tuple, optional): Controls the stride for the convolution, i.e., how
        many pixels the filter moves across the input image each time. A stride of 1 means
        moving the filter one pixel at a time. If it's an integer n, it implies a stride of
        (n, n). If it's a tuple (n, m), it specifies the stride as (vertical stride,
        horizontal stride). The default value is 1.

        padding (int or tuple, optional): Used to add padding to the input image. Padding
        adds additional pixels around the input image. This parameter controls the amount
        of padding. If it's an integer n, it adds n pixels of padding on all sides. If it's
        a tuple (n, m), it adds padding of (n pixels on top and bottom, m pixels on left
        and right). The default value is 0 (no padding).

        dilation (int or tuple, optional): Controls the spacing between the kernel elements.
        It can be used to control the receptive field and the output size. A dilation of 1
        means there is no spacing, i.e., the standard convolution. If it's an integer n, it
        implies a dilation of (n, n). If it's a tuple (n, m), it specifies the dilation as
        (vertical dilation, horizontal dilation). The default value is 1.

        groups (int, optional): Controls the connections between inputs and outputs.
        groups=1 denotes a standard convolution where each input is convolved with every
        output filter. groups=in_channels denotes a depthwise convolution, i.e., each input
        channel is convolved with its own set of filters (of size out_channels / in_channels).
        The default value is 1.

        bias (bool, optional): If True, adds a learnable bias to the output. This is
        typically set to True (the default) unless you are stacking several convolutional
        layers together and using batch normalization.

        For a visualization of a convultion you can check out this link
        https://ezyang.github.io/convolution-visualizer/
        """
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            # We're reducing the size of the image (px area)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            # Although the image size is reduced we're increaing the feature count
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            # We're reducing the size of the image (px area)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            # https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
            nn.GroupNorm(32, 512),

            # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
            nn.SiLU(),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            """
            So why is the model built like this?
            A: Usually in deep learning communities, especially during research, we do not try to
            reinvent the wheel. Whoever created the stable diffusion model, checks what kind of
            similar model already exists that are working well. It's probable that whoever wrote
            the stable diffusion model saw an architecture like above that has been working well
            as a variational encoder and made small modifications.
            """
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, output_channels, height/8, width/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # Pad: (padding_left, padding_right, padding_top, padding_bottom)
                # Pad with zeros on the right and bottom.
                # (batch_size, channel, height, width) ->
                #   (batch_size, channel, height + padding_top + padding_bottom, width + padding_left + padding_right) = (batch_size, channel, height + 1, width + 1)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # https://pytorch.org/docs/stable/generated/torch.chunk.html
        # Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
        # (batch_size, 8, height/8, width/8)
        # -> two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        """
        The tensor x is assumed to have a structure where the first half along the specified dimension
        (dim=1, which corresponds to the channel dimension in a Convolutional Neural Network) is meant
        to represent the mean (mean) of some latent variables, and the second half is meant to represent
        the log variance (log_variance) of those latent variables.

        This approach is efficient and concise for a couple of reasons:

        Compact representation: It allows the model to output both necessary components for the
        reparameterization trick (used in VAEs) in a single forward pass. This is particularly useful
        in neural network architectures where the final layer outputs both parameters directly.

        Simplicity: Using torch.chunk simplifies the code, avoiding the need for manual slicing operations.
        This can make the code easier to read and maintain, especially in complex models.
        The underlying assumption is that the output tensor x has been designed and trained to produce
        values such that when split in half, the first part can be effectively used as the mean of a
        distribution, and the second part can be treated as the log variance. This design is intrinsic
        to how the network is structured and trained, specifically in the context of VAEs or similar
        models where the output is meant to parameterize a probability distribution in the latent space.

        The choice of representing variance in the log space (and hence using log_variance) is due to
        numerical stability and efficiency reasons. Operating in log space helps avoid issues with
        floating-point underflow and overflow when dealing with very small or large variances. It also
        simplifies operations like exponentiation (to get the actual variance) and ensures that variance
        values are positive, as the exponentiation of any real number is positive.
        """

        # https://pytorch.org/docs/stable/generated/torch.clamp.html
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        # https://pytorch.org/docs/stable/generated/torch.exp.html
        # Returns a new tensor with the exponential of the elements of the input tensor input.
        # yi = e^xi
        variance = log_variance.exp()

        # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        std_dev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev)
        # How do we sample from the latent space, how do we sample from the gaussian distribution?
        # If you have a sample N(0, 1) we can sample any other part of the gaussion if we have the
        # mean and the variance.
        # Z = N(0, 1) -> N(mean, variance)?
        x = mean + std_dev * noise

        # Scale by a constant
        # Constant taken from:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215

        return x

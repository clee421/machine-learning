import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    # AKA negative prompt
    unconditional_prompt=None,
    input_image=None,
    # strength: how much attention do was want to give the input_image
    strength=0.8,
    # enable_cfg: enabled classifier_free_guidance, enhance the quality and relevance of generated images
    enable_cfg=True,
    # output = weight * (conditioned_output - unconditioned_output) + unconditioned_output
    # The conditioned_output here is the conditioning signal which is the prompt. So the
    # cfg_scale is the weight here for how much we care about the prompt
    # cfg_scale: how much do we want to pay attention to the prompt (value: 1-14)
    cfg_scale=7.5,
    sampler_type="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    # https://pytorch.org/docs/stable/generated/torch.no_grad.html
    """
    Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure that you will not
    call Tensor.backward(). It will reduce memory consumption for computations that would otherwise
    have requires_grad=True.
    """
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # I don't like this style. It's ok to be more verbose here and
        # have additional conditionals below
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # https://pytorch.org/docs/stable/generated/torch.Generator.html
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # pretrained model
        clip = models["clip"]
        clip.to(device)

        if enable_cfg:
            # Convert the prompt into tokens using tokenizer

            # Convert into a list of length sequence_length=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # (batch_size, sequence_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (batch_size, sequence_length) -> (batch_size, sequence_length, dimension)
            cond_context = clip(cond_tokens)

            # Convert into a list of length sequence_length=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [unconditional_prompt], padding="max_length", max_length=77
            ).input_ids

            # (batch_size, sequence_length)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (batch_size, sequence_length) -> (batch_size, sequence_length, dimension)
            uncond_context = clip(uncond_tokens)

            # (batch_size, sequence_length, dimension) + (batch_size, sequence_length, dimension) -> (2 * batch_size, sequence_length, dimension)
            # (2 * batch_size, sequence_length, dimension) = (2, 77, 768)
            # https://pytorch.org/docs/stable/generated/torch.cat.html
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length sequence_length=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # (batch_size, sequence_length)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (batch_size, sequence_length) -> (batch_size, sequence_length, dimension)
            # (batch_size, sequence_length, dimension) = (1, 77, 768)
            context = clip(tokens)

        to_idle(clip)

        if sampler_type == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))

            # (height, width, channel)
            input_image_tensor = np.array(input_image_tensor)

            # (height, width, channel) -> (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            # (height, width, channel) -> (height, width, channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (height, width, channel) -> (batch_size, height, width, channel)
            # https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            # https://pytorch.org/docs/stable/generated/torch.permute.html
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (batch_size, 4, latents_height, latents_width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # (batch_size, 4, latents_height, latents_width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (batch_size, 4, latents_height, latents_width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (batch_size, 4, latents_height, latents_width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents

            if enable_cfg:
                # (batch_size, 4, latents_height, latents_width) -> (2 * batch_size, 4, latents_height, latents_width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (batch_size, 4, latents_height, latents_width) -> (batch_size, 4, latents_height, latents_width)
            model_output = diffusion(model_input, context, time_embedding)

            if enable_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (batch_size, 4, latents_height, latents_width) -> (batch_size, 4, latents_height, latents_width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (batch_size, 4, latents_height, latents_width) -> (batch_size, 3, height, width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
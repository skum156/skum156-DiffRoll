import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import math
# Assuming RollDiffusion and SpecRollDiffusion are in task/diffusion.py
from task.diffusion import RollDiffusion, SpecRollDiffusion
import torchaudio
import logging # Import logging

# Set up a logger for this module
log = logging.getLogger(__name__)

EPSILON = 1e-6

def exists(x):
    return x is not None

def align_timesteps(x1, x2):
    """
    x1 and x2 must be of the shape (B, C, T, F)
    Then this function will discard the extra timestep
    in either x1 or x2 such that they match in T dimension.
    """
    if not exists(x1):
        log.error("align_timesteps received x1 as None")
        raise ValueError("x1 cannot be None in align_timesteps")
    if not exists(x2):
        log.error("align_timesteps received x2 as None")
        raise ValueError("x2 cannot be None in align_timesteps")

    T1 = x1.shape[2]
    T2 = x2.shape[2]

    Tmin = min(T1, T2)
    x1 = x1[:, :, :Tmin, :]
    x2 = x2[:, :, :Tmin, :]

    return x1, x2

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # taking exp
        embeddings = time[:, None] * embeddings[None, :] # boardcasting (B, dim//2)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # apply sin and cos (B, dim)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    """[https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class Unet(RollDiffusion):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )



    def forward(self, x, time):
        # x = (B, 1, dim, dim)
        x = self.init_conv(x) # (B, 18, dim, dim)
        t = self.time_mlp(time) if exists(self.time_mlp) else None # (B, dim*4)
        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x) # Save x *before* downsample
            x = downsample(x) # Then apply downsample

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = upsample(x) # Upsample first
            x_skip = h.pop() # Then pop the corresponding skip connection
            x = torch.cat((x, x_skip), dim=1) # Concatenate
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

        x = self.final_conv(x)

        return x


class SpecConvNextBlock(nn.Module):
    """[https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.spec_ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.spec_net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, spec, time_emb=None):
        h = self.ds_conv(x)
        spec_h = self.spec_ds_conv(spec)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            # Add condition to both branches, if desired for spec_h as well
            h = h + rearrange(condition, "b c -> b c 1 1")
            spec_h = spec_h + rearrange(condition, "b c -> b c 1 1") # Apply condition to spec_h as well

        h = self.net(h)
        spec_h = self.spec_net(spec_h)
        return h + self.res_conv(x), spec_h


class SpecUnet(SpecRollDiffusion):
    # Unet conditioned on spectrogram
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3, # Expected channels of the UNET input `x` (e.g., 3 for RGB-like).
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        spec_args={},
        inference_frequency_dim=16, # Added parameter for the target frequency dimension
        **kwargs, # This captures all extra arguments from checkpoint hparams
    ):
        # --- FIXES for 'Ignoring from checkpoint hparams' WARNINGS ---
        # Discarding irrelevant hparams from kwargs to prevent warnings from SpecRollDiffusion's __init__
        known_irrelevant_hparams = [
            'residual_channels', 'unconditional', 'condition', 'n_mels',
            'norm_args', 'residual_layers', 'kernel_size', 'dilation_base',
            'dilation_bound', 'spec_dropout'
        ]
        for hparam_name in known_irrelevant_hparams:
            if hparam_name in kwargs:
                value = kwargs.pop(hparam_name)
                log.warning(
                    f"Ignoring '{hparam_name}={value}' from checkpoint hparams "
                    "as it's not expected by SpecRollDiffusion's __init__. "
                    "Check base class signature or consider if this hparam is still relevant."
                )

        super().__init__(**kwargs)

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        # Initial layers for spectrograms
        self.spec_init_conv = nn.Conv2d(1, init_dim, 7, padding=3) # Takes 1 channel (Mel spec after unsqueeze)

        if 'n_mels' not in spec_args:
            raise ValueError("spec_args must contain 'n_mels'")
        self.spec_init_fc = nn.Linear(spec_args['n_mels'], inference_frequency_dim) # Output to target freq dim

        self.mel_layer = torchaudio.transforms.MelSpectrogram(**spec_args)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # dims: [init_dim, dim*1, dim*2, dim*4, dim*8] for dim_mults=(1,2,4,8)
        
        in_out = list(zip(dims[:-1], dims[1:]))
        # in_out: [(init_dim, dim*1), (dim*1, dim*2), (dim*2, dim*4), (dim*4, dim*8)]

        if use_convnext:
            block_klass = partial(SpecConvNextBlock, mult=convnext_mult)
            # SpecConvNextBlockUp is currently identical to SpecConvNextBlock.
            # If there's no functional difference, it's better to use one name.
            # Keeping it as `up_block_klass` for clarity if future changes differentiate them.
            up_block_klass = partial(SpecConvNextBlock, mult=convnext_mult) 
        else:
            raise NotImplementedError("ResnetBlock for SpecUnet is not implemented with spec conditioning.")

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_downsampling_stages = len(dim_mults) # e.g., 4 for (1,2,4,8)

        # Downsampling Path (Encoder)
        # All `downs` blocks should perform a spatial downsample for symmetry.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))), # Attn only on 'x'
                        Downsample(dim_out), # ALWAYS apply downsample for each stage in encoder
                        Downsample(dim_out), # ALWAYS apply downsample for spec
                    ]
                )
            )

        mid_dim = dims[-1] # This is `dim * 8` for dim_mults=(1,2,4,8)
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) # Attn only on 'x'
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling Path (Decoder)
        # Iterate over `in_out` in reverse.
        # The number of upsampling blocks should match the number of downsampling blocks.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            # `ind == 0` corresponds to the deepest part (mid_block output connects here).
            # `ind == (num_downsampling_stages - 1)` corresponds to the shallowest part.
            # The `Upsample` operation should be applied for all but the very last `ups` block.
            # The `Upsample` here is the `ConvTranspose2d` which increases spatial size.
            should_upsample_spatial = ind < (num_downsampling_stages) # `num_downsampling_stages` actual upsample ops
                                                                   # The last block *doesn't* have a spatial upsample if it's the final layer before output.
                                                                   # Corrected condition: The number of upsamples equals the number of downs.
                                                                   # For `len(in_out)` pairs, there are `len(in_out)` downsamples.
                                                                   # So, we need `len(in_out)` upsamples.
                                                                   # The `Upsample` layer in `self.ups.append` refers to the one *before* concat.
                                                                   # The last `ups` block's `upsample_x` should output the original input size.
                                                                   # The `final_block` and `final_conv` handle the last channels.

            # The `Upsample` function uses `dim` for input/output channels.
            # Here, `dim_out` is the current channels from the deeper part.
            # `dim_in` is the channels of the skip connection and the target output channels of the block.
            
            # The `upsample_x` should transform `x` from `dim_in` to `dim_in` (channels wise) but upsample spatially.
            # The `ups` loop has `dim_in, dim_out` which are reversed from `downs`.
            # So, `dim_out` is the larger (deeper) channel count, `dim_in` is the smaller (shallower) channel count.
            # `Upsample(dim_in)` is correct for the `Upsample` layer itself.

            self.ups.append(
                nn.ModuleList(
                    [
                        # Input to block1 after concatenation: (current_x_channels + skip_x_channels)
                        # `dim_out` from `reversed(in_out)` is the channel count from the *previous* upsample block (after its blocks processed),
                        # and also the channel count of the skip connection from the corresponding downsample.
                        # So, `dim_out + dim_out` is the correct input for `block1`.
                        up_block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim), 
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))), # Attn only on 'x'
                        Upsample(dim_in), # Always upsample for each stage in decoder
                        Upsample(dim_in), # Always upsample for spec
                    ]
                )
            )
        
        # After this `ups` loop, `x` and `spec` will be at their original spatial dimensions
        # and will have `dims[1]` (init_dim) channels (the `dim_in` of the last `ups` block).
        # This matches the input channels for `final_block`.
        self.final_block = block_klass(dims[0], dims[0], time_emb_dim=time_dim) # Should be `dims[0]` if it's `init_dim`
        # Wait, the `dims` list is `[init_dim, dim*1, dim*2, dim*4, dim*8]`.
        # The last `dim_in` in `reversed(in_out)` is `init_dim`. So it should be `dims[0]`.
        self.final_conv = nn.Conv2d(dims[0], out_dim, 1)

    @property
    def device(self):
        # Get the device of the first parameter of the model
        return next(self.parameters()).device

    def forward(self, x, waveform, time):
        # x: (B, C, T, F_target) e.g., (1, 3, 128, 16)
        # waveform: (B, Samples) e.g., (1, 5888)

        log.info(f"DEBUG(SpecUnet.forward): Input x shape: {x.shape if exists(x) else 'None'}")
        log.info(f"DEBUG(SpecUnet.forward): Input waveform shape: {waveform.shape if exists(waveform) else 'None'}")

        # Ensure x is on the same device as the model
        if x.device != self.device:
            x = x.to(self.device)
        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        spec = self.mel_layer(waveform) # (B, n_mels, T_mel)
        spec = torch.log(spec + EPSILON)

        # Permute to (B, T_mel, n_mels) for linear layer application on last dim
        spec = spec.transpose(1, 2) # (B, T_mel, n_mels)

        # Apply the linear layer to the last dimension (n_mels)
        # The output shape will be (B, T_mel, inference_frequency_dim)
        spec = self.spec_init_fc(spec)

        # Add a channel dimension: (B, 1, T_mel, inference_frequency_dim)
        spec = spec.unsqueeze(1)

        log.info(f"DEBUG(SpecUnet.forward): Spec shape after linear and unsqueeze: {spec.shape}")

        # Align timesteps
        spec, x = align_timesteps(spec, x)

        log.info(f"DEBUG(SpecUnet.forward): x shape after align_timesteps: {x.shape}")
        log.info(f"DEBUG(SpecUnet.forward): spec shape after align_timesteps: {spec.shape}")

        # Initial convolutions
        x = self.init_conv(x) # (B, init_dim, T_min, F_target)
        spec = self.spec_init_conv(spec) # (B, init_dim, T_min, F_target)

        log.info(f"DEBUG(SpecUnet.forward): x shape after init_conv: {x.shape}")
        log.info(f"DEBUG(SpecUnet.forward): spec shape after spec_init_conv: {spec.shape}")

        t = self.time_mlp(time) if exists(self.time_mlp) else None # (B, time_dim)
        h = [] # For x skips
        spec_h = [] # For spec skips

        # downsample (Encoder)
        for block_idx, (block1, block2, attn, downsample_x, downsample_spec) in enumerate(self.downs):
            x_out, spec_out = block1(x, spec, t)
            x_out, spec_out = block2(x_out, spec_out, t)
            
            x_out = attn(x_out) # Apply attention to the 'x' branch only

            # Append the tensors *before* the downsampling to be used as skip connections
            h.append(x_out)
            spec_h.append(spec_out) 

            # Apply downsample
            x = downsample_x(x_out)
            spec = downsample_spec(spec_out)

            log.info(f"DEBUG(SpecUnet.forward): After downsample block {block_idx} - x shape: {x.shape}, spec shape: {spec.shape}")

        # bottleneck
        x, spec = self.mid_block1(x, spec, t)
        x = self.mid_attn(x) # Assuming attention only on 'x' branch
        x, spec = self.mid_block2(x, spec, t)
        log.info(f"DEBUG(SpecUnet.forward): After mid_blocks - x shape: {x.shape}, spec shape: {spec.shape}")

        # upsample (Decoder)
        for block_idx, (block1, block2, attn, upsample_x, upsample_spec) in enumerate(self.ups):
            # Pop skip connections first, as they are from the higher resolution.
            # They are pushed in order [L1, L2, L3, L4] (L1 shallowest, L4 deepest)
            # So `pop()` will give [L4, L3, L2, L1]
            x_skip = h.pop()
            spec_skip = spec_h.pop()

            # Upsample both x and spec
            # `x` and `spec` here are from the previous (deeper) block's output.
            x = upsample_x(x)
            spec = upsample_spec(spec)
            
            # Critical check for spatial mismatch *before* concatenation
            if x.shape[2:] != x_skip.shape[2:]:
                log.error(f"Spatial dimension mismatch before concatenation in upsample block {block_idx}!")
                log.error(f"x shape: {x.shape}, x_skip shape: {x_skip.shape}")
                # This should no longer happen with the revised Downsample/Upsample structure.
                raise RuntimeError(f"Spatial dimension mismatch in upsample. x: {x.shape}, x_skip: {x_skip.shape}")

            # Concatenate skip connections
            x = torch.cat((x, x_skip), dim=1)
            spec = torch.cat((spec, spec_skip), dim=1) 

            log.info(f"DEBUG(SpecUnet.forward): After upsample {block_idx} and concat - x shape: {x.shape}, spec shape: {spec.shape}")

            # Apply blocks
            x, spec = block1(x, spec, t) # block1 now takes concatenated x and spec
            x, spec = block2(x, spec, t)
            x = attn(x) # Assuming attention only on 'x' branch

            log.info(f"DEBUG(SpecUnet.forward): After upsample blocks {block_idx} - x shape: {x.shape}, spec shape: {spec.shape}")

        # Final block and convolution
        # `x` should have `dims[0]` channels (init_dim) and original spatial dimensions here.
        x, _ = self.final_block(x, spec, t) 
        x = self.final_conv(x)

        log.info(f"DEBUG(SpecUnet.forward): Final output x shape: {x.shape}")

        return x
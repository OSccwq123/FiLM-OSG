"""
OSG-compatible extra backbones for FiLM-OSG experiments on 2D regular-grid PDE data.

Place this file at:
    due/networks/osg_extra_backbones.py

It provides paired direct-lag and FiLM-conditioned variants:

    osg_uno2d / osg_uno2d_with_film
    osg_mambano2d / osg_mambano2d_with_film
    osg_transolver2d / osg_transolver2d_with_film

Design contract:
    - input  x: (B, H, W, C + 1), last channel is normalized lag code delta
    - output y: (B, H, W, C), using OSG outer-increment update
              y = x0 + physical_dt * increment
    - predict(x, dt, device) follows the same normalization / autoregressive
      rollout style as the existing osg_fno2d implementation.

Important implementation note:
    due.utils.get_activation("gelu") may return torch.nn.functional.gelu,
    i.e. a function, not a torch.nn.Module. Therefore, whenever an activation
    is inserted into torch.nn.Sequential, we wrap it with ActivationModule.
"""

import torch
import torch.nn.functional as F

from .nn import nn as DUEBase
from ..utils import get_activation


# ---------------------------------------------------------------------
# Optional Mamba dependency
# ---------------------------------------------------------------------

try:
    from mamba_ssm import Mamba as _OfficialMamba
except Exception:
    _OfficialMamba = None


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

class ActivationModule(torch.nn.Module):
    """Wrap a functional activation so it can be used inside nn.Sequential."""
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)


def as_module_activation(activation_fn):
    if isinstance(activation_fn, torch.nn.Module):
        return activation_fn
    return ActivationModule(activation_fn)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


def make_time_encoder(time_width: int, film_channels):
    """Return an MLP that outputs raw gamma/beta vectors for all FiLM sites."""
    total = 2 * int(sum(film_channels))
    enc = torch.nn.Sequential(
        torch.nn.Linear(1, time_width),
        torch.nn.GELU(),
        ResidualMLPBlock(time_width),
        ResidualMLPBlock(time_width),
        torch.nn.Linear(time_width, total),
    )

    # Zero initialization makes the FiLM path start close to identity:
    # gamma = 1 + 0.1 * raw_gamma, beta = 0.1 * raw_beta.
    torch.nn.init.zeros_(enc[-1].weight)
    torch.nn.init.zeros_(enc[-1].bias)
    return enc


def split_film(film, film_channels):
    """Split raw FiLM output into [(gamma, beta), ...]."""
    out = []
    cursor = 0
    for c in film_channels:
        c = int(c)
        raw_gamma = film[:, cursor:cursor + c]
        cursor += c
        raw_beta = film[:, cursor:cursor + c]
        cursor += c

        gamma = 1.0 + 0.1 * raw_gamma
        beta = 0.1 * raw_beta
        out.append((gamma, beta))
    return out


def apply_film_2d(x, gamma, beta):
    """x: (B, C, H, W), gamma/beta: (B, C)."""
    return gamma[:, :, None, None] * x + beta[:, :, None, None]


def apply_film_tokens(x, gamma, beta):
    """x: (B, N, C), gamma/beta: (B, C)."""
    return gamma[:, None, :] * x + beta[:, None, :]


def decode_dt(dt_norm, tmin, tmax, multiscale):
    """Decode normalized lag code to physical lag."""
    dt = dt_norm * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
    if multiscale:
        dt = 10.0 ** dt
    return dt


def make_grid_features(batch_size, height, width, device, dtype):
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


class BaseOSG2D(DUEBase):
    """Shared normalization and autoregressive prediction logic."""

    def _normalize_state(self, x):
        vmin = self.vmin.to(x.device)[..., 0]
        vmax = self.vmax.to(x.device)[..., 0]
        return 2.0 * (x - 0.5 * (vmax + vmin)) / (vmax - vmin)

    def _denormalize_rollout(self, y):
        vmin = self.vmin.to(y.device)
        vmax = self.vmax.to(y.device)
        return y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)

    def _normalize_dt_array(self, dt, device):
        dt = torch.from_numpy(dt).float().to(device)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2.0 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        return dt

    def predict(self, x, dt, device):
        self.to(device)
        assert x.shape[-1] == self.output_dim

        steps = dt.shape[1]
        dt_norm = self._normalize_dt_array(dt, device)

        x = torch.from_numpy(x).float().to(device)
        x = self._normalize_state(x)

        y = torch.unsqueeze(x.clone(), -1)
        self.eval()

        with torch.no_grad():
            for t in range(steps):
                dt_t = dt_norm[:, t][:, None, None, None].repeat(
                    1, y.shape[1], y.shape[2], 1
                )
                xx = torch.cat((y[..., -1], dt_t), dim=-1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = self._denormalize_rollout(y)
        return y.cpu().numpy()


# ---------------------------------------------------------------------
# 1. U-NO-style operator backbone
# ---------------------------------------------------------------------

class SpectralConv2dUno(torch.nn.Module):
    """
    2D UNO spectral integral block.
    Allows output size different from input size.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = (1.0 / max(1, 2 * self.in_channels)) ** 0.5
        self.weights1 = torch.nn.Parameter(
            scale * torch.randn(
                self.in_channels, self.out_channels,
                self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = torch.nn.Parameter(
            scale * torch.randn(
                self.in_channels, self.out_channels,
                self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul2d(x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x, out_h=None, out_w=None):
        B, C, H, W = x.shape
        out_h = H if out_h is None else int(out_h)
        out_w = W if out_w is None else int(out_w)

        x_ft = torch.fft.rfft2(x, norm="forward")
        out_ft = torch.zeros(
            B, self.out_channels, out_h, out_w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, x_ft.shape[-2], out_ft.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1], out_ft.shape[-1])

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        return torch.fft.irfft2(out_ft, s=(out_h, out_w), norm="forward")


class PointwiseResize2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, x, out_h=None, out_w=None):
        x = self.conv(x)
        if out_h is not None and out_w is not None and x.shape[-2:] != (out_h, out_w):
            x = F.interpolate(x, size=(int(out_h), int(out_w)), mode="bicubic", align_corners=True)
        return x


class UnoOperatorBlock2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, activation, normalize=False):
        super().__init__()
        self.spectral = SpectralConv2dUno(in_channels, out_channels, modes1, modes2)
        self.pointwise = PointwiseResize2d(in_channels, out_channels)
        self.normalize = bool(normalize)
        self.activation = activation
        self.norm = torch.nn.InstanceNorm2d(int(out_channels), affine=True) if normalize else torch.nn.Identity()

    def forward(self, x, out_h=None, out_w=None, nonlin=True):
        y = self.spectral(x, out_h, out_w) + self.pointwise(x, out_h, out_w)
        y = self.norm(y)
        if nonlin:
            y = self.activation(y)
        return y


class osg_uno2d(BaseOSG2D):
    """
    Direct-lag OSG-U-NO.

    Lag code is concatenated as a broadcast input channel.
    """
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.modes1 = int(config.get("modes1", 12))
        self.modes2 = int(config.get("modes2", 12))
        self.activation = get_activation(config["activation"])
        self.normalize = bool(config.get("uno_norm", False))

        w = self.width

        self.lift = torch.nn.Conv2d(self.output_dim + 1, w, kernel_size=1)

        self.b0 = UnoOperatorBlock2d(w, w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b1 = UnoOperatorBlock2d(w, 2 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b2 = UnoOperatorBlock2d(2 * w, 4 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b3 = UnoOperatorBlock2d(4 * w, 4 * w, self.modes1, self.modes2, self.activation, self.normalize)

        self.u2 = UnoOperatorBlock2d(8 * w, 2 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.u1 = UnoOperatorBlock2d(4 * w, w, self.modes1, self.modes2, self.activation, self.normalize)

        self.proj = torch.nn.Sequential(
            torch.nn.Conv2d(w, 4 * w, kernel_size=1),
            as_module_activation(self.activation),
            torch.nn.Conv2d(4 * w, self.output_dim, kernel_size=1),
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        z = x.permute(0, 3, 1, 2)
        h = self.lift(z)

        h0 = self.b0(h, H, W)
        h1 = self.b1(h0, max(1, H // 2), max(1, W // 2))
        h2 = self.b2(h1, max(1, H // 4), max(1, W // 4))
        h3 = self.b3(h2, max(1, H // 4), max(1, W // 4))

        u2 = self.u2(torch.cat((h3, h2), dim=1), max(1, H // 2), max(1, W // 2))
        u1 = self.u1(torch.cat((u2, h1), dim=1), H, W)

        incr = self.proj(u1 + h0).permute(0, 2, 3, 1)
        return x0 + dt * incr


class osg_uno2d_with_film(BaseOSG2D):
    """
    FiLM-OSG-U-NO.

    Lag code drives layerwise FiLM; the state encoder does not receive lag as an
    input channel.
    """
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.modes1 = int(config.get("modes1", 12))
        self.modes2 = int(config.get("modes2", 12))
        self.activation = get_activation(config["activation"])
        self.normalize = bool(config.get("uno_norm", False))

        w = self.width

        self.lift = torch.nn.Conv2d(self.output_dim, w, kernel_size=1)

        self.b0 = UnoOperatorBlock2d(w, w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b1 = UnoOperatorBlock2d(w, 2 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b2 = UnoOperatorBlock2d(2 * w, 4 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.b3 = UnoOperatorBlock2d(4 * w, 4 * w, self.modes1, self.modes2, self.activation, self.normalize)

        self.u2 = UnoOperatorBlock2d(8 * w, 2 * w, self.modes1, self.modes2, self.activation, self.normalize)
        self.u1 = UnoOperatorBlock2d(4 * w, w, self.modes1, self.modes2, self.activation, self.normalize)

        self.proj = torch.nn.Sequential(
            torch.nn.Conv2d(w, 4 * w, kernel_size=1),
            as_module_activation(self.activation),
            torch.nn.Conv2d(4 * w, self.output_dim, kernel_size=1),
        )

        self.film_channels = [w, 2 * w, 4 * w, 4 * w, 2 * w, w]
        self.time_encoder = make_time_encoder(
            int(config.get("time_width", w)),
            self.film_channels,
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        dt_scalar = dt_norm.mean(dim=(1, 2))
        film = split_film(self.time_encoder(dt_scalar), self.film_channels)

        z = x0.permute(0, 3, 1, 2)
        h = self.lift(z)

        # In the FiLM version, the block output is kept pre-activation.
        # We apply FiLM first and then use one activation:
        #     activation(gamma * block_output + beta).
        # This avoids unintended double activation.
        h0 = self.b0(h, H, W, nonlin=False)
        h0 = self.activation(apply_film_2d(h0, *film[0]))

        h1 = self.b1(h0, max(1, H // 2), max(1, W // 2), nonlin=False)
        h1 = self.activation(apply_film_2d(h1, *film[1]))

        h2 = self.b2(h1, max(1, H // 4), max(1, W // 4), nonlin=False)
        h2 = self.activation(apply_film_2d(h2, *film[2]))

        h3 = self.b3(h2, max(1, H // 4), max(1, W // 4), nonlin=False)
        h3 = self.activation(apply_film_2d(h3, *film[3]))

        u2 = self.u2(torch.cat((h3, h2), dim=1), max(1, H // 2), max(1, W // 2), nonlin=False)
        u2 = self.activation(apply_film_2d(u2, *film[4]))

        u1 = self.u1(torch.cat((u2, h1), dim=1), H, W, nonlin=False)
        u1 = self.activation(apply_film_2d(u1, *film[5]))

        incr = self.proj(u1 + h0).permute(0, 2, 3, 1)
        return x0 + dt * incr


# ---------------------------------------------------------------------
# 2. MambaNO-style backbone
# ---------------------------------------------------------------------

class FallbackTokenMixer(torch.nn.Module):
    """Fallback token mixer used when mamba_ssm is unavailable."""
    def __init__(self, width):
        super().__init__()
        self.dw = torch.nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width)
        self.pw = torch.nn.Linear(width, width)

    def forward(self, x):
        # x: (B, N, C)
        y = x.transpose(1, 2)
        y = self.dw(y).transpose(1, 2)
        return self.pw(y)


class MambaTokenMixer(torch.nn.Module):
    """Uses official mamba_ssm.Mamba when available; otherwise a fallback mixer."""
    def __init__(self, width, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.uses_official_mamba = _OfficialMamba is not None
        if self.uses_official_mamba:
            self.mixer = _OfficialMamba(
                d_model=width,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = FallbackTokenMixer(width)

    def forward(self, x):
        return self.mixer(x)


class MambaNOBlock2d(torch.nn.Module):
    """
    MambaNO-style block with:
      - global row-major scan
      - global column-major scan
      - local depthwise 2D convolution branch
      - MLP channel mixing
    """
    def __init__(self, width, activation, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.width = int(width)
        self.norm_row = torch.nn.LayerNorm(width)
        self.norm_col = torch.nn.LayerNorm(width)

        self.row_mixer = MambaTokenMixer(width, d_state=d_state, d_conv=d_conv, expand=expand)
        self.col_mixer = MambaTokenMixer(width, d_state=d_state, d_conv=d_conv, expand=expand)

        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width),
            torch.nn.Conv2d(width, width, kernel_size=1),
        )

        self.norm_mlp = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width),
            as_module_activation(activation),
            torch.nn.Linear(4 * width, width),
        )

        self.scale = torch.nn.Parameter(torch.ones(3))

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, N, C = x.shape
        assert N == H * W

        row = self.row_mixer(self.norm_row(x))

        x_img = x.reshape(B, H, W, C)
        col_tokens = x_img.transpose(1, 2).reshape(B, W * H, C)
        col = self.col_mixer(self.norm_col(col_tokens))
        col = col.reshape(B, W, H, C).transpose(1, 2).reshape(B, H * W, C)

        local = x_img.permute(0, 3, 1, 2)
        local = self.local(local).permute(0, 2, 3, 1).reshape(B, H * W, C)

        weights = torch.softmax(self.scale, dim=0)
        x = x + weights[0] * row + weights[1] * col + weights[2] * local
        x = x + self.mlp(self.norm_mlp(x))
        return x


class osg_mambano2d(BaseOSG2D):
    """Direct-lag OSG-MambaNO-style backbone."""
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.depth = int(config.get("depth", 4))
        self.activation = get_activation(config["activation"])

        self.d_state = int(config.get("mamba_d_state", 16))
        self.d_conv = int(config.get("mamba_d_conv", 4))
        self.expand = int(config.get("mamba_expand", 2))

        self.en = torch.nn.Linear(self.output_dim + 1 + 2, self.width)
        self.blocks = torch.nn.ModuleList([
            MambaNOBlock2d(
                self.width,
                self.activation,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
            for _ in range(self.depth)
        ])
        self.de = torch.nn.Sequential(
            torch.nn.LayerNorm(self.width),
            torch.nn.Linear(self.width, 4 * self.width),
            as_module_activation(self.activation),
            torch.nn.Linear(4 * self.width, self.output_dim),
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        grid = make_grid_features(B, H, W, x.device, x.dtype)
        feat = torch.cat((grid, x0, dt_norm), dim=-1)

        h = self.en(feat).reshape(B, H * W, self.width)
        for block in self.blocks:
            h = block(h, H, W)

        incr = self.de(h).reshape(B, H, W, self.output_dim)
        return x0 + dt * incr


class osg_mambano2d_with_film(BaseOSG2D):
    """FiLM-OSG-MambaNO-style backbone."""
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.depth = int(config.get("depth", 4))
        self.activation = get_activation(config["activation"])

        self.d_state = int(config.get("mamba_d_state", 16))
        self.d_conv = int(config.get("mamba_d_conv", 4))
        self.expand = int(config.get("mamba_expand", 2))

        self.en = torch.nn.Linear(self.output_dim + 2, self.width)
        self.blocks = torch.nn.ModuleList([
            MambaNOBlock2d(
                self.width,
                self.activation,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
            for _ in range(self.depth)
        ])
        self.de = torch.nn.Sequential(
            torch.nn.LayerNorm(self.width),
            torch.nn.Linear(self.width, 4 * self.width),
            as_module_activation(self.activation),
            torch.nn.Linear(4 * self.width, self.output_dim),
        )

        self.film_channels = [self.width for _ in range(self.depth)]
        self.time_encoder = make_time_encoder(
            int(config.get("time_width", self.width)),
            self.film_channels,
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        grid = make_grid_features(B, H, W, x.device, x.dtype)
        dt_scalar = dt_norm.mean(dim=(1, 2))

        film = split_film(self.time_encoder(dt_scalar), self.film_channels)

        feat = torch.cat((grid, x0), dim=-1)
        h = self.en(feat).reshape(B, H * W, self.width)

        for i, block in enumerate(self.blocks):
            h = block(h, H, W)
            h = self.activation(apply_film_tokens(h, *film[i]))

        incr = self.de(h).reshape(B, H, W, self.output_dim)
        return x0 + dt * incr


# ---------------------------------------------------------------------
# 3. Transolver-style backbone with Physics Attention
# ---------------------------------------------------------------------

class PhysicsAttention2d(torch.nn.Module):
    """
    Transolver-style physics attention:
      (1) slice tokens from point/node tokens,
      (2) self-attend among slice tokens,
      (3) deslice back to point/node tokens.
    """
    def __init__(self, dim, heads=4, dim_head=None, dropout=0.0, slice_num=32):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim // heads if dim_head is None else dim_head)
        inner_dim = self.heads * self.dim_head

        if inner_dim != self.dim:
            raise ValueError(
                f"hidden dim={self.dim} must be divisible by heads={self.heads}, "
                f"or pass a compatible dim_head."
            )

        self.scale = self.dim_head ** -0.5
        self.temperature = torch.nn.Parameter(torch.ones(1, self.heads, 1, 1) * 0.5)

        self.in_project_x = torch.nn.Linear(dim, inner_dim)
        self.in_project_fx = torch.nn.Linear(dim, inner_dim)
        self.in_project_slice = torch.nn.Linear(self.dim_head, int(slice_num))

        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = torch.nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = torch.nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = torch.nn.Linear(self.dim_head, self.dim_head, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()  # B,H,N,D

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()  # B,H,N,D

        # Slice
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = torch.softmax(slice_logits, dim=-1)  # B,H,N,G
        slice_norm = slice_weights.sum(dim=2)  # B,H,G

        slice_token = torch.einsum("bhnd,bhng->bhgd", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[:, :, :, None] + 1e-5)

        # Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out_slice = torch.matmul(attn, v)  # B,H,G,D

        # Deslice
        out = torch.einsum("bhgd,bhng->bhnd", out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)

        return self.to_out(out)


class TransolverBlock(torch.nn.Module):
    def __init__(self, hidden_dim, heads, dropout, activation, mlp_ratio=4, slice_num=32):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention2d(
            hidden_dim,
            heads=heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            as_module_activation(activation),
            torch.nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class osg_transolver2d(BaseOSG2D):
    """
    Direct-lag OSG-Transolver-style backbone.
    Uses physics attention over regular-grid tokens.
    """
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.depth = int(config.get("depth", 4))
        self.n_head = int(config.get("n_head", 4))
        self.slice_num = int(config.get("slice_num", 32))
        self.dropout = float(config.get("dropout", 0.0))
        self.mlp_ratio = int(config.get("mlp_ratio", 4))
        self.activation = get_activation(config["activation"])

        self.pre = torch.nn.Linear(self.output_dim + 1 + 2, self.width)
        self.blocks = torch.nn.ModuleList([
            TransolverBlock(
                self.width,
                heads=self.n_head,
                dropout=self.dropout,
                activation=self.activation,
                mlp_ratio=self.mlp_ratio,
                slice_num=self.slice_num,
            )
            for _ in range(self.depth)
        ])
        self.post = torch.nn.Sequential(
            torch.nn.LayerNorm(self.width),
            torch.nn.Linear(self.width, 4 * self.width),
            as_module_activation(self.activation),
            torch.nn.Linear(4 * self.width, self.output_dim),
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        grid = make_grid_features(B, H, W, x.device, x.dtype)
        feat = torch.cat((grid, x0, dt_norm), dim=-1)

        h = self.pre(feat).reshape(B, H * W, self.width)
        for block in self.blocks:
            h = block(h)

        incr = self.post(h).reshape(B, H, W, self.output_dim)
        return x0 + dt * incr


class osg_transolver2d_with_film(BaseOSG2D):
    """FiLM-OSG-Transolver-style backbone."""
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=False):
        super().__init__()
        self.register_buffer("vmin", torch.from_numpy(vmin).float())
        self.register_buffer("vmax", torch.from_numpy(vmax).float())

        self.tmin = tmin
        self.tmax = tmax
        self.multiscale = multiscale

        self.output_dim = config["problem_dim"]
        self.width = int(config.get("width", 20))
        self.depth = int(config.get("depth", 4))
        self.n_head = int(config.get("n_head", 4))
        self.slice_num = int(config.get("slice_num", 32))
        self.dropout = float(config.get("dropout", 0.0))
        self.mlp_ratio = int(config.get("mlp_ratio", 4))
        self.activation = get_activation(config["activation"])

        self.pre = torch.nn.Linear(self.output_dim + 2, self.width)
        self.blocks = torch.nn.ModuleList([
            TransolverBlock(
                self.width,
                heads=self.n_head,
                dropout=self.dropout,
                activation=self.activation,
                mlp_ratio=self.mlp_ratio,
                slice_num=self.slice_num,
            )
            for _ in range(self.depth)
        ])
        self.post = torch.nn.Sequential(
            torch.nn.LayerNorm(self.width),
            torch.nn.Linear(self.width, 4 * self.width),
            as_module_activation(self.activation),
            torch.nn.Linear(4 * self.width, self.output_dim),
        )

        self.film_channels = [self.width for _ in range(self.depth)]
        self.time_encoder = make_time_encoder(
            int(config.get("time_width", self.width)),
            self.film_channels,
        )

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        dt = decode_dt(dt_norm, self.tmin, self.tmax, self.multiscale)

        B, H, W, _ = x0.shape
        grid = make_grid_features(B, H, W, x.device, x.dtype)
        dt_scalar = dt_norm.mean(dim=(1, 2))
        film = split_film(self.time_encoder(dt_scalar), self.film_channels)

        feat = torch.cat((grid, x0), dim=-1)
        h = self.pre(feat).reshape(B, H * W, self.width)

        for i, block in enumerate(self.blocks):
            h = block(h)
            h = self.activation(apply_film_tokens(h, *film[i]))

        incr = self.post(h).reshape(B, H, W, self.output_dim)
        return x0 + dt * incr

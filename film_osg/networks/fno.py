"""FNO backbones for FiLM-OSG reproducibility experiments.

Portions of this module are adapted from the DUE project:
https://github.com/AI4Equations/due
DUE is distributed under the LGPL-2.1 license. The Fourier neural operator
implementation also follows the public neuraloperator reference:
https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d_time.py
"""

import torch
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
from .nn import nn
from ..utils import get_activation
torch.manual_seed(0)

class SpectralConv2d(nn):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn):
    def __init__(self, in_channels, out_channels, mid_channels, activation):
        super(MLP, self).__init__()
        self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class GlobalLocalFNOBlock2d(nn):
    """Global-local FNO block with a lightweight 2D local correction."""

    def __init__(
        self,
        channels,
        modes1,
        modes2,
        activation,
        kernel_size=3,
        pool_factor=2,
        layer_scale=1e-3,
        post_activation=True,
        local_padding_mode="zeros",
        coupling_mode="raw",
        coupling_scale=1.0,
        local_scale=1.0,
    ):
        super().__init__()
        self.activation = activation
        self.pool_factor = max(1, int(pool_factor))
        self.post_activation = bool(post_activation)
        self.coupling_mode = str(coupling_mode)
        self.coupling_scale = float(coupling_scale)
        self.local_scale = float(local_scale)
        valid_coupling = {"raw", "tanh"}
        if self.coupling_mode not in valid_coupling:
            raise ValueError(f"Unknown coupling_mode={self.coupling_mode}; expected one of {sorted(valid_coupling)}")
        padding_mode = str(local_padding_mode)
        if padding_mode == "zeros":
            padding_mode = "zeros"
        elif padding_mode != "circular":
            raise ValueError("local_padding_mode must be 'zeros' or 'circular'")
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.spectral_mlp = MLP(channels, channels, channels, activation)
        self.spectral_w = torch.nn.Conv2d(channels, channels, 1)
        padding = int(kernel_size) // 2
        self.local_conv1 = torch.nn.Conv2d(
            channels, channels, int(kernel_size), padding=padding, groups=channels, padding_mode=padding_mode
        )
        self.local_pw1 = torch.nn.Conv2d(channels, channels, 1)
        self.local_conv2 = torch.nn.Conv2d(
            channels, channels, int(kernel_size), padding=padding, groups=channels, padding_mode=padding_mode
        )
        self.local_pw2 = torch.nn.Conv2d(channels, channels, 1)
        self.local_fuse = torch.nn.Conv2d(2 * channels, channels, 1)
        self.global_gate = torch.nn.Conv2d(channels, channels, 1)
        self.local_gate = torch.nn.Conv2d(channels, channels, 1)
        self.mix = torch.nn.Conv2d(4 * channels, channels, 1)
        self.layer_scale = torch.nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale)))

    def _local_branch(self, x):
        if self.pool_factor > 1 and min(x.shape[-2:]) >= self.pool_factor:
            coarse = torch.nn.functional.avg_pool2d(x, kernel_size=self.pool_factor, stride=self.pool_factor)
        else:
            coarse = x
        coarse = self.local_conv1(coarse)
        coarse = self.local_pw1(coarse)
        coarse = self.activation(coarse)
        coarse = self.local_conv2(coarse)
        coarse = self.local_pw2(coarse)
        if coarse.shape[-2:] != x.shape[-2:]:
            coarse = torch.nn.functional.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.local_scale * self.activation(self.local_fuse(torch.cat((x, coarse), dim=1)))

    def forward(self, x, global_film=None, local_film=None):
        g = self.spectral_mlp(self.spectral(x)) + self.spectral_w(x)
        g = self.activation(g)
        l = self._local_branch(x)
        if global_film is not None:
            gamma, beta = global_film
            g = gamma * g + beta
        if local_film is not None:
            gamma, beta = local_film
            l = gamma * l + beta
        cg = self.global_gate(g)
        cl = self.local_gate(l)
        coupling_mode = getattr(self, "coupling_mode", "raw")
        if coupling_mode == "tanh":
            cg = torch.tanh(cg)
            cl = torch.tanh(cl)
        c = getattr(self, "coupling_scale", 1.0) * cg * cl
        update = self.mix(torch.cat((x, g, l, c), dim=1))
        out = x + self.layer_scale * update
        if getattr(self, "post_activation", False):
            out = self.activation(out)
        return out


class osg_fno2d(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno2d, self).__init__()

        self.vmin       = torch.from_numpy(vmin)
        self.vmax       = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim  = config["problem_dim"]+1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes1"]
        self.modes2 = config["modes2"]
        self.nblocks = config["depth"]
        self.hid_dim  = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp  = torch.nn.ModuleList()
        self.w    = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv2d(self.hid_dim, self.hid_dim, self.modes1, self.modes2))
            self.mlp.append(MLP(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv2d(self.hid_dim, self.hid_dim, 1))
        self.de = MLP(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation) # output channel is 1: u(x, y)

    def forward(self, x):
        
        x0   = x[...,:-1]
        dt   = x[...,-1:] * 0.5 * (self.tmax-self.tmin) + 0.5 * (self.tmax+self.tmin)
        x = self.en(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)
        if getattr(self, "conserve_mean", False):
            x = x - x.mean(dim=(1, 2), keepdim=True)
        return x0 + x * dt
        
    def predict(self, x, dt, device):
        self.to(device)

        vmin = self.vmin.to(device)
        vmax = self.vmax.to(device)

        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        dt = torch.from_numpy(dt).to(device)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.tile(dt, [1, x.shape[1], x.shape[2], 1])
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)

        x = torch.from_numpy(x).to(device)
        x = 2 * (x - 0.5 * (vmax[..., 0] + vmin[..., 0])) / (vmax[..., 0] - vmin[..., 0])

        y = torch.unsqueeze(x.clone(), -1)
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[..., -1], dt[..., t:t+1]), dim=-1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)
        y = y.cpu()

        return y.numpy()



class vt_fno2d(osg_fno2d):
    """Variable-time FNO baseline without OSG outer-increment structure."""

    def forward(self, x):
        x = self.en(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = self.activation(x1 + x2)
        x = self.de(x)
        return x.permute(0, 2, 3, 1)

class osg_fno2d_with_film(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno2d_with_film, self).__init__()

        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim = config["problem_dim"]
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes1"]
        self.modes2 = config["modes2"]
        self.nblocks = config["depth"]
        self.hid_dim = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))
        
        self.film_dim = config["width"]
        self.time_encoder = self._build_time_encoder()
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.w = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv2d(self.hid_dim, self.hid_dim, self.modes1, self.modes2))
            self.mlp.append(MLP(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv2d(self.hid_dim, self.hid_dim, 1))
            
        self.de = MLP(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def _build_time_encoder(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim),
            torch.nn.GELU(),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            torch.nn.Linear(self.film_dim, 2 * self.nblocks * self.hid_dim)
        )

        last = net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

        return net
        
    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]

        batch_size = x.shape[0]

        dt_scalar = dt_norm.mean(dim=[1, 2])   # [B, 1]

        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)

        raw_gammas = film_params[:, :, 0, :]
        raw_betas  = film_params[:, :, 1, :]

        gammas = 1.0 + 0.1 * raw_gammas
        betas  = 0.1 * raw_betas

        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt

        x = self.en(x0)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2

            gamma = gammas[:, i, :].view(batch_size, self.hid_dim, 1, 1)
            beta  = betas[:, i, :].view(batch_size, self.hid_dim, 1, 1)

            x = gamma * x + beta
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)
        if getattr(self, "conserve_mean", False):
            x = x - x.mean(dim=(1, 2), keepdim=True)

        return x0 + x * dt
    
    def predict(self, x, dt, device):
        self.to(device)

        vmin = self.vmin.to(device)
        vmax = self.vmax.to(device)

        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        dt = torch.from_numpy(dt).to(device)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.tile(dt, [1, x.shape[1], x.shape[2], 1])
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)

        x = torch.from_numpy(x).to(device)
        x = 2 * (x - 0.5 * (vmax[..., 0] + vmin[..., 0])) / (vmax[..., 0] - vmin[..., 0])

        y = torch.unsqueeze(x.clone(), -1)
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[..., -1], dt[..., t:t+1]), dim=-1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)
        y = y.cpu()

        return y.numpy()

class vt_fno2d_with_film(osg_fno2d_with_film):
    """Time-conditioned FiLM-FNO baseline with direct next-state prediction.

    This is an external variable-time baseline: the lag is used only by the
    FiLM encoder, while the network directly predicts the normalized next
    state. It intentionally does not use the OSG outer-increment update.
    """

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        batch_size = x.shape[0]

        dt_scalar = dt_norm.mean(dim=[1, 2])
        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)

        raw_gammas = film_params[:, :, 0, :]
        raw_betas = film_params[:, :, 1, :]
        gammas = 1.0 + 0.1 * raw_gammas
        betas = 0.1 * raw_betas

        x = self.en(x0)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2

            gamma = gammas[:, i, :].view(batch_size, self.hid_dim, 1, 1)
            beta = betas[:, i, :].view(batch_size, self.hid_dim, 1, 1)

            x = gamma * x + beta
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)
        return x


class osg_fno2d_with_film_ablation(osg_fno2d_with_film):
    """
    FiLM-OSG-FNO ablation variant.

    This class reuses the standard FiLM-OSG-FNO implementation but allows
    controlled ablations of the FiLM modulation.

    Supported config keys:
        film_mode:
            "full"       -> gamma + beta
            "gamma_only" -> gamma only, beta disabled
            "beta_only"  -> beta only, gamma disabled
            "none"       -> no FiLM modulation

        film_placement:
            "all"   -> apply FiLM at every FNO block
            "last"  -> apply FiLM only at the last FNO block
            "first" -> apply FiLM only at the first FNO block
            "none"  -> apply no FiLM modulation
    """

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno2d_with_film_ablation, self).__init__(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=multiscale,
        )

        self.film_mode = config.get("film_mode", "full")
        self.film_placement = config.get("film_placement", "all")

        valid_modes = {"full", "gamma_only", "beta_only", "none"}
        valid_placements = {"all", "last", "first", "none"}

        if self.film_mode not in valid_modes:
            raise ValueError(
                f"Unknown film_mode={self.film_mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        if self.film_placement not in valid_placements:
            raise ValueError(
                f"Unknown film_placement={self.film_placement}. "
                f"Expected one of {sorted(valid_placements)}."
            )

    def _use_film_at_block(self, block_id):
        if self.film_placement == "none":
            return False
        if self.film_placement == "all":
            return True
        if self.film_placement == "first":
            return block_id == 0
        if self.film_placement == "last":
            return block_id == self.nblocks - 1
        raise RuntimeError(f"Unhandled film_placement={self.film_placement}")

    def _apply_film_mode(self, gammas, betas):
        """
        gammas, betas have shape [B, nblocks, hid_dim].
        We keep the same time_encoder output size across variants. This makes
        the ablation change only how the modulation is used, rather than
        changing the surrounding FNO structure.
        """
        if self.film_mode == "full":
            return gammas, betas

        if self.film_mode == "gamma_only":
            return gammas, torch.zeros_like(betas)

        if self.film_mode == "beta_only":
            return torch.ones_like(gammas), betas

        if self.film_mode == "none":
            return torch.ones_like(gammas), torch.zeros_like(betas)

        raise RuntimeError(f"Unhandled film_mode={self.film_mode}")

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]

        batch_size = x.shape[0]

        # Since dt is broadcast spatially, the spatial mean recovers the scalar code.
        dt_scalar = dt_norm.mean(dim=[1, 2])  # [B, 1]

        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)

        raw_gammas = film_params[:, :, 0, :]
        raw_betas = film_params[:, :, 1, :]

        gammas = 1.0 + 0.1 * raw_gammas
        betas = 0.1 * raw_betas

        gammas, betas = self._apply_film_mode(gammas, betas)

        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt

        x = self.en(x0)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2

            if self._use_film_at_block(i):
                gamma = gammas[:, i, :].view(batch_size, self.hid_dim, 1, 1)
                beta = betas[:, i, :].view(batch_size, self.hid_dim, 1, 1)
                x = gamma * x + beta

            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)

        return x0 + x * dt
    

class Lightweight_osg_fno2d(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(Lightweight_osg_fno2d, self).__init__()

        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim = config["problem_dim"] + 1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes1"]
        self.modes2 = config["modes2"]
        self.nblocks = config["depth"]
        self.hid_dim = config["width"]
        self.multiscale = multiscale
        
        self.film_dim = config["width"] // 2
        self.time_encoder = self._build_time_encoder()
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.w = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv2d(self.hid_dim, self.hid_dim, self.modes1, self.modes2))
            self.mlp.append(MLP(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv2d(self.hid_dim, self.hid_dim, 1))
            
        self.de = MLP(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def _build_time_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim),
            torch.nn.GELU(),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            torch.nn.Linear(self.film_dim, 2* self.hid_dim)
        )
        
    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        
        batch_size = x.shape[0]
        
        dt_scalar = dt_norm.mean(dim=[1, 2])

        raw = self.time_encoder(dt_scalar)
        raw = raw.view(batch_size, 2, self.hid_dim)
        delta_gammas = raw[:, 0, :]
        delta_betas = raw[:, 1, :] 
        
        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt

        dt_mean = dt.mean(dim=[1, 2])
        gammas = 1 + dt_mean * delta_gammas
        betas = dt_mean * delta_betas

        x = self.en(x)  
        x = x.permute(0, 3, 1, 2)  

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2
            
            if i == self.nblocks - 1:      
                gamma = gammas.view(batch_size, self.hid_dim, 1, 1)
                beta = betas.view(batch_size, self.hid_dim, 1, 1)
                x = gamma * x + beta

            
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1) 
        
        return x0 + x * dt
    
    def predict(self, x, dt, device):
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]
        
        dt = torch.from_numpy(dt)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.tile(dt, [1, x.shape[1], x.shape[2], 1])
        dt = dt.to(device)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        
        x = torch.from_numpy(x)
        x = 2 * (x - 0.5 * (self.vmax[..., 0] + self.vmin[..., 0])) / (self.vmax[..., 0] - self.vmin[..., 0])
        x = x.to(device)

        y = torch.unsqueeze(x.clone(), -1)
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[..., -1], dt[..., t:t+1]), dim=-1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        y = y.cpu()        
        y = y * 0.5 * (self.vmax - self.vmin) + 0.5 * (self.vmax + self.vmin)
        
        return y.numpy()
    
class dual_osg_fno2d(osg_fno2d):
    """
    Dual-OSG_FNO2D built upon two OSG-FNO2D networks and a gating network
    """
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        
        self.osg_fno1 = osg_fno2d(vmin, vmax, tmin, tmax, config, multiscale)
        self.osg_fno2 = osg_fno2d(vmin, vmax, tmin, tmax, config, multiscale)
        self.gate    = torch.nn.ModuleList()
        
        self.gate.append(torch.nn.Linear(1, self.hid_dim))
        self.gate.append(torch.nn.Linear(self.hid_dim, 2))
        
        self.temperature = 1.0
        self.register_buffer('current_temperature', torch.tensor(1.0))
        
        self.name = "dual_osg_fno2d"
                    
    def forward(self, x):
        dt_norm = x[..., -1:]
        
        p = torch.nn.Softmax(dim=-1)(self.gate[1](self.activation(self.gate[0](dt_norm))) / self.current_temperature)
        
        y1 = self.osg_fno1(x)
        y2 = self.osg_fno2(x)
        
        return p[..., 0:1] * y1 + p[..., 1:2] * y2
    
    def update_temperature(self, new_temperature):
        """
        update the temperature parameter 
        """
        self.current_temperature = torch.tensor(new_temperature, device=self.current_temperature.device)

    def get_gate_weights(self, dt_norm):
        """
        get the gate weights for given normalized time steps
        """
        with torch.no_grad():
            p = torch.nn.Softmax(dim=-1)(
                self.gate[1](self.activation(self.gate[0](dt_norm))) / self.current_temperature
            )
        return p.cpu().numpy()


class dual_osg_fno2d_with_film(osg_fno2d_with_film):
    """
    Dual-OSG_FNO2D with FiLM built upon two OSG-FNO2D-FiLM networks and a gating network
    """
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        
        self.osg_fno_film1 = osg_fno2d_with_film(vmin, vmax, tmin, tmax, config, multiscale)
        self.osg_fno_film2 = osg_fno2d_with_film(vmin, vmax, tmin, tmax, config, multiscale)
        self.gate    = torch.nn.ModuleList()
        
        self.gate.append(torch.nn.Linear(1, self.hid_dim))
        self.gate.append(torch.nn.Linear(self.hid_dim, 2))
        
        self.temperature = 1.0
        self.register_buffer('current_temperature', torch.tensor(1.0))
        
        self.name = "dual_osg_fno2d_with_film"
                    
    def forward(self, x):
        dt_norm = x[..., -1:]
        
        p = torch.nn.Softmax(dim=-1)(
            self.gate[1](self.activation(self.gate[0](dt_norm))) / self.current_temperature
        )
        
        y1 = self.osg_fno_film1(x)
        y2 = self.osg_fno_film2(x)
        
        return p[..., 0:1] * y1 + p[..., 1:2] * y2
    
    def update_temperature(self, new_temperature):
        self.current_temperature = torch.tensor(new_temperature, device=self.current_temperature.device)

    def get_gate_weights(self, dt_norm):

        with torch.no_grad():
            p = torch.nn.Softmax(dim=-1)(
                self.gate[1](self.activation(self.gate[0](dt_norm))) / self.current_temperature
            )
        return p.cpu().numpy()
    
class ResidualBlock(nn):
    def __init__(self, in_features, out_features, activation=torch.nn.GELU):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.activation = activation()
        self.linear2 = torch.nn.Linear(out_features, out_features)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = out + x  
        out = self.activation(out)
        return out
    

class SpectralConv1d(nn):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP1d(nn):
    def __init__(self, in_channels, out_channels, mid_channels, activation):
        super(MLP1d, self).__init__()
        self.mlp1 = torch.nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv1d(mid_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x

class GlobalLocalFNOBlock1d(nn):
    """Lag-ready global-local FNO block for sharp-front Burgers tests."""

    def __init__(self, channels, modes, activation, kernel_size=5, pool_factor=2, layer_scale=1e-3):
        super().__init__()
        self.activation = activation
        self.pool_factor = max(1, int(pool_factor))
        self.post_activation = True
        self.coupling_mode = "raw"
        self.coupling_scale = 1.0
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.spectral_mlp = MLP1d(channels, channels, channels, activation)
        self.spectral_w = torch.nn.Conv1d(channels, channels, 1)
        padding = int(kernel_size) // 2
        self.local_conv1 = torch.nn.Conv1d(channels, channels, int(kernel_size), padding=padding, groups=channels)
        self.local_pw1 = torch.nn.Conv1d(channels, channels, 1)
        self.local_conv2 = torch.nn.Conv1d(channels, channels, int(kernel_size), padding=padding, groups=channels)
        self.local_pw2 = torch.nn.Conv1d(channels, channels, 1)
        self.local_fuse = torch.nn.Conv1d(2 * channels, channels, 1)
        self.global_gate = torch.nn.Conv1d(channels, channels, 1)
        self.local_gate = torch.nn.Conv1d(channels, channels, 1)
        self.mix = torch.nn.Conv1d(4 * channels, channels, 1)
        self.layer_scale = torch.nn.Parameter(torch.full((1, channels, 1), float(layer_scale)))

    def _local_branch(self, x):
        if self.pool_factor > 1 and x.shape[-1] >= self.pool_factor:
            coarse = torch.nn.functional.avg_pool1d(x, kernel_size=self.pool_factor, stride=self.pool_factor)
        else:
            coarse = x
        coarse = self.local_conv1(coarse)
        coarse = self.local_pw1(coarse)
        coarse = self.activation(coarse)
        coarse = self.local_conv2(coarse)
        coarse = self.local_pw2(coarse)
        if coarse.shape[-1] != x.shape[-1]:
            coarse = torch.nn.functional.interpolate(coarse, size=x.shape[-1], mode="linear", align_corners=False)
        return self.activation(self.local_fuse(torch.cat((x, coarse), dim=1)))

    def forward(self, x, global_film=None, local_film=None):
        g = self.spectral_mlp(self.spectral(x)) + self.spectral_w(x)
        g = self.activation(g)
        l = self._local_branch(x)
        if global_film is not None:
            gamma, beta = global_film
            g = gamma * g + beta
        if local_film is not None:
            gamma, beta = local_film
            l = gamma * l + beta
        cg = self.global_gate(g)
        cl = self.local_gate(l)
        coupling_mode = getattr(self, "coupling_mode", "raw")
        if coupling_mode == "tanh":
            cg = torch.tanh(cg)
            cl = torch.tanh(cl)
        c = getattr(self, "coupling_scale", 1.0) * cg * cl
        update = self.mix(torch.cat((x, g, l, c), dim=1))
        out = x + self.layer_scale * update
        if getattr(self, "post_activation", False):
            out = self.activation(out)
        return out


class osg_fno1d(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno1d, self).__init__()

        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim = config["problem_dim"] + 1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes"]
        self.nblocks = config["depth"]
        self.hid_dim = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.w = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv1d(self.hid_dim, self.hid_dim, self.modes1))
            self.mlp.append(MLP1d(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv1d(self.hid_dim, self.hid_dim, 1))
        
        self.de = MLP1d(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def forward(self, x):
        """
        Forward pass for FNO1D
        
        Args:
            x: input tensor of shape (batch_size, L, input_dim) 
               where input_dim = problem_dim + 1 = 2 (u + dt)
        """
        x0 = x[..., :-1]
        

        dt = x[..., -1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt
        
        x = self.en(x)  
        x = x.permute(0, 2, 1)  
        
        for i in range(self.nblocks):
            x1 = self.conv[i](x) 
            x1 = self.mlp[i](x1)  
            x2 = self.w[i](x)     
            x = x1 + x2          
            x = self.activation(x) 
        
        x = self.de(x)  
        x = x.permute(0, 2, 1)  

        if getattr(self, "conserve_mean", False):
            x = x - x.mean(dim=1, keepdim=True)
        
        return x0 + x * dt
        
    def predict(self, x, dt, device):
        """
        Predict trajectories given initial conditions

        Args:
            x: unnormalized initial conditions, numpy array (N, L, D)
            dt: time steps, numpy array (N, steps)
            device: computation device

        Returns:
            y: unnormalized predictions, numpy array (N, L, D, steps+1)
        """
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        # ---- 强制用 CPU 版 vmin/vmax 做归一化，避免 checkpoint/map_location 造成设备混用 ----
        vmin = self.vmin.detach().cpu()
        vmax = self.vmax.detach().cpu()

        # ---- dt: CPU -> normalize -> device ----
        dt = torch.from_numpy(dt).float()
        dt = torch.unsqueeze(dt, 1)                  # (N, 1, steps)
        dt = torch.tile(dt, [1, x.shape[1], 1])     # (N, L, steps)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        dt = dt.to(device)

        # ---- x: CPU 上归一化，再搬到 device ----
        x = torch.from_numpy(x).float()
        x = 2 * (x - 0.5 * (vmax[..., 0] + vmin[..., 0])) / (
            vmax[..., 0] - vmin[..., 0]
        )
        x = x.to(device)

        y = torch.unsqueeze(x.clone(), -1)  # (N, L, D, 1)

        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[..., -1], dt[..., t:t+1]), dim=-1)  # (N, L, D+1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        # ---- 回 CPU 后再反归一化 ----
        y = y.cpu()
        y = y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)

        return y.numpy()
    


class vt_fno1d(osg_fno1d):
    """Variable-time FNO baseline without OSG outer-increment structure."""

    def forward(self, x):
        x = self.en(x)
        x = x.permute(0, 2, 1)
        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = self.activation(x1 + x2)
        x = self.de(x)
        return x.permute(0, 2, 1)

class osg_fno1d_with_film(nn):
    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(osg_fno1d_with_film, self).__init__()

        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim = config["problem_dim"]
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes"]
        self.nblocks = config["depth"]
        self.hid_dim = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))
        
        self.film_dim = config["width"]
        self.time_encoder = self._build_time_encoder()
        
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        
        self.conv = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.w = torch.nn.ModuleList()
        
        for i in range(self.nblocks):
            self.conv.append(SpectralConv1d(self.hid_dim, self.hid_dim, self.modes1))
            self.mlp.append(MLP1d(self.hid_dim, self.hid_dim, self.hid_dim, self.activation))
            self.w.append(torch.nn.Conv1d(self.hid_dim, self.hid_dim, 1))
            
        self.de = MLP1d(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def _build_time_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim),
            torch.nn.GELU(),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            torch.nn.Linear(self.film_dim, 2 * self.nblocks * self.hid_dim)
        )
        
    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        
        batch_size = x.shape[0]
        
        dt_scalar = dt_norm.mean(dim=[1])


        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)
        gammas = film_params[:, :, 0, :]  
        betas = film_params[:, :, 1, :]  
        
        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt
        
        x = self.en(x0)  
        x = x.permute(0, 2, 1)  
        

        for i in range(self.nblocks):
            x1 = self.conv[i](x)  
            x1 = self.mlp[i](x1)  
            x2 = self.w[i](x)    
            x = x1 + x2          
            
            gamma = gammas[:, i, :]  
            beta = betas[:, i, :]   
            
            gamma = gamma.view(batch_size, self.hid_dim, 1)  
            beta = beta.view(batch_size, self.hid_dim, 1)   
            
            x = gamma * x + beta  
            
            x = self.activation(x)  
        
        x = self.de(x) 
        x = x.permute(0, 2, 1) 

        if getattr(self, "conserve_mean", False):
            x = x - x.mean(dim=1, keepdim=True)
        
        return x0 + x * dt
    
    def predict(self, x, dt, device):
        """
        Predict trajectories given initial conditions

        Args:
            x: unnormalized initial conditions, numpy array (N, L, D)
            dt: time steps, numpy array (N, steps)
            device: computation device

        Returns:
            y: unnormalized predictions, numpy array (N, L, D, steps+1)
        """
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        # ---- 强制用 CPU 版 vmin/vmax 做归一化，避免 checkpoint/map_location 造成设备混用 ----
        vmin = self.vmin.detach().cpu()
        vmax = self.vmax.detach().cpu()

        # ---- dt: CPU -> normalize -> device ----
        dt = torch.from_numpy(dt).float()
        dt = torch.unsqueeze(dt, 1)                  # (N, 1, steps)
        dt = torch.tile(dt, [1, x.shape[1], 1])     # (N, L, steps)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        dt = dt.to(device)

        # ---- x: CPU 上归一化，再搬到 device ----
        x = torch.from_numpy(x).float()
        x = 2 * (x - 0.5 * (vmax[..., 0] + vmin[..., 0])) / (
            vmax[..., 0] - vmin[..., 0]
        )
        x = x.to(device)

        y = torch.unsqueeze(x.clone(), -1)  # (N, L, D, 1)

        self.eval()
        with torch.no_grad():
            for t in range(steps):
                xx = torch.cat((y[..., -1], dt[..., t:t+1]), dim=-1)  # (N, L, D+1)
                pred = self.forward(xx)
                y = torch.cat((y, torch.unsqueeze(pred, dim=-1)), dim=-1)

        # ---- 回 CPU 后再反归一化 ----
        y = y.cpu()
        y = y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)

        return y.numpy()


class vt_fno1d_with_film(osg_fno1d_with_film):
    """Time-conditioned FiLM-FNO baseline with direct next-state prediction.

    The lag modulates hidden channels through FiLM, but the model is not an
    operator semigroup and does not apply the outer-increment update.
    """

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        batch_size = x.shape[0]

        dt_scalar = dt_norm.mean(dim=[1])
        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)
        gammas = film_params[:, :, 0, :]
        betas = film_params[:, :, 1, :]

        x = self.en(x0)
        x = x.permute(0, 2, 1)

        for i in range(self.nblocks):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2

            gamma = gammas[:, i, :].view(batch_size, self.hid_dim, 1)
            beta = betas[:, i, :].view(batch_size, self.hid_dim, 1)

            x = gamma * x + beta
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 1)
        return x


class gl_osg_fno1d(nn):
    """Global-local direct-lag OSG-FNO for low-risk sharp-front diagnostics."""

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(gl_osg_fno1d, self).__init__()

        self.vmin = torch.from_numpy(vmin)
        self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin
        self.tmax = tmax
        self.input_dim = config["problem_dim"] + 1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes"]
        self.nblocks = config["depth"]
        self.hid_dim = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))

        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        self.blocks = torch.nn.ModuleList([
            GlobalLocalFNOBlock1d(
                self.hid_dim,
                self.modes1,
                self.activation,
                kernel_size=config.get("local_kernel_size", 5),
                pool_factor=config.get("local_pool_factor", 2),
                layer_scale=config.get("gl_layer_scale", 1e-3),
            )
            for _ in range(self.nblocks)
        ])
        self.de = MLP1d(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def forward(self, x):
        x0 = x[..., :-1]
        dt = x[..., -1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt

        z = self.en(x)
        z = z.permute(0, 2, 1)
        for block in self.blocks:
            z = block(z)
        inc = self.de(z).permute(0, 2, 1)
        if getattr(self, "conserve_mean", False):
            inc = inc - inc.mean(dim=1, keepdim=True)
        return x0 + inc * dt

    predict = osg_fno1d.predict


class gl_osg_fno1d_with_film(gl_osg_fno1d):
    """Global-local FiLM-OSG-FNO with branchwise lag modulation."""

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        self.input_dim = config["problem_dim"]
        self.film_dim = config["width"]
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        self.gl_film_mode = config.get("gl_film_mode", "branchwise")
        self.gl_local_film_mode = config.get("gl_local_film_mode", "affine")
        valid_modes = {"branchwise", "global_only"}
        valid_local_modes = {"affine", "gamma"}
        if self.gl_film_mode not in valid_modes:
            raise ValueError(f"Unknown gl_film_mode={self.gl_film_mode}. Expected one of {sorted(valid_modes)}.")
        if self.gl_local_film_mode not in valid_local_modes:
            raise ValueError(
                f"Unknown gl_local_film_mode={self.gl_local_film_mode}. Expected one of {sorted(valid_local_modes)}."
            )
        self.time_encoder = self._build_time_encoder()

    def _build_time_encoder(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim),
            torch.nn.GELU(),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            torch.nn.Linear(self.film_dim, 4 * self.nblocks * self.hid_dim),
        )
        last = net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        return net

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        batch_size = x.shape[0]
        dt_scalar = dt_norm.mean(dim=1)

        params = self.time_encoder(dt_scalar).view(batch_size, self.nblocks, 4, self.hid_dim)
        gamma_g = 1.0 + 0.1 * params[:, :, 0, :]
        beta_g = 0.1 * params[:, :, 1, :]
        gamma_l = 1.0 + 0.1 * params[:, :, 2, :]
        beta_l = 0.1 * params[:, :, 3, :]

        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt

        z = self.en(x0)
        z = z.permute(0, 2, 1)
        for i, block in enumerate(self.blocks):
            gfilm = (
                gamma_g[:, i, :].view(batch_size, self.hid_dim, 1),
                beta_g[:, i, :].view(batch_size, self.hid_dim, 1),
            )
            gl_film_mode = getattr(self, "gl_film_mode", "branchwise")
            gl_local_film_mode = getattr(self, "gl_local_film_mode", "branchwise")
            if gl_film_mode == "global_only":
                lfilm = None
            elif gl_local_film_mode == "gamma":
                lfilm = (
                    gamma_l[:, i, :].view(batch_size, self.hid_dim, 1),
                    0.0,
                )
            else:
                lfilm = (
                    gamma_l[:, i, :].view(batch_size, self.hid_dim, 1),
                    beta_l[:, i, :].view(batch_size, self.hid_dim, 1),
                )
            z = block(z, global_film=gfilm, local_film=lfilm)

        inc = self.de(z).permute(0, 2, 1)
        if getattr(self, "conserve_mean", False):
            inc = inc - inc.mean(dim=1, keepdim=True)
        return x0 + inc * dt


class gl_osg_fno2d(nn):
    """Global-local direct-lag OSG-FNO2D."""

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super(gl_osg_fno2d, self).__init__()
        self.vmin = torch.from_numpy(vmin); self.vmax = torch.from_numpy(vmax)
        self.tmin = tmin; self.tmax = tmax
        self.input_dim = config["problem_dim"] + 1
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.modes1 = config["modes1"]; self.modes2 = config["modes2"]
        self.nblocks = config["depth"]; self.hid_dim = config["width"]
        self.multiscale = multiscale
        self.conserve_mean = bool(config.get("conserve_mean", False))
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        self.blocks = torch.nn.ModuleList([
            GlobalLocalFNOBlock2d(self.hid_dim, self.modes1, self.modes2, self.activation,
                                  kernel_size=config.get("local_kernel_size", 3),
                                  pool_factor=config.get("local_pool_factor", 2),
                                  layer_scale=config.get("gl_layer_scale", 1e-3),
                                  post_activation=config.get("gl_post_activation", True),
                                  local_padding_mode=config.get("local_padding_mode", "zeros"),
                                  coupling_mode=config.get("gl_coupling_mode", "raw"),
                                  coupling_scale=config.get("gl_coupling_scale", 1.0),
                                  local_scale=config.get("gl_local_scale", 1.0))
            for _ in range(self.nblocks)
        ])
        self.de = MLP(self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation)

    def forward(self, x):
        x0 = x[..., :-1]
        dt = x[..., -1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt
        z = self.en(x).permute(0, 3, 1, 2)
        for block in self.blocks:
            z = block(z)
        inc = self.de(z).permute(0, 2, 3, 1)
        if getattr(self, "conserve_mean", False):
            inc = inc - inc.mean(dim=(1, 2), keepdim=True)
        return x0 + inc * dt

    predict = osg_fno2d.predict


class gl_osg_fno2d_with_film(gl_osg_fno2d):
    """Global-local FiLM-OSG-FNO2D with branchwise or global-only FiLM."""

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        self.input_dim = config["problem_dim"]
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        self.film_dim = config["width"]
        self.gl_film_mode = config.get("gl_film_mode", "global_only")
        self.gl_local_film_mode = config.get("gl_local_film_mode", "affine")
        valid_modes = {"branchwise", "global_only"}
        valid_local_modes = {"affine", "gamma"}
        if self.gl_film_mode not in valid_modes:
            raise ValueError(f"Unknown gl_film_mode={self.gl_film_mode}. Expected one of {sorted(valid_modes)}.")
        if self.gl_local_film_mode not in valid_local_modes:
            raise ValueError(
                f"Unknown gl_local_film_mode={self.gl_local_film_mode}. Expected one of {sorted(valid_local_modes)}."
            )
        self.time_encoder = self._build_time_encoder()

    def _build_time_encoder(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim), torch.nn.GELU(),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            ResidualBlock(self.film_dim, self.film_dim),
            torch.nn.Linear(self.film_dim, 4 * self.nblocks * self.hid_dim),
        )
        torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
        return net

    def forward(self, x):
        x0 = x[..., :-1]
        dt_norm = x[..., -1:]
        batch_size = x.shape[0]
        dt_scalar = dt_norm.mean(dim=(1, 2))
        params = self.time_encoder(dt_scalar).view(batch_size, self.nblocks, 4, self.hid_dim)
        gamma_g = 1.0 + 0.1 * params[:, :, 0, :]; beta_g = 0.1 * params[:, :, 1, :]
        gamma_l = 1.0 + 0.1 * params[:, :, 2, :]; beta_l = 0.1 * params[:, :, 3, :]
        dt = dt_norm * 0.5 * (self.tmax - self.tmin) + 0.5 * (self.tmax + self.tmin)
        if self.multiscale:
            dt = 10 ** dt
        z = self.en(x0).permute(0, 3, 1, 2)
        for i, block in enumerate(self.blocks):
            gfilm = (gamma_g[:, i, :].view(batch_size, self.hid_dim, 1, 1), beta_g[:, i, :].view(batch_size, self.hid_dim, 1, 1))
            gl_film_mode = getattr(self, "gl_film_mode", "branchwise")
            gl_local_film_mode = getattr(self, "gl_local_film_mode", "branchwise")
            if gl_film_mode == "global_only":
                lfilm = None
            elif gl_local_film_mode == "gamma":
                lfilm = (gamma_l[:, i, :].view(batch_size, self.hid_dim, 1, 1), 0.0)
            else:
                lfilm = (gamma_l[:, i, :].view(batch_size, self.hid_dim, 1, 1), beta_l[:, i, :].view(batch_size, self.hid_dim, 1, 1))
            z = block(z, global_film=gfilm, local_film=lfilm)
        inc = self.de(z).permute(0, 2, 3, 1)
        if getattr(self, "conserve_mean", False):
            inc = inc - inc.mean(dim=(1, 2), keepdim=True)
        return x0 + inc * dt

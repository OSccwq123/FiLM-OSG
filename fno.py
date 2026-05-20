""" This is adapted from the implementation of Fourier neural operator. Reference to https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d_time.py"""

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
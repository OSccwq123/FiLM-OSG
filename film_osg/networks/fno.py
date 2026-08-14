"""Fourier neural operator backbones used in the FiLM-OSG experiments.

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


class SpectralConv2d(nn):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = torch.nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

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
        self.input_dim = config["problem_dim"] + 1
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
        self.de = MLP(
            self.hid_dim, self.output_dim, self.hid_dim * 4, self.activation
        )

    def forward(self, x):
        
        x0 = x[..., :-1]
        dt = x[..., -1:] * 0.5 * (self.tmax - self.tmin) + 0.5 * (
            self.tmax + self.tmin
        )
        if self.multiscale:
            dt = 10 ** dt
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
        if self.conserve_mean:
            x = x - x.mean(dim=(1, 2), keepdim=True)
        return x0 + x * dt
        
    def predict(self, x, dt, device):
        self.to(device)

        vmin = self.vmin.to(device)
        vmax = self.vmax.to(device)

        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        dt = torch.from_numpy(dt).float().to(device)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.tile(dt, [1, x.shape[1], x.shape[2], 1])
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)

        x = torch.from_numpy(x).float().to(device)
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
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
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

        dt_scalar = dt_norm.mean(dim=[1, 2])

        film_params = self.time_encoder(dt_scalar)
        film_params = film_params.view(batch_size, self.nblocks, 2, self.hid_dim)

        raw_gammas = film_params[:, :, 0, :]
        raw_betas = film_params[:, :, 1, :]

        gammas = 1.0 + 0.1 * raw_gammas
        betas = 0.1 * raw_betas

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
            beta = betas[:, i, :].view(batch_size, self.hid_dim, 1, 1)

            x = gamma * x + beta
            x = self.activation(x)

        x = self.de(x)
        x = x.permute(0, 2, 3, 1)
        if self.conserve_mean:
            x = x - x.mean(dim=(1, 2), keepdim=True)

        return x0 + x * dt
    
    def predict(self, x, dt, device):
        self.to(device)

        vmin = self.vmin.to(device)
        vmax = self.vmax.to(device)

        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        dt = torch.from_numpy(dt).float().to(device)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.unsqueeze(dt, 1)
        dt = torch.tile(dt, [1, x.shape[1], x.shape[2], 1])
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)

        x = torch.from_numpy(x).float().to(device)
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
    """Variable-time FiLM-FNO with direct next-state prediction."""

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


class ResidualBlock(nn):
    def __init__(self, features, activation=torch.nn.GELU):
        super().__init__()
        self.linear1 = torch.nn.Linear(features, features)
        self.activation = activation()
        
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
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, dtype=torch.cfloat
            )
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
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
    """One-dimensional FNO block with spectral and local branches."""

    def __init__(self, channels, modes, activation, kernel_size=5, pool_factor=2, layer_scale=1e-3):
        super().__init__()
        self.activation = activation
        self.pool_factor = max(1, int(pool_factor))
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.spectral_mlp = MLP1d(channels, channels, channels, activation)
        self.spectral_w = torch.nn.Conv1d(channels, channels, 1)
        padding = int(kernel_size) // 2
        self.local_conv1 = torch.nn.Conv1d(
            channels, channels, int(kernel_size), padding=padding, groups=channels
        )
        self.local_pw1 = torch.nn.Conv1d(channels, channels, 1)
        self.local_conv2 = torch.nn.Conv1d(
            channels, channels, int(kernel_size), padding=padding, groups=channels
        )
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
            coarse = torch.nn.functional.interpolate(
                coarse, size=x.shape[-1], mode="linear", align_corners=False
            )
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
        c = self.global_gate(g) * self.local_gate(l)
        update = self.mix(torch.cat((x, g, l, c), dim=1))
        out = x + self.layer_scale * update
        return self.activation(out)


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
        """Evaluate the one-dimensional OSG-FNO update."""
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

        if self.conserve_mean:
            x = x - x.mean(dim=1, keepdim=True)
        
        return x0 + x * dt
        
    def predict(self, x, dt, device):
        """Roll out unnormalized states over the supplied time intervals."""
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        # Normalize on CPU so checkpoints loaded with map_location remain valid.
        vmin = self.vmin.detach().cpu()
        vmax = self.vmax.detach().cpu()

        dt = torch.from_numpy(dt).float()
        dt = torch.unsqueeze(dt, 1)                  # (N, 1, steps)
        dt = torch.tile(dt, [1, x.shape[1], 1])     # (N, L, steps)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        dt = dt.to(device)

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
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
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

        if self.conserve_mean:
            x = x - x.mean(dim=1, keepdim=True)
        
        return x0 + x * dt
    
    def predict(self, x, dt, device):
        """Roll out unnormalized states over the supplied time intervals."""
        self.to(device)
        assert x.shape[-1] == self.output_dim
        steps = dt.shape[1]

        # Normalize on CPU so checkpoints loaded with map_location remain valid.
        vmin = self.vmin.detach().cpu()
        vmax = self.vmax.detach().cpu()

        dt = torch.from_numpy(dt).float()
        dt = torch.unsqueeze(dt, 1)                  # (N, 1, steps)
        dt = torch.tile(dt, [1, x.shape[1], 1])     # (N, L, steps)
        if self.multiscale:
            dt = torch.log10(dt)
        dt = 2 * (dt - 0.5 * (self.tmax + self.tmin)) / (self.tmax - self.tmin)
        dt = dt.to(device)

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

        y = y.cpu()
        y = y * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)

        return y.numpy()


class gl_osg_fno1d(nn):
    """Global--local input-concatenation OSG-FNO for one-dimensional data."""

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
        if self.conserve_mean:
            inc = inc - inc.mean(dim=1, keepdim=True)
        return x0 + inc * dt

    predict = osg_fno1d.predict


class gl_osg_fno1d_with_film(gl_osg_fno1d):
    """Global--local FiLM-OSG-FNO with branchwise conditioning on Delta."""

    def __init__(self, vmin, vmax, tmin, tmax, config, multiscale=True):
        super().__init__(vmin, vmax, tmin, tmax, config, multiscale)
        self.input_dim = config["problem_dim"]
        self.film_dim = config["width"]
        self.en = torch.nn.Linear(self.input_dim, self.hid_dim)
        self.gl_film_mode = config.get("gl_film_mode", "branchwise")
        valid_modes = {"branchwise", "global_only"}
        if self.gl_film_mode not in valid_modes:
            raise ValueError(f"Unknown gl_film_mode={self.gl_film_mode}. Expected one of {sorted(valid_modes)}.")
        self.time_encoder = self._build_time_encoder()

    def _build_time_encoder(self):
        net = torch.nn.Sequential(
            torch.nn.Linear(1, self.film_dim),
            torch.nn.GELU(),
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
            ResidualBlock(self.film_dim),
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
            if self.gl_film_mode == "global_only":
                lfilm = None
            else:
                lfilm = (
                    gamma_l[:, i, :].view(batch_size, self.hid_dim, 1),
                    beta_l[:, i, :].view(batch_size, self.hid_dim, 1),
                )
            z = block(z, global_film=gfilm, local_film=lfilm)

        inc = self.de(z).permute(0, 2, 1)
        if self.conserve_mean:
            inc = inc - inc.mean(dim=1, keepdim=True)
        return x0 + inc * dt

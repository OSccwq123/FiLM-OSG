"""PDE training with the OSG semigroup loss.

Portions of this module are adapted from the DUE project:
https://github.com/AI4Equations/due
DUE is distributed under the LGPL-2.1 license.
"""

from time import time
import torch

from .pde import PDE

class PDE_osg(PDE):
    """Train a PDE model with supervised and semigroup losses."""
    def __init__(self, trainX, trainY, network, config):
        super(PDE_osg, self).__init__(trainX, trainY, network, config)
        
        self.sg_pairing  = config["sg_pairing"] # non-negative interger
        self.sg_weight   = config["sg_weight"]
        self.hf_weight = float(config.get("hf_weight", 0.0))
        self.hf_sg_weight = float(config.get("hf_sg_weight", 0.0))
        self.hf_warmup_frac = float(config.get("hf_warmup_frac", 0.0))
        self.hf_band_frac = float(config.get("hf_band_frac", 1.0 / 3.0))
        self.hf_power = float(config.get("hf_power", 2.0))
        self._prepare_auxiliary_samples()
        
    def _prepare_auxiliary_samples(self):
        """Generate the fixed auxiliary states and lag pairs used in training."""
        if self.sg_pairing == 0:
            return

        tmin = self.mynet.tmin
        tmax = self.mynet.tmax
        self.u0_rand = 2.0 * torch.rand(
            self.trainX.shape[0],
            self.sg_pairing,
            *self.trainX.shape[1:-1],
            self.mynet.output_dim,
            dtype=self.trainX.dtype,
        ) - 1.0
        self.dt_rand = 2.0 * torch.rand(
            self.trainX.shape[0], self.sg_pairing, 2, dtype=self.trainX.dtype
        ) - 1.0

        t1_rand = self.dt_rand[..., :1] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
        t2_rand = self.dt_rand[..., 1:] * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
        if self.mynet.multiscale:
            t1_rand = 10 ** t1_rand
            t2_rand = 10 ** t2_rand

        t12_rand = t1_rand + t2_rand
        if self.mynet.multiscale:
            t12_rand = torch.log10(t12_rand)
        t12_rand = 2 * (t12_rand - 0.5 * (tmax + tmin)) / (tmax - tmin)
        self.dt_rand = torch.concat((self.dt_rand, t12_rand), dim=-1)

        for _ in range(len(self.u0_rand.shape[2:-1])):
            self.dt_rand = torch.unsqueeze(self.dt_rand, 2)
        self.dt_rand = torch.tile(
            self.dt_rand, [1, 1, *self.u0_rand.shape[2:-1], 1]
        )
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.trainX, self.trainY, self.u0_rand, self.dt_rand
            ),
            batch_size=self.bsize,
            shuffle=True,
        )
        
    def _hf_warmup(self, ep):
        if self.hf_warmup_frac <= 0.0:
            return 1.0
        warmup_epochs = max(1, int(self.nepochs * self.hf_warmup_frac))
        return min(1.0, float(ep + 1) / float(warmup_epochs))

    def _high_frequency_loss(self, pred, target):
        err = pred - target
        spatial_dims = tuple(range(1, err.ndim - 1))
        if not spatial_dims:
            return err.new_tensor(0.0)

        err_ch = err.movedim(-1, 1)
        if len(spatial_dims) == 1:
            coeff = torch.fft.rfft(err_ch, dim=-1)
            nfreq = coeff.shape[-1]
            start = max(1, int((1.0 - self.hf_band_frac) * nfreq))
            if start >= nfreq:
                start = max(1, nfreq - 1)
            k = torch.linspace(0.0, 1.0, nfreq, device=err.device, dtype=err.real.dtype)
            weights = k.clamp_min(1.0 / max(1, nfreq - 1)).pow(self.hf_power)
            return (coeff[..., start:].abs().pow(2) * weights[start:]).mean()

        if len(spatial_dims) == 2:
            coeff = torch.fft.rfft2(err_ch, dim=(-2, -1))
            nx = err_ch.shape[-2]
            ny = err_ch.shape[-1]
            kx = torch.fft.fftfreq(nx, device=err.device, dtype=err.real.dtype).abs().view(1, 1, nx, 1)
            ky = torch.fft.rfftfreq(ny, device=err.device, dtype=err.real.dtype).abs().view(1, 1, 1, ny // 2 + 1)
            radius = torch.sqrt(kx * kx + ky * ky)
            rmax = radius.max().clamp_min(torch.finfo(err.real.dtype).eps)
            mask = radius >= (1.0 - self.hf_band_frac) * rmax
            weights = (radius / rmax).clamp_min(torch.finfo(err.real.dtype).eps).pow(self.hf_power)
            return (coeff.abs().pow(2) * weights * mask).sum() / mask.sum().clamp_min(1.0) / coeff.shape[0] / coeff.shape[1]

        return err.new_tensor(0.0)

    def train(self):
        self.summary()
        self.hist   = torch.zeros(self.nepochs,1)
        start = time()
        
        min_loss = 10000000000.0
        for ep in range(self.nepochs):
            self.mynet.train()
            
            train_step = 0
            if self.sg_pairing == 0:
                for xx, yy in self.train_loader:
                    
                    xx = xx.to(self.device)
                    yy = yy.to(self.device)
                    
                    pred = self.mynet(xx) #(batch_size, output_dim)
                        
                    data_loss = self.loss_func(yy, pred)
                    if self.hf_weight > 0.0:
                        data_loss = data_loss + self._hf_warmup(ep) * self.hf_weight * self._high_frequency_loss(pred, yy)

                    loss = data_loss
                    train_step += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler != None:
                        self.scheduler.step()
                        
            else:
                for xx, yy, uu, tt in self.train_loader:
                    
                    xx = xx.to(self.device)
                    yy = yy.to(self.device)
                    uu = uu.view(-1,*self.trainY.shape[1:]).to(self.device)
                    tt = tt.view(-1,*self.trainY.shape[1:-1], 3).to(self.device)
                    
                    pred    = self.mynet(xx) #(batch_size, output_dim)
                    pred01  = self.mynet(torch.cat((uu, tt[...,0:1]),dim=-1))
                    pred012 = self.mynet(torch.cat((pred01, tt[...,1:2]),dim=-1))
                    pred02  = self.mynet(torch.cat((uu, tt[...,1:2]),dim=-1))
                    pred021 = self.mynet(torch.cat((pred02, tt[...,0:1]),dim=-1))
                    pred2   = self.mynet(torch.cat((uu, tt[...,2:3]),dim=-1))
                        
                    data_loss = self.loss_func(yy, pred)
                    sg_loss = 0.5 * (self.loss_func(pred012, pred2) + self.loss_func(pred021, pred2))
                    warmup = self._hf_warmup(ep)
                    if self.hf_weight > 0.0:
                        data_loss = data_loss + warmup * self.hf_weight * self._high_frequency_loss(pred, yy)
                    if self.hf_sg_weight > 0.0:
                        sg_loss = sg_loss + warmup * self.hf_sg_weight * 0.5 * (
                            self._high_frequency_loss(pred012, pred2)
                            + self._high_frequency_loss(pred021, pred2)
                        )

                    loss = (data_loss + self.sg_weight * sg_loss) / (1.0 + self.sg_weight)
                    train_step += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler != None:
                        self.scheduler.step()
                
            train_step /= len(self.train_loader)
            if train_step < min_loss:
                torch.save(self.mynet, self.save_path+"/model")
                min_loss = train_step
            self.hist[ep,0] = train_step
            
            if (ep+1)%self.verbose ==0:
                end = time()
                print(f"Epoch {ep+1} --- Time: {end-start:.2f} seconds --- Training loss: {train_step}")
                start = end

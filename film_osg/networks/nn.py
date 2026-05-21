"""Base neural-network utilities for FiLM-OSG reproducibility experiments.

Portions of this module are adapted from the DUE project:
https://github.com/AI4Equations/due
DUE is distributed under the LGPL-2.1 license. Local changes here preserve the
DUE-style base API used by the reproduced OSG/FiLM-OSG models.
"""

import os

import torch

from ..utils import get_activation


class nn(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()

    def count_params(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

    def load_params(self, save_path):
        """Load the trained model for the NN."""
        return torch.load(save_path)

    def set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class pit_fixdt(nn):
    """Base class for Position-induced Transformers."""

    def __init__(self, mesh1, mesh2, device, config):
        super(pit_fixdt, self).__init__()

        self.device = device
        self.msh_qry = mesh1
        self.msh_ltt = mesh2.astype(mesh1.dtype)
        self.m_cross = self.pairwise_dist(self.msh_qry, self.msh_ltt, self.device)
        self.m_latent = self.pairwise_dist(self.msh_ltt, self.msh_ltt, self.device)
        self.npoints = self.msh_qry.shape[0]

        self.memory = config["memory"]
        self.input_dim = config["problem_dim"] * (self.memory + 1)
        self.output_dim = config["problem_dim"]
        self.activation = get_activation(config["activation"])
        self.hid_dim = config["width"]
        self.n_head = config["n_head"]
        self.n_blocks = config["depth"]
        self.en_local = config["locality_encoder"]
        self.de_local = config["locality_decoder"]
        self.set_seed(config["seed"])

    def get_mesh(self, inputs):
        mesh = torch.from_numpy(self.msh_qry)
        mesh = mesh.to(self.device)
        return torch.tile(torch.unsqueeze(mesh, dim=0), [inputs.shape[0], 1, 1])

    def predict(self, x, steps, device):
        xx = torch.from_numpy(x)
        xx = xx.to(device)

        yy = torch.zeros(*xx.shape[:-1], steps + self.memory + 1, device=device)
        yy[..., :self.memory + 1] = xx
        self.eval()
        with torch.no_grad():
            for t in range(steps):
                yy[..., self.memory + t + 1] = self.forward(yy[..., t:self.memory + t + 1])

        return yy.cpu().numpy()

    def pairwise_dist(self, mesh1, mesh2, device):
        try:
            mesh1 = torch.from_numpy(mesh1)
            mesh2 = torch.from_numpy(mesh2)
        except Exception:
            pass
        mesh1 = mesh1.to(device)
        mesh2 = mesh2.to(device)
        dist = torch.cdist(mesh1, mesh2, p=2)
        dist2 = dist**2
        dist2 = dist2 / torch.max(dist2)
        return dist2

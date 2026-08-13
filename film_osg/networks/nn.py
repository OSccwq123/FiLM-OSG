"""Base neural-network utilities for FiLM-OSG reproducibility experiments.

Portions of this module are adapted from the DUE project:
https://github.com/AI4Equations/due
DUE is distributed under the LGPL-2.1 license. Local changes here preserve the
DUE-style base API used by the reproduced OSG/FiLM-OSG models.
"""

import torch


class nn(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()

    def count_params(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

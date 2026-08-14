"""Base neural-network utilities for FiLM-OSG reproducibility experiments.

This file is derived from AI4Equations/DUE:
https://github.com/AI4Equations/due
It was modified for FiLM-OSG in 2026; individual change dates are recorded in
the Git history. This file is distributed under the GNU LGPL v2.1; see
THIRD_PARTY_LICENSES/DUE-LGPL-2.1.txt.
"""

import torch


class nn(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()

    def count_params(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

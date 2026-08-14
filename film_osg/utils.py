"""Training utilities used by the FiLM-OSG models.

This file is derived from AI4Equations/DUE:
https://github.com/AI4Equations/due
It was modified for FiLM-OSG in 2026; individual change dates are recorded in
the Git history. This file is distributed under the GNU LGPL v2.1; see
THIRD_PARTY_LICENSES/DUE-LGPL-2.1.txt.
"""

import torch
        
def rel_l1_norm(true, pred):
    
    bsize = true.shape[0]
    rel_error  = torch.norm(true.reshape(bsize,-1)-pred.reshape(bsize,-1), p=1, dim=1) / torch.norm(true.reshape(bsize,-1), p=1, dim=1)#(bsize,)
    return torch.mean(rel_error)

def rel_l2_norm(true, pred):
    
    bsize = true.shape[0]
    rel_error  = torch.norm(true.reshape(bsize,-1)-pred.reshape(bsize,-1), p=2, dim=1) / torch.norm(true.reshape(bsize,-1), p=2, dim=1)#(bsize,)
    return torch.mean(rel_error)
    
def rel_l2_norm_pde(true, pred):
    """
    true, pred: (N,L,D,T)
    """
    true = true.reshape(true.shape[0], -1, true.shape[-2], true.shape[-1])
    pred = pred.reshape(pred.shape[0], -1, pred.shape[-2], pred.shape[-1])
    rel_error  = torch.norm(true-pred, p=2, dim=1) / torch.norm(true, p=2, dim=1)#(N,D,T)
    return torch.mean(rel_error)

def rel_l1_norm_pde(true, pred):
    """
    true, pred: (N,L,D,T)
    """
    rel_error  = torch.norm(true-pred, p=1, dim=1) / torch.norm(true, p=1, dim=1)#(N,D,T)
    return torch.mean(rel_error)
    
def get_activation(name):

    if name in ['tanh', 'Tanh']:
        return torch.nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return torch.nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return torch.nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return torch.nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return torch.nn.Softplus()
    elif name in ['gelu', 'Gelu']:
        return torch.nn.functional.gelu
        
    else:
        raise ValueError(f'unknown or unsupported activation function: {name}')
        
def get_optimizer(name, model, lr):

    if name in ['adam', 'Adam', 'ADAM']:
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name in ['nadam', 'NAdam', 'NADAM']:
        return torch.optim.NAdam(model.parameters(), lr=lr)
    elif name in ['adamw', 'AdamW', 'ADAMW']:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif name in ['SGD', 'sgd', 'Sgd']:
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'unknown or unsupported optimizer: {name}')
        
def get_schedule(optimizer, name, epochs, batch_size, ntrain):

    if name in ['cyclic_cosine', 'Cyclic_cosine', 'Cyclic_Cosine']:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=(epochs//5)*(ntrain//batch_size))
    
    elif name in ['cosine', 'Cosine']:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (ntrain//batch_size))
    elif name in ['one_cycle', 'One_Cycle', 'OneCycle']:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=epochs * (ntrain//batch_size))
        
    elif name in ['none', 'None']:
        return None
        
    else:
        raise ValueError(f'unknown or unsupported learning schedule: {name}')
        
def get_loss(name):
    if name in ['mse', 'Mse', 'MSE']:
        return torch.nn.MSELoss(reduction="mean")
    elif name in ['mae', 'Mae', 'MAE']:
        return torch.nn.L1Loss(reduction="mean")
    elif name in ['rel_l2', 'Rel_l2', 'Rel_L2']:
        return rel_l2_norm
    elif name in ['rel_l2_pde', 'Rel_l2_pde', 'Rel_L2_pde']:
        return rel_l2_norm_pde
    elif name in ['rel_l1_pde', 'Rel_l1_pde', 'Rel_L1_pde']:
        return rel_l1_norm_pde
    elif name in ['rel_l1', 'Rel_l1', 'Rel_L1']:
        return rel_l1_norm
    else:
        raise ValueError(f'unknown or unsupported loss function: {name}')

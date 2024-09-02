import torch


def add_mask(x, ratio=0.1):
    ''' motion_2d: (N,T,14,2)
    '''
    N, T, J, C = x.shape
    mask = torch.rand(N, T, J, 1, dtype=x.dtype, device=x.device) > ratio
    mask_T = torch.rand(1, T, 1, 1, dtype=x.dtype, device=x.device) > ratio
    x = x * mask * mask_T
    return x

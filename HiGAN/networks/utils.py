import torch
from torch.nn import init
import torch.nn as nn


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.Embedding)):
            if init_type == 'N02':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type in ['glorot', 'xavier']:
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'ortho':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    if init_type in ['N02', 'glorot', 'xavier', 'kaiming', 'ortho']:
        # print('initialize network {} with {}'.format(net.__class__.__name__, init_type))
        net.apply(init_func)  # apply the initialization function <init_func>
    return net


def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from networks.utils import _len2mask


# Architectures for G
# Attention is passed in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64'):
    arch = {64: {'in_channels': [ch * item for item in [8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [4, 2, 1, 1]],
                 'upsample': [(2, 1), (2, 2), (2, 2), (2, 2)],
                 'resolution': [8, 16, 32, 64],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(2, 7)}}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, style_dim=32, embed_dim=120,
                 bottom_width=4, bottom_height=4, resolution=64,
                 G_attn='0', n_class=80,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1.e-05, SN_eps=1.e-08, input_nc=1,
                 embed_pad_idx=0, embed_max_norm=1.0
                 ):
        super(Generator, self).__init__()
        dim_z = style_dim
        # The initial width dimensions
        self.bottom_width = bottom_width
        # The initial height dimension
        self.bottom_height = bottom_height
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_class
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Architecture dict
        self.arch = G_arch(G_ch, G_attn)[resolution]

        self.text_embedding = nn.Embedding(self.n_classes, embed_dim,
                                           padding_idx=embed_pad_idx,
                                           max_norm=embed_max_norm)

        self.which_conv = functools.partial(layers.SNConv2d,
                                            kernel_size=3, padding=1,
                                            num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                            eps=SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                              eps=SN_eps)

        bn_linear = functools.partial(self.which_linear, bias=False)

        self.which_bn = functools.partial(layers.ccbn,
                                          which_linear=bn_linear,
                                          input_size=dim_z,
                                          norm_style='bn',
                                          eps=BN_eps)

        self.filter_linear = self.which_linear(embed_dim + dim_z,
                                               self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
        self.style_linear = self.which_linear(dim_z,
                                              dim_z * len(self.arch['in_channels']))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv1=self.which_conv,
                                           which_conv2=self.which_conv,
                                           which_bn=self.which_bn,
                                           activation=self.activation,
                                           upsample=(functools.partial(F.interpolate,
                                                                       scale_factor=self.arch['upsample'][index])
                                                     if index < len(self.arch['upsample']) else None))]]

            # If attention on this block, attach it to the end
            # print('index ', index, self.arch['resolution'][index])
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1]),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], input_nc))

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, y_lens):
        ys = self.style_linear(z).split(32, dim=1)

        # This is the change we made to the Big-GAN generator architecture.
        # The input goes into classes go into the first layer only.
        y = self.text_embedding(y).float().to(y.device)
        z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)
        h = self.filter_linear(z)

        # Reshape - when y is not a single class value but rather an array of classes, the reshape is needed to create
        # a separate vertical patch for each input.
        h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
        h = h.permute(0, 3, 2, 1)

        # Loop over blocks
        len_scale = 1
        x_lens = y_lens * self.bottom_width
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                if isinstance(block, layers.Attention):
                    h = block(h, x_lens=x_lens * len_scale)
                else:
                    h = block(h, y=ys[index])
            len_scale *= self.arch['upsample'][index][1]

        # Apply batchnorm-relu-conv-tanh at output
        output = torch.tanh(self.output_layer(h))

        # Mask blanks
        if not self.training:
            out_lens = y_lens * output.size(-2) // 2
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32).to(z.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)

        return output

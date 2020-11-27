from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.initializers import init_he_normal, init_gamma, init_beta, init_kaiming


# Changing bias from False to True
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), 
                 use_norm=True, norm='bn', stride=(1, 1), use_activation=True, 
                 activation='relu', bias=True):
        
        _bias = bias
        
#         if use_norm:
#             _bias = bias == "instance"
        
        super(ConvBlock, self).__init__()
        
        conv = nn.Conv2d(in_channels, out_channels, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         stride=stride,
                         bias=_bias)
        
        layers = [conv]
        if use_norm:
            if norm == 'bn':
                bn = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                bn = nn.InstanceNorm2d(out_channels)
            else:
                bn = None
            if bn is not None:
                layers.append(bn)
            
        if use_activation:
            if activation == 'relu':
                act = nn.ReLU()
            elif 'leaky_relu' in activation:
                slope = float(activation.split(':')[-1])
                act = nn.LeakyReLU(negative_slope=slope)
            elif activation == 'prelu':
                act = nn.PReLU()
            layers.append(act)
        
        self.block = nn.Sequential(*layers)
        self.block.apply(init_he_normal)
        self.block.apply(init_gamma)
        self.block.apply(init_beta)
    
    def forward(self, x):
        return self.block(x)
    
class DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), 
                 use_norm=True, norm='bn', nblocks=1, downsample=True, activation='relu'):
        
        super(DownBlock, self).__init__()
        
        modules = [ConvBlock(in_channels, out_channels, 
                            kernel_size=kernel_size, padding=padding, 
                            use_norm=use_norm, norm=norm, activation=activation)]
        for i in range(1, nblocks):
            modules.append(ConvBlock(out_channels, out_channels, 
                                     kernel_size=kernel_size, padding=padding, 
                                     use_norm=use_norm, norm=norm, activation=activation))
        
        self.module = nn.Sequential(*modules)
        
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        
    def forward(self, x):
        
        x = self.module(x)
        if self.downsample:
            pooled_x = self.pool(x)
            return x, pooled_x
        return x
        
# Changing boas to True
class DeconvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(0, 0), 
                 stride=2, use_norm=True, norm='bn', activation='relu'):
        
        super(DeconvBlock, self).__init__()
        
        conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            stride=stride,
            bias=True#(not use_norm or norm=='instance')
        )
        layers = [conv]
        if use_norm:
            if norm == 'bn':
                bn = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                bn = nn.InstanceNorm2d(out_channels)
            else:
                bn = None
            if bn is not None:
                layers.append(bn)
        if activation == 'relu':
            act = nn.ReLU()
        elif 'leaky_relu' in activation:
            slope = float(activation.split(':')[-1])
            act = nn.LeakyReLU(negative_slope=slope)
        elif activation == 'prelu':
            act = nn.PReLU()
        layers.append(act)
        
        self.block = nn.Sequential(*layers)
        self.block.apply(init_he_normal)
    
    def forward(self, x):
        return self.block(x)

    
class UpBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=2,
                 use_norm=True, norm='bn', nblocks=1, activation='relu'):
        
        super(UpBlock, self).__init__()
        
        self.up = DeconvBlock(
            in_channels, out_channels, 
            kernel_size=(2, 2), padding=(0, 0), 
            stride=stride,
            use_norm=use_norm, norm=norm, activation=activation
        )
        
        modules = []
        _in_channels = out_channels * 2
        for i in range(nblocks):
            modules.append(ConvBlock(_in_channels, out_channels, 
                                     kernel_size=kernel_size, padding=padding, 
                                     use_norm=use_norm, norm=norm, activation=activation))
            _in_channels = out_channels
        
        self.module = nn.Sequential(*modules)
        
    def forward(self, x, branch_x):
        
        x = self.up(x)
        # Concatenate
        x = torch.cat([x, branch_x], dim=1)
        
        return self.module(x)
    

# Changing bn to instance, bias to True
class GatingSignal(nn.Module):
    def __init__(self, in_size, out_size, use_bn=False):
        super(GatingSignal, self).__init__()
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, stride=1, kernel_size=1, padding=0, bias=True),
                nn.Batchnorm2d(out_size),
                # nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, stride=1, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True)
            )
        
    
    def forward(self, x):
        return self.conv(x)
    
    
    
class GridAttentionBlock(nn.Module):
    """
    Code credits: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/3031a7f9e94d725b36ba133c581107c923651f4e/models/layers/grid_attention_layer.py
    """
    
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 sub_sample_factor=(2,2), use_norm=True, norm="bn"):
        
        super(GridAttentionBlock, self).__init__()

        self.sub_sample_factor = sub_sample_factor
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.use_norm = use_norm
        self.norm = norm

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.upsample_mode = 'bilinear'
        
        # Output transform
        
        self.W = ConvBlock(self.in_channels, self.in_channels, kernel_size=1, padding=0, 
                           stride=1, use_norm=self.use_norm, norm=self.norm, 
                           use_activation=False)
        
        
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = ConvBlock(self.in_channels, self.inter_channels, \
                               kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_kernel_size, \
                               padding=0, bias=False, use_activation=False, use_norm=False)
        
        self.phi = ConvBlock(self.gating_channels, self.inter_channels, \
                             kernel_size=(1, 1), stride=(1, 1), \
                             padding=0, use_activation=False, use_norm=False, \
                             bias=True)
        
        self.psi = ConvBlock(self.inter_channels, 1, \
                             kernel_size=(1, 1), stride=(1, 1), \
                             padding=0, bias=True, use_activation=False, 
                             use_norm=False)
        

    def forward(self, x, g):
        '''
        :param x: (b, c, h, w)
        :param g: (b, g_d)
        :return:
        '''

        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))

        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
    

class MultiGridAttentionBlock(nn.Module):
    
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 sub_sample_factor=(2,2), use_norm=True, norm="bn", nclasses=3):
        
        super(MultiGridAttentionBlock, self).__init__()
        
        self.nclasses = nclasses
        
        blocks = []
        for i in range(nclasses):
            blocks.append(
                GridAttentionBlock(in_channels=in_channels, gating_channels=gating_channels, 
                                   inter_channels=inter_channels,
                                   sub_sample_factor=sub_sample_factor, use_norm=use_norm, 
                                   norm=norm)
            )
            
        self.blocks = nn.ModuleList(blocks)
        self.comb = ConvBlock(in_channels*nclasses, out_channels=in_channels, kernel_size=(1, 1), 
                              padding=(0, 0), 
                              use_norm=use_norm, norm=norm, 
                              stride=(1, 1), use_activation=True, 
                              activation='relu', bias=False)
        
    def forward(self, x, g):
        
        blocks, attns = [], []
        for c in range(self.nclasses):
            blk, attn = self.blocks[c](x, g)
            blocks.append(blk)
            attns.append(attn)
            
        return self.comb(torch.cat(blocks, dim=1)), attns
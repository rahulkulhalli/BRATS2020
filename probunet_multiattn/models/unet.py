import torch
import torch.nn as nn
from .modules import DownBranch, UpBranch
from .module_utils import ConvBlock

class UNet(nn.Module):
    
    def __init__(self, input_shape=(4, 240, 240), output_shape=(3, 240, 240), 
                 base_filters=16, depth=4, nblocks=1, activation='relu', 
                 norm='bn', visualize=False, nclasses=3):
        
        super(UNet, self).__init__()
        
        self.visualize = visualize
        
        self.down_branch = DownBranch(input_shape=input_shape, base_filters=base_filters, 
                                      depth=depth, nblocks=nblocks, activation=activation, 
                                      norm=norm)
        self.gating = ConvBlock(in_channels=self.down_branch.center_filters, out_channels=self.down_branch.center_filters, 
                                kernel_size=(3, 3), padding=(1, 1), 
                                use_norm=True, norm=norm, stride=(1, 1), use_activation=True, 
                                activation=activation, bias=False)
        
        self.up_branch = UpBranch(output_shape=output_shape, base_filters=base_filters, 
                                  depth=depth, nblocks=nblocks, activation=activation, 
                                  norm=norm, attention_channels=self.down_branch.attention_channels, 
                                  center_filters=self.down_branch.center_filters, visualize=visualize, nclasses=nclasses)
#         in_channels = self.up_branch.block1.block[0].out_channels
#         self.final_block = ConvBlock(in_channels=in_channels, out_channels=8, 
#                                 kernel_size=(1, 1), padding=(0, 0), 
#                                 use_norm=True, norm=norm, stride=(1, 1), use_activation=True, 
#                                 activation=activation, bias=False)
        
    def forward(self, x):
        
        x, branches = self.down_branch(x)
        
        gating = self.gating(x)
        
        if self.visualize:
            x, attn_blocks = self.up_branch(x, gating, branches)
        else:
            x = self.up_branch(x, gating, branches)
        
        # x = self.final_block(x)
        
        if self.visualize:
            return x, attn_blocks
        return x
    
    
class fcomb(nn.Module):
    
    def __init__(self, in_channels, base_filters=16, output_shape=(3, 224, 224), kernel_size=(1, 1), padding=(0, 0), 
                 nblocks=3, use_norm=True, activation='relu', norm='bn'):
        
        super(fcomb, self).__init__()
        
        self.output_shape = output_shape
        
        modules = []
        _in_channels = in_channels
        out_channels = base_filters
        for i in range(0, nblocks-1):
            modules.append(ConvBlock(_in_channels, out_channels, 
                                     kernel_size=kernel_size, padding=padding, 
                                     use_norm=False, norm='none', activation=activation))
            _in_channels = out_channels
            out_channels = max(8, out_channels//2)
        modules.append(
            ConvBlock(_in_channels, output_shape[0], 
                      kernel_size=(1, 1), padding=(0, 0), 
                      use_norm=False, norm='none', use_activation=False, activation=activation)
        )
        
        modules.append(nn.Sigmoid())
        
        self.module = nn.Sequential(*modules)
        
    def forward(self, unet_features, prior_samples, reduce='none'):
        
        N, C, G1, G2 = unet_features.shape
        NS, zdim = prior_samples.shape
        S = NS // N
        
        if reduce != 'none':
            assert S > 1
        
        if S > 1:
            x = torch.repeat_interleave(unet_features, S, dim=0)
        else: 
            x = unet_features
            
        # Broadcast
        samples = prior_samples.repeat(1, G1*G2).view(NS, G1, G2, zdim)
        samples = samples.permute(0, 3, 1, 2)
        
        # Concatenate
        x = torch.cat([x, samples], dim=1)
        
        out = self.module(x)
        if S > 1:
            out = out.view(N, S, self.output_shape[0], G1, G2)
        
        if reduce == 'none':
            return out
        elif reduce == 'sum':
            return out.sum(dim=1)
        elif reduce == 'mean':
            return out.mean(dim=1)
        
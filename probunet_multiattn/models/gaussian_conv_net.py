import torch
import torch.nn as nn
from .module_utils import DownBlock, MultiGridAttentionBlock
from ..utils.initializers import init_he_normal


class GaussianConvNet(nn.Module):
    
    def __init__(self, input_shape=(4, 240, 240), base_filters=16, depth=5, nblocks=1, zdim=6, 
                 activation='relu', norm='bn', attention_depths=[2, 3], visualize=False):
        
        super(GaussianConvNet, self).__init__()
        
        self.zdim = zdim
        self.depth = depth
        self.attention_depths = attention_depths
        
        self.visualize = visualize
        
        blocks = []
        attention_channels = []
        in_channels = input_shape[0]
        out_channels = base_filters
        for d in range(depth):
            blocks.append(
                DownBlock(
                    in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=(3, 3), padding=(1, 1), 
                    use_norm=True, norm=norm, nblocks=nblocks, downsample=True, activation=activation
                )
            )
            
            if d in attention_depths:
                attention_channels.append(out_channels)
                
            in_channels = out_channels
            out_channels *= 2
        self.blocks = nn.ModuleList(blocks)
        
        attention_blocks = []
        final_linear_in_filters = 0
        for d in range(len(attention_depths)):
            attention_blocks.append(
                MultiGridAttentionBlock(in_channels=attention_channels[d], gating_channels=out_channels, 
                                   inter_channels=None,
                                   sub_sample_factor=(2,2), use_norm=True, norm=norm)
            )
            final_linear_in_filters += attention_channels[d]
        self.attention_modules = nn.ModuleList(attention_blocks)
        
        # Bottleneck
        out_channels = in_channels * 2
        self.bottleneck = DownBlock(in_channels=in_channels, out_channels=out_channels, 
                             kernel_size=(3, 3), padding=(1, 1), use_norm=True, norm=norm, 
                                nblocks=nblocks, downsample=False, activation=activation)
        
        in_features = out_channels + final_linear_in_filters
        out_features = 2 * zdim
        self.distribution_params = nn.Linear(in_features=in_features, out_features=out_features)
        self.distribution_params.apply(init_he_normal)
        self.div = zdim
        # distribution_params:
        # means:
        # 0:zdim
        # log_std:
        # zdim:
        
    def forward(self, x):
        
        compatibility_blocks = []
        for d in range(self.depth):
            block, x = self.blocks[d](x)
            if d in self.attention_depths:
                compatibility_blocks.append(block)
        x = self.bottleneck(x)
        
        # Attention
        gated_blocks = []
        if self.visualize:
            attn_blocks = []
        for i, blk in enumerate(compatibility_blocks):
            g_blk, atttn_blk = self.attention_modules[i](blk, x)
            g_blk = g_blk.mean(dim=(2, 3))
            gated_blocks.append(g_blk)
            if self.visualize:
                attn_blocks.append(atttn_blk)
        
        # Global Average Pooling
        x = x.mean(dim=(2, 3))
        gated_blocks.append(x)
        
        x = torch.cat(gated_blocks, dim=1)
        params = self.distribution_params(x)
        
        if self.visualize:
            return params[:, :self.div], params[:, self.div:], attn_blocks
        return params[:, :self.div], params[:, self.div:]
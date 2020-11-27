import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_utils import ConvBlock, DeconvBlock, DownBlock, UpBlock, GridAttentionBlock, MultiGridAttentionBlock, GatingSignal

class DownBranch(nn.Module):
    
    def __init__(self, input_shape=(4, 240, 240), base_filters=16, depth=4, nblocks=1, activation='relu', 
                 norm='bn'):
        
        super(DownBranch, self).__init__()
        
        self.depth = depth
        blocks = []
        gates = []
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
            attention_channels.append(out_channels)
            
            in_channels = out_channels
            out_channels *= 2
        self.blocks = nn.ModuleList(blocks)
        self.attention_channels = [channels for channels in reversed(attention_channels)]
        
        # Bottleneck
        out_channels = in_channels * 2
        self.bottleneck = DownBlock(in_channels=in_channels, out_channels=out_channels, 
                             kernel_size=(3, 3), padding=(1, 1), use_norm=True, norm=norm, 
                                nblocks=nblocks, downsample=False, activation=activation)
        self.center_filters = out_channels
        
        
    def forward(self, x):
        
        blocks = []
        for d in range(self.depth):
            block, x = self.blocks[d](x)
            blocks.append(block)
        
        center = self.bottleneck(x)
        
        return center, [b for b in reversed(blocks)]
        

class UpBranch(nn.Module):
    
    def __init__(self, output_shape=(4, 240, 240), base_filters=16, depth=4, nblocks=1, 
                 activation='relu', norm='bn', attention_channels=None, center_filters=512, 
                 visualize=False, nclasses=3):
        
        super(UpBranch, self).__init__()
        
        self.visualize = visualize
        
        self.depth = depth
        blocks = []
        attention_blocks = []
        self.attention_channels = reversed(attention_channels)
        
        in_channels = center_filters
        
        for d in range(depth):
            # mul = 2 if d > 0 else 1 # Concatenation from the second block
            out_channels = attention_channels[d]
            attention_blocks.append(
                MultiGridAttentionBlock(in_channels=out_channels, gating_channels=in_channels, inter_channels=None,
                                   sub_sample_factor=(2,2), use_norm=True, norm=norm, nclasses=nclasses)
            )
            blocks.append(
                UpBlock(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(3, 3), padding=(1, 1), stride=2,
                    use_norm=True, norm=norm, nblocks=1, activation=activation
                )
            )
            
            in_channels = out_channels
        
        self.blocks = nn.ModuleList(blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        
        self.block1 = ConvBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), padding=(1, 1),
            use_norm=True, norm=norm, use_activation=True, activation=activation
        )
        
        
    def forward(self, center, gate, branches):
        
        if self.visualize:
            attn_blocks = []
        
        x = center
        _gate = gate
        for d in range(self.depth):
            # Attention
            g_block, attn = self.attention_blocks[d](branches[d], _gate)
            if self.visualize:
                attn_blocks.append(attn)
            # Upsample, concat, conv
            x = self.blocks[d](x, g_block)
            _gate = x
        
        x = self.block1(x)
        
        if self.visualize:
            return x, attn_blocks
        return x
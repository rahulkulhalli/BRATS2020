from __future__ import absolute_import

import torch.nn.init as init
import torch.nn as nn


def init_xavier_uniform(m):
    # Initialize weights to Glorot Uniform, bias to 0
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
            
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)

def init_he_normal(m):
    # Initialize weights to He Normal, bias to 0
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
            
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)

        
def init_kaiming(m):
    # Attention UNet init
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        
        
def init_gamma(m):
        
    if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        init.ones_(m.weight.data)

        
def init_beta(m):

    if isinstance(m, nn.BatchNorm2d) and m.bias is not None:
        init.zeros_(m.bias.data)
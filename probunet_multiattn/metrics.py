import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


bce = nn.BCELoss(reduction='none')


def kl_divergence(mu1, log_var1, mu2, log_var2, reduce='mean'):
    
    if mu1.ndimension() < 3:
        mu1 = mu1.unsqueeze(dim=-1)
        log_var1 = log_var1.unsqueeze(dim=-1)
        mu2 = mu2.unsqueeze(dim=-1)
        log_var2 = log_var2.unsqueeze(dim=-1)
    
    distribution_dimension = mu1.size(1)
    
    # KL Divergence between prior and posterior
    var2 = log_var2.exp()
    var1 = log_var1.exp()
    
    var2_inverse = 1. / var2
    
    trace = torch.bmm(var2_inverse.permute((0, 2, 1)), var1)
    mean_diff = mu2 - mu1
    mu_var_mu = mean_diff * var2_inverse
    mu_var_mu = torch.bmm(
        mu_var_mu.permute((0, 2, 1)),
        mean_diff
    )
    
    kl_div = 0.5 * ( torch.sum(log_var2, dim=(1, 2), keepdim=True) - torch.sum(log_var1, dim=(1, 2), keepdim=True) + 
                     trace + mu_var_mu - distribution_dimension ).squeeze(dim=-1).squeeze(dim=-1)
    
    if reduce == 'mean':
        kl_div = kl_div.mean()
    elif reduce == 'sum':
        kl_div = kl_div.sum()
    
    return kl_div


def loss(criterion, ypred, ytrue, prior_mean, prior_log_var, 
         posterior_mean, posterior_log_var, zdim=6, **kwargs):

    segmentation_loss = criterion(ypred, ytrue, **kwargs)
    
    kl_div = kl_divergence(posterior_mean, posterior_log_var, prior_mean, prior_log_var)
    
    return segmentation_loss, kl_div

def loss_unit(criterion, ypred, ytrue, prior_mean, prior_log_var, 
         posterior_mean, posterior_log_var, zdim=6, **kwargs):

    segmentation_loss = criterion(ypred, ytrue, **kwargs)
    
    kl_div = kl_divergence(posterior_mean, posterior_log_var, prior_mean, prior_log_var)
    
    _prior_var = 4.
    _prior_log_var = torch.ones_like(prior_log_var) * np.log(_prior_var)
    _prior_mean = torch.zeros_like(prior_mean)
    
    _prior_mean.requires_grad = False
    _prior_log_var.requires_grad = False
    # kl_div_unit = -0.5 * torch.sum(1 + prior_log_var - prior_mean.pow(2) - prior_log_var.exp())
    kl_div_unit = kl_divergence(prior_mean, prior_log_var, _prior_mean, _prior_log_var)
    
    return segmentation_loss, kl_div, kl_div_unit


def soft_dice(ypred, ytrue, reduce_classes='none', reduce='mean', smooth=1.):
    
    N, C, G, _ = ytrue.size()
    
    input_flat = ypred.view(N, C, G*G)
    target_flat = ytrue.view(N, C, G*G)
    
    intersection = 2.*torch.sum(input_flat*target_flat, dim=-1) + smooth
    union = torch.sum(input_flat, dim=-1) + torch.sum(target_flat, dim=-1) + smooth
    
    dice = intersection / union
    
    if reduce == 'mean':
        dice = torch.mean(dice, dim=0)
    elif reduce == 'sum':
        dice = torch.sum(dice, dim=0)
        
    if reduce_classes == 'mean':
        dice = torch.mean(dice, dim=-1)
    elif reduce_classes == 'sum':
        dice = torch.sum(dice, dim=-1)
    
    return dice


def dice_loss(ypred, ytrue, reduce_classes='none', reduce='mean', smooth=1.):
    
    return 1. - soft_dice(ytrue, ypred, reduce_classes=reduce_classes, 
                          reduce=reduce, smooth=smooth)


def dice_coeff(ypred, ytrue, reduce_classes='none', reduce='mean', eps=1e-8):
    
    input_flat = torch.round(ypred.detach())
    # input_flat = ypred.detach()
    target_flat = ytrue.detach()
    
    N, C, G1, G2 = ytrue.size()
    
    input_flat = input_flat.view(N, C, G1*G2)
    target_flat = target_flat.view(N, C, G1*G2)
    
    intersection = 2.*torch.sum(input_flat*target_flat, dim=-1)
    union = torch.sum(input_flat, dim=-1) + torch.sum(target_flat, dim=-1) + eps
    
    dice = intersection / union

    if reduce == 'mean':
        dice = torch.mean(dice, dim=0)
    elif reduce == 'sum':
        dice = torch.sum(dice, dim=0)

    if reduce_classes == 'mean':
        dice = torch.mean(dice, dim=-1)
    elif reduce_classes == 'sum':
        dice = torch.sum(dice, dim=-1)

    return dice

def dice_bce(ypred, ytrue, reduce_classes='none', reduce='mean', smooth=1., alpha=1., beta=1e-2):
    
    _dice = dice_loss(ypred, ytrue, reduce_classes=reduce_classes, reduce=reduce, smooth=smooth)
    _bce = weighted_bce(ypred, ytrue, reduce='mean', reduce_classes=reduce_classes)
        
    return alpha * _dice + beta * _bce
    return _bce


def weighted_bce(ypred, ytrue, reduce_classes='none', reduce='mean'):
    
    ytrue = ytrue.detach()

    N, C, G1, G2 = ytrue.size()
    
    with torch.no_grad():
        w_positive = (G1 * G2) / (2. * torch.sum(ytrue, dim=(2, 3), keepdim=True) + 1e-8)
        w_negative = (G1 * G2) / (2. * torch.sum(1. - ytrue, dim=(2, 3), keepdim=True) + 1e-8)

        weights_positive = ytrue * w_positive
        weights_negative = (1. - ytrue) * w_negative
        weights = weights_positive + weights_negative

    _bce = bce(ypred, ytrue)
    _bce = _bce * weights
    _bce = _bce.permute(0, 2, 3, 1)

    if reduce == 'mean':    
        _bce = torch.mean(_bce, dim=(0, 1, 2))
    elif reduce == 'sum':
        _bce = torch.sum(_bce, dim=(0, 1, 2))
        
    if reduce_classes == 'mean':
        _bce = torch.mean(_bce, dim=-1)
    elif reduce_classes == 'sum':
        _bce = torch.sum(_bce, dim=-1)

    return _bce

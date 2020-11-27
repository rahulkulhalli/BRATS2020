import os
import traceback

import torch
import torch.optim as optim
import torch.nn as nn

from .metrics import loss, dice_coeff, dice_loss, dice_bce, weighted_bce
from .utils.logger import Logger
from .utils.misc import MetricTracker, get_timestamp
try:
    from IPython import get_ipython
    runtime = get_ipython().__class__.__name__
    # Is it Jupyter?
    if runtime == 'ZMQInteractiveShell':
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm


class Trainer:
    
    def __init__(self, model, config, dataloader, val_dataloader=None, DEBUG=False):
        
        self.config = config
        self.epochs = config.epochs
        self.nclasses = config.nclasses
        self.zdim = config.zdim
        self.split_seg_loss = config.split_seg_loss
        try:
            self.display = config.display
        except:
            self.display = 'running'
        
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        metrics = ['segmentation_loss', 'total_loss', 'kl_div']
        
        if self.config.reduce_class_dice:
            metrics.append('dice_coeff')
        else:
            metrics += ['dice_coeff({})'.format(i) for i in range(self.nclasses)]
        
        if self.split_seg_loss:
            metrics += ['seg_loss({})'.format(i) for i in range(self.nclasses)]
            
        self.loss_f = config.loss['f']
        if self.loss_f == 'dice+bce':
            self.criterion = dice_bce
        elif self.loss_f == 'dice':
            self.criterion = dice_loss
        elif self.loss_f == 'weighted_bce':
            self.criterion = weighted_bce
        self.criterion_args = config.loss['args']
            
        self.decay_every = config.decay_every

        self.metrics = MetricTracker(metrics)
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.validate_every = config.validate_every
        self.checkpoint_every = config.checkpoint_every
        time = get_timestamp()
        self.checkpoint_path = os.path.join(config.checkpoint_path, time)
        
        self.logger = Logger(time, config.train_logdir, config.val_logdir, config.config_log, config)

    def train(self):
        
        try:
            for epoch in range(self.epochs):

                self.logger.epoch_begin()

                self.metrics.reset()

                loop = tqdm(self.dataloader, desc='Training')

                for i, (X, y) in enumerate(loop):
                    
                    y = y.float()
                    X = X.float()

                    self.optimizer.zero_grad()

                    out, prior_means, prior_log_vars, posterior_means, posterior_log_vars = self.model(
                        X, y, 
                        nsamples=1
                    )

                    y = y.to(out.device)
                    prior_means = prior_means.to(out.device)
                    prior_log_vars = prior_log_vars.to(out.device)
                    posterior_means = posterior_means.to(out.device)
                    posterior_log_vars = posterior_log_vars.to(out.device)

                    seg_loss, kl_div = loss(
                        self.criterion, out, y, prior_means, prior_log_vars, 
                        posterior_means, posterior_log_vars, zdim=self.config.zdim, 
                        reduce='mean', **self.criterion_args
                    )
                    
                    batch_dice_coeff = dice_coeff(out.detach(), y, reduce_classes='none', reduce='mean')
                    
                    current_stats, running_stats, total_loss = self.summarize_loss(
                        seg_loss, kl_div, 
                        batch_dice_coeff, 
                        y.size(0), out=out, y=y)

                    total_loss.backward()
                    self.optimizer.step()

                    self.logger.log_iteration(running_stats, i, 'train')

                    loop.set_description('Training: Epoch {}/{}'.format(epoch, self.epochs))
                    if self.display == 'running':
                        loop.set_postfix(**running_stats)
                    else:
                        loop.set_postfix(**current_stats)

                self.logger.log_epoch(running_stats, 'train')
                if ((epoch+1) % self.validate_every) == 0:
                    self.val(epoch)
                    
                if ((epoch+1) % self.checkpoint_every) == 0:
                    self.checkpoint(epoch+1)
                    
                if ((epoch + 1) % self.decay_every) == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.8

            self.logger.close()
        except Exception as e:
            traceback.print_exc()
            self.logger.close()
        finally:
            self.logger.close()
                
    def val(self, epoch):
        
        try:
        
            with torch.no_grad():
                # Use the same metric store
                self.metrics.reset()

                val_loop = tqdm(self.val_dataloader, desc='Validating')

                for i, (X, y) in enumerate(val_loop):
                    
                    y = y.float()
                    X = X.float()
                    
                    out, prior_means, prior_log_vars, posterior_means, posterior_log_vars = self.model(
                        X, y, nsamples=1, 
                        sample_from='prior'
                    )

                    y = y.to(out.device)
                    prior_means = prior_means.to(out.device)
                    prior_log_vars = prior_log_vars.to(out.device)
                    posterior_means = posterior_means.to(out.device)
                    posterior_log_vars = posterior_log_vars.to(out.device)

                    seg_loss, kl_div = loss(
                        dice_loss, out, y, prior_means, prior_log_vars, 
                        posterior_means, posterior_log_vars, 
                        zdim=self.config.zdim, 
                        reduce='mean'
                    )
                    
                    batch_dice_coeff = dice_coeff(out.detach(), y, reduce_classes='none', reduce='mean')

                    current_stats, running_stats, total_loss = self.summarize_loss(
                        seg_loss, kl_div, 
                        batch_dice_coeff,
                        y.size(0))
                    self.logger.log_iteration(running_stats, i, 'val')

                    val_loop.set_description('Validating: Epoch {}/{}'.format(epoch+1, self.epochs))
                    if self.display == 'running':
                        val_loop.set_postfix(**running_stats)
                    else:
                        val_loop.set_postfix(**current_stats)

            self.logger.log_epoch(running_stats, 'val')
        except Exception as e:
            traceback.print_exc()
            self.logger.close('val')

    def running_stats(self, stats, nsamples, status='running'):
        
        # Update the metrics
        self.metrics.update(stats)
        return self.metrics.format()

    def current_stats(self):
        return self.metrics.current_stats()
    
    def summarize_loss(self, seg_loss, kl_div, batch_dice_coeff, nsamples, status='running', out=None, y=None):
        
        '''
            Summarize to a loggable dictionary. Combine all losses to one loss to
            backpropagate.
        '''
        
        print_summary = dict()
        kl_div_loss = kl_div
        print_summary['kl_div'] = kl_div.item()

        if self.split_seg_loss and seg_loss.size(0) > 1:
            print_summary.update({'seg_loss({})'.format(i): seg_loss[i].item() for i in range(self.nclasses)})
        segmentation_loss = torch.sum(seg_loss)
        print_summary['segmentation_loss'] = segmentation_loss.item()
        total_loss = segmentation_loss + kl_div_loss
        print_summary['total_loss'] = total_loss.item()
        
        if batch_dice_coeff.size(0) > 1:
            print_summary.update({'dice_coeff({})'.format(i): batch_dice_coeff[i].item() for i in range(self.nclasses)})
        else:
            print_summary['dice_coeff'] = batch_dice_coeff.item()

        running_stats = self.running_stats(print_summary, nsamples, status=status)

        current_stats = self.current_stats()
        
        return current_stats, running_stats, total_loss
    
    def checkpoint(self, epoch):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.checkpoint(self.checkpoint_path, epoch)
        else:
            self.model.checkpoint(self.checkpoint_path, epoch)
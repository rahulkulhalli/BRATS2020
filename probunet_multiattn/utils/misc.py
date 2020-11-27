from __future__ import absolute_import
from datetime import datetime


# Source https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def format(self, status='running'):
        '''
        Return a dict of the metric in Loggable format
        '''
        if status == 'running':
            return {self.name: ('{'+self.fmt+'}').format(self.avg)}
        elif status == 'current':
            return {self.name: ('{'+self.fmt+'}').format(self.val)}
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    
    def __init__(self, metrics, fmt=':.4f'):
        
        self.fmt = fmt
        self.metric_list = metrics
        self.metrics = {metric: AverageMeter(metric, fmt) for metric in metrics}
        self.display_stats = dict()
        self.reset()
        
    def reset(self):
        
        for metric in self.metric_list: 
            self.metrics[metric].reset()
            self.display_stats[metric] = ('{'+self.fmt+'}').format(0.)
    
    def update(self, stats, nsamples=1, status='running'):
        
        for metric in self.metric_list:
            self.metrics[metric].update(stats[metric], nsamples)
            val_to_log = self.metrics[metric].avg if status == 'running' else self.metrics[metric].val
            self.display_stats[metric] = ('{'+self.fmt+'}').format(val_to_log)

    def current_stats(self):
        display_current_stats = dict()
        for metric in self.metric_list:
            val_to_log = self.metrics[metric].val
            display_current_stats[metric] = ('{'+self.fmt+'}').format(val_to_log)
        return display_current_stats
    
    def format(self):
        
        return self.display_stats


def get_timestamp(fmt='%d-%m-%Y-%H-%M-%S'):
    now = datetime.now()
    return now.strftime(fmt)
from __future__ import absolute_import
import os
import json


class Logger:
    
    def __init__(self, time, train_logfile=None, val_logfile=None, config_logpath=None, config=None, log_phases=['train', 'val']):
        
        assert all(phase in ['train', 'val'] for phase in log_phases)
        if 'train' in log_phases: assert train_logfile is not None
        if 'val' in log_phases: assert val_logfile is not None

        self.log_phases = log_phases
        self.train_logfile = train_logfile
        self.val_logfile = val_logfile

        self.handlers = {}
        for phase, filename in zip(log_phases, [train_logfile, val_logfile]):
            basepath = os.path.dirname(filename)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            self.handlers[phase] = open(filename, 'a')
            self.handlers[phase].write('[{}]:\n'.format(time))
        
        with open(os.path.join(config_logpath, '{}.json'.format(time)), 'w') as f:
            f.write(json.dumps(config.__dict__))

        self.epoch = 1
        
    def log(self, log_str, phase):
        self.handlers[phase].write(log_str)

    def make_pretty(self, metrics, iteration=None):
        logging_strs = []
        for metric, value in metrics.items():
            print_str = '{}: {}'.format(metric, value)
            logging_strs.append(print_str)
        print_str = ' | '.join(logging_strs) + '\n'
        if iteration is not None:
            print_str = 'Iteration: {} => '.format(iteration) + print_str
        return print_str 

    def generate_epoch_log(self, metrics):
        return '\n'.join(['[Epoch Average] => ' + self.make_pretty(metrics), ''])

    def log_epoch(self, metrics, phase):
        self.log(self.generate_epoch_log(metrics), phase)
        if phase == 'train':
            self.epoch += 1

    def generate_iteration_log(self, metrics, iteration):
        return self.make_pretty(metrics, iteration)

    def log_iteration(self, metrics, iteration, phase):
        self.log(self.generate_iteration_log(metrics, iteration), phase)

    def epoch_begin(self):
        for phase in self.log_phases:
            self.handlers[phase].write('[Epoch {}]\n'.format(self.epoch))

    def close(self, close_phase=None):
        if close_phase is not None:
            self.handlers[close_phase].close()
            return
        for phase in self.log_phases:
            self.handlers[phase].close()

    def __del__(self):
        for phase in self.log_phases:
            self.handlers[phase].close()


if __name__ == '__main__':

    import random
    from misc import get_timestamp

    time = get_timestamp()
    logger = Logger(time, train_logfile='tests/train.log', val_logfile='tests/val.log')
    for epoch in range(3):
        logger.epoch_begin()
        for i in range(2):
            metrics = {'seg_loss': random.gauss(0., 1.), 'kl_div': random.gauss(0., 1.)}
            logger.log_iteration(metrics, i, 'train')
        logger.log_epoch(metrics, 'train')

        for j in range(3):
            metrics = {'seg_loss': random.gauss(0., 1.), 'kl_div': random.gauss(0., 1.)}
            logger.log_iteration(metrics, j, 'val')
        logger.log_epoch(metrics, 'val')
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import parallel_apply

from .modules import DownBranch, UpBranch
from .unet import UNet, fcomb
from .gaussian_conv_net import GaussianConvNet
from .module_utils import ConvBlock

class ProbUNet(nn.Module):
    
    def __init__(
        self, input_shape=(4, 240, 240), output_shape=(3, 240, 240),
        base_filters=16, 
        depth=4, nblocks=1, 
        nclasses=3, zdim=6, use_posterior=True,
        devices={
            'unet': 'cuda:0',
            'prior_net': 'cuda:1',
            'posterior_net': 'cuda:2',
            'output': 'cuda:2'
        },
        checkpoints=None, activation='relu', norm='bn', 
        visualize=False, nattn_blocks=3
    ):
        
        super(ProbUNet, self).__init__()
        
        self.devices = {_net: torch.device(device) for _net, device in devices.items()}
        unique_devices = list(set(device for device in devices.values()))
        n_unique_devices = len(unique_devices)
        self.n_unique_devices = n_unique_devices
#         if n_unique_devices > 1:
#             # If models are to be run parallely, ensure that they all 
#             # are on  different devices for now
#             assert n_unique_devices == len(devices)
        
        self.zdim = zdim
        self.nclasses = nclasses
        self.nattn_blocks = nattn_blocks
        
        self.visualize = visualize
        
        #### UNet ####
        self.unet = UNet(input_shape, output_shape, base_filters, depth, nblocks, activation=activation, 
                         norm=norm, visualize=visualize, nclasses=self.nattn_blocks)
        if checkpoints is not None:
            print('Loading predtrained UNet')
            self.unet.load_state_dict(torch.load(checkpoints['unet'], map_location='cpu'))
        self.unet = self.unet.to(self.devices['unet'])
        # TODO: Maybe parallalize UNet
        
        #### Prior Net ####
        self.prior_net = GaussianConvNet(
            input_shape=input_shape, base_filters=base_filters, depth=depth,
            nblocks=nblocks, zdim=zdim, activation=activation, norm=norm, visualize=visualize
        )
        if checkpoints is not None:
            print('Loading predtrained Prior Net')
            self.prior_net.load_state_dict(torch.load(checkpoints['prior_net'], map_location='cpu'))
        self.prior_net = self.prior_net.to(self.devices['prior_net'])
        
        #### Posterior Net ####
        self.use_posterior = use_posterior# or self.training
        if use_posterior:
            posterior_input_shape = (input_shape[0]+nclasses, ) + input_shape[1:]
            self.posterior_net = GaussianConvNet(
                input_shape=posterior_input_shape, base_filters=base_filters, 
                depth=depth, nblocks=nblocks, zdim=zdim, activation=activation, norm=norm, 
                visualize=False
            )
            if checkpoints is not None:
                print('Loading predtrained Posterior Net')
                self.posterior_net.load_state_dict(torch.load(checkpoints['posterior_net'], map_location='cpu'))
            self.posterior_net = self.posterior_net.to(self.devices['posterior_net'])
        
        #### Combination to generate output ####
        # in_channels = self.unet.final_block.block[0].out_channels + zdim
        in_channels = self.unet.up_branch.block1.block[0].out_channels + zdim
        self.comb = fcomb(in_channels=in_channels, output_shape=output_shape, base_filters=base_filters, 
                          activation=activation, norm=norm)
        
        if checkpoints is not None:
            print('Loading predtrained fcomb')
            self.comb.load_state_dict(torch.load(checkpoints['fcomb'], map_location='cpu'))
        self.comb = self.comb.to(self.devices['output'])

    def forward(self, x, y=None, nsamples=1, sample_from='posterior', reduce='none', 
                samples=None):
        
        if self.use_posterior:
            assert (y is not None)
        
        x_unet = x.to(self.devices['unet'])
        modules = [self.unet]
        inputs = [x_unet]
        devices = [self.devices['unet']]
        
        if self.training or sample_from == 'prior':
            x_prior = Variable(x.data.to(self.devices['prior_net']))
            modules.append(self.prior_net)
            inputs.append(x_prior)
            devices.append(self.devices['prior_net'])
        
        if self.training or self.use_posterior:
            x_posterior = Variable(x.data.to(self.devices['posterior_net']))
            y_posterior = Variable(y.data.to(self.devices['posterior_net']))
            posterior_in = torch.cat([x_posterior, y_posterior], dim=1)
            modules.append(self.posterior_net)
            inputs.append(posterior_in)
            devices.append(self.devices['posterior_net'])
        
        if self.n_unique_devices > 1:
            output = parallel_apply(modules, inputs, devices=devices)
        else:
            output = [module(module_input) for module, module_input in zip(modules, inputs)]
        
        # Sample
        # Prior
        if sample_from == 'prior' or self.training:
            
            prior_params = output[1]
            if self.visualize:
                attn_blocks_prior = prior_params[2]
            prior_means = prior_params[0]
            prior_log_vars = prior_params[1]
            prior_samples = self.sample(means=prior_means, log_vars=prior_log_vars, 
                                        nsamples=nsamples, sample_from='prior')
        
        # Posterior
        if sample_from == 'posterior' or self.training:
            # If training, a samples from prior has also been drawn
            if self.training:
                posterior_params = output[2]
            # If eval, sample from prior won't be drawn
            else:
                posterior_params = output[1]
            posterior_means = posterior_params[0]
            posterior_log_vars = posterior_params[1]
            posterior_samples = self.sample(means=posterior_means, log_vars=posterior_log_vars, 
                                        nsamples=nsamples, sample_from='posterior')
        
        if samples is None:
            samples = posterior_samples if self.training else prior_samples
        
        if self.visualize:
            unet_features = output[0][0].to(self.devices['output'])
            attn_blocks_unet = output[0][1]
        else:
            unet_features = output[0].to(self.devices['output'])
        samples = samples.to(self.devices['output'])
        out = self.comb(unet_features, samples, reduce=reduce)
        
        if self.training or self.use_posterior:
            return out, prior_means, prior_log_vars, posterior_means, posterior_log_vars
        elif sample_from == 'prior':
            if self.visualize:
                return out, prior_means, prior_log_vars, attn_blocks_unet, attn_blocks_prior
            return out, prior_means, prior_log_vars
        else:
            return out, posterior_means, posterior_log_vars
 
    def sample(self, x=None, means=None, log_vars=None, nsamples=1, sample_from='prior'):
        
        if (means is None) and (log_vars is None):
            # This will only occur if explicitly sampling and 
            # never occur during training
            assert x is not None
            with torch.no_grad():
                return self.forward(x, None, nsamples, sample_from=sample_from)
        
        N = means.shape[0]
        if nsamples > 1:
            _means = torch.repeat_interleave(means, nsamples, dim=0)
            _log_vars = torch.repeat_interleave(log_vars, nsamples, dim=0)
        else:
            _means = means
            _log_vars = log_vars
        
        
        #############
        # Changed samples = _means + torch.randn_like(_means) * torch.exp(_log_vars / 2.) -> 
        # samples = _means + torch.randn_like(_means) * torch.exp(_log_vars)
        ##############
        
        # samples are samples per class
        # samples = _means + torch.randn_like(_means) * torch.exp(_log_vars)
        samples = _means + torch.randn_like(_means) * torch.exp(_log_vars / 2.)
        
        return samples

    def checkpoint(self, checkpoint_path, epoch):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Checkpoint all the modules separately
        path = os.path.join(checkpoint_path, 'unet_{}.pth'.format(epoch))
        torch.save(self.unet.state_dict(), path)

        path = os.path.join(checkpoint_path, 'prior_net_{}.pth'.format(epoch))
        torch.save(self.prior_net.state_dict(), path)
        
        path = os.path.join(checkpoint_path, 'posterior_net_{}.pth'.format(epoch))
        torch.save(self.posterior_net.state_dict(), path)
        
        path = os.path.join(checkpoint_path, 'fcomb_{}.pth'.format(epoch))
        torch.save(self.comb.state_dict(), path)
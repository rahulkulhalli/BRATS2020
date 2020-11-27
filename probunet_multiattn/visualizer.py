'''
Visualizer visualizes the effect of varying the latent samples per slice
'''

import os
import traceback

import torch
import torch.nn as nn

import numpy as np
import nibabel

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
    
import matplotlib.pyplot as plt
#%matplotlib inline
    
    
class Visualizer:
    
    def __init__(self, model, config, dataset):
        
        self.config = config
        
        self.original_dim = config.original_dim
        
        self.batch_size = config.batch_size
        self.nclasses = config.nclasses
        self.zdim = config.zdim

        self.model = model
        self.model = self.model.eval()
        
        self.dataset = dataset
                
    def predict_and_visualize(self, pid, slice_no, vary_dim=[0, 1], sample_range=(-2., 2.), nsteps=10):
        '''
        param pid: patient ID from BRATS dataset
        param slice_no: slices to visualize. (int)
        '''
        assert isinstance(slice_no, int)
        try:
        
            with torch.no_grad():
                
                ix = self.dataset.index[pid]
                
                X, y, _pid = self.dataset.__getitem__(ix)
                assert pid == _pid, 'pids dont match. pid: {}, _pid: {}'.format(pid, _pid)
                
                _X = X[slice_no, :, :, :].clone()
                if _X.dim() < 4:
                    _X = _X.unsqueeze(dim=0)
                
                predictions = self._predict(_X, slice_no=slice_no, vary_dim=vary_dim, sample_range=sample_range, nsteps=nsteps)

                # Postprocess
                # predictions = self.post_process(predictions)
                return predictions
                
        except Exception as e:
            traceback.print_exc()

    def _predict(self, X, y=None, slice_no=None, vary_dim=[0, 1], sample_range=(-2., 2.), nsteps=10):

        X = X.float()

        slices, channels, G1, G2 = X.size()
        predictions = np.zeros((1, G1, G2))
        
        ## Need to have two passes
        out, prior_means, prior_log_vars = self.model(
            X, nsamples=1,
            sample_from='prior', 
            reduce='none', 
            samples=None
        )
        
        samples = self.sample(prior_means, prior_log_vars, vary_dim=vary_dim, sample_range=sample_range, nsteps=nsteps)
        out, prior_means, prior_log_vars = self.model(
            X, nsamples=1,
            sample_from='prior', 
            reduce='none', 
            samples=samples
        )

        # Format output according to priority

        # output = torch.LongTensor(size=(_slices, G1, G2))
        out = out.detach() > 0.5
        output = out.sum(dim=2).squeeze(dim=0)
        predictions = output.cpu().numpy()
        
        self.visualize(X, predictions, y, slice_no, vary_dim=vary_dim, nsteps=nsteps)
        
        # TODO: Cast predictions to dtype of segmentations
        return predictions
    
    def sample(self, mean, log_var, vary_dim=[0, 1], sample_range=(-2., 2.), nsteps=10):
        
        samples = (mean + torch.randn_like(mean) * torch.exp(log_var / 2.)).detach().cpu().numpy()
        
        sample_values = np.linspace(sample_range[0], sample_range[1], num=nsteps)
        if len(vary_dim) > 1:
            assert len(vary_dim) == 2
            
            samples = np.tile(samples.T, nsteps).T
            samples = np.repeat(samples, nsteps, axis=0)
            
            dim1 = np.repeat(sample_values, nsteps)
            dim2 = np.tile(sample_values, nsteps)
            # samples = np.zeros((dim1.shape[0], self.zdim))
            samples[:, vary_dim[0]] = dim1
            samples[:, vary_dim[1]] = dim2
        else:
            assert len(vary_dim) == 1
            
            samples = np.tile(samples.T, sample_values.shape[0]).T
            
            # samples = np.zeros((sample_values.shape[0], self.zdim))
            samples[:, vary_dim[0]] = sample_values
        return torch.FloatTensor(samples)
    
    def display_input(self, X):

        # Shot the inputs
        # f = plt.figure(figsize=(7, 4))
        f = plt.figure(dpi=300)
        x_img = X[0, :, :, :].detach().cpu().numpy()
        plt.subplot(141)
        plt.imshow(x_img[0, :, :], cmap='gray')
        plt.subplot(142)
        plt.imshow(x_img[1, :, :], cmap='gray')
        plt.subplot(143)
        plt.imshow(x_img[2, :, :], cmap='gray')
        plt.subplot(144)
        plt.imshow(x_img[3, :, :], cmap='gray')
        plt.show()
        
    def generate_grid(self, output, vary_dim, nsteps):
        
        N, H, W = output.shape
        
        # Format predictions
        img = np.zeros(output.shape+(3,), dtype=np.uint8)
        img[output == 1, :] = np.array([255, 255, 0])
        img[output == 2, :] = np.array([0, 255, 0])
        img[output == 3, :] = np.array([255, 0, 0])
        
        if len(vary_dim) == 1:
            # Only one dimension varies, 1-d stacking
            out = np.hstack([img[i, :, :, :] for i in range(N)])
            
            # f = plt.figure(figsize=(12, 12))
            f = plt.figure(dpi=300)
            plt.imshow(out)
            plt.axis('off')
            plt.show()
            
            return out
        else:
            out = np.zeros((nsteps*H, nsteps*W, 3), dtype=img.dtype)
            for i in range(nsteps):
                rstart = i*H
                rend = rstart + H
                for j in range(nsteps):
                    cstart = j * W
                    cend = cstart + W
                    out[rstart:rend, cstart:cend, :] = img[i + j*nsteps, ...]
            
            # f = plt.figure(figsize=(10, 10))
            f = plt.figure(dpi=300)
            plt.imshow(out)
            plt.show()
            
            return out

    def visualize(self, X, predictions, y=None, slice_no=None, vary_dim=[0], nsteps=10):
        
        # self.display_input(X)
        
        output_display_image = self.generate_grid(predictions, vary_dim, nsteps)

        if y is not None:
            _y = y[i, :, :, :]
            _y = _y.cpu().numpy().astype(np.uint8)

            x_img = X[i, :, :, :].detach().cpu().numpy()
            plt.subplot(241)
            plt.imshow(x_img[0, :, :], cmap='gray')
            plt.subplot(242)
            plt.imshow(x_img[1, :, :], cmap='gray')
            plt.subplot(243)
            plt.imshow(x_img[2, :, :], cmap='gray')
            plt.subplot(244)
            plt.imshow(x_img[3, :, :], cmap='gray')

            plt.subplot(245)
            plt.imshow(img)
            plt.subplot(246)
            plt.imshow((_y[0, :, :]*255).astype(np.uint8), cmap='gray')
            plt.subplot(247)
            plt.imshow((_y[1, :, :]*255).astype(np.uint8), cmap='gray')
            plt.subplot(248)
            plt.imshow((_y[2, :, :]*255).astype(np.uint8), cmap='gray')
            
    def post_process(self, predictions):
        
        H, W, slices = predictions.shape
        
        out = np.zeros(self.original_dim + (slices,))
        
        c_x, c_y = out.shape[0]//2, out.shape[1]//2
        crop_offset = H//2
        out[c_x-crop_offset:c_x+crop_offset, c_y-crop_offset:c_y+crop_offset] = predictions.copy()
        
        return out.astype(np.float64)
import os
import traceback

import torch
import torch.nn as nn

import numpy as np
import nibabel

import cv2

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
        
        self.save_base = '/path/to/base/dir'
        
        self.dataset = dataset
                
    def predict_and_visualize(self, pid, slice_no):
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
                _y = y[slice_no, :, :, :].clone()
                if _X.dim() < 4:
                    _X = _X.unsqueeze(dim=0)
                    _y = _y.unsqueeze(dim=0)
                
                unet_attn, prior_attn, output = self._predict(_X, slice_no=slice_no)
                self.display_input_and_output(_X, output, _y.detach().cpu().numpy(), pid, slice_no)
                
                print('UNet')
                for block in unet_attn:
                    print(block[0].shape[2:])
                    self.visualize_attn(_X, block, pid, 'unet', slice_no)
                    
                print('Prior')
                for block in prior_attn:
                    print(block[0].shape[2:])
                    self.visualize_attn(_X, block, pid, 'prior', slice_no)
                
        except Exception as e:
            traceback.print_exc()
    
    def visualize_attn(self, X, blocks, pid, network, slice_no):
        
        f = plt.figure(dpi=100)
        for i, _block in enumerate(blocks):
            plt.subplot(1, len(blocks), i+1)
            
            vblock = _block[0, 0, :, :].detach().cpu().numpy()
            plt.imshow(vblock, cmap='gray')
            plt.axis('off')
            
            s = vblock.shape[-1]
            _dir = self.save_base + '{}/{}'.format(pid, slice_no)
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            fname = _dir + '/{}_{}_{}.png'.format(network, s, i)
            
            vblock = (vblock * 255.).astype(np.uint8)
            
            # cv2.imwrite(fname, vblock)
        plt.show()
    
    def _predict(self, X, y=None, slice_no=None, vary_dim=[0, 1], sample_range=(-2., 2.), nsteps=10):

        X = X.float()

        slices, channels, G1, G2 = X.size()
        predictions = np.zeros((1, G1, G2))
        
        ## Need to have two passes
        out, prior_means, prior_log_vars, attn_blocks_unet, attn_blocks_prior = self.model(
            X, nsamples=10,
            sample_from='prior', 
            reduce='mean', 
            samples=None
        )
        
        out = out.detach() > 0.5
        output = out.sum(dim=1).squeeze(dim=0)
        predictions = output.cpu().numpy()
        
        # self.visualize(X, predictions, y, slice_no, vary_dim=vary_dim, nsteps=nsteps)
        
        return attn_blocks_unet, attn_blocks_prior, predictions
    
    def display_input_and_output(self, X, output, y, pid, slice_no):
        
        H, W = output.shape
        
        # Format predictions
        img = np.zeros(output.shape+(3,), dtype=np.uint8)
        img[output == 1, :] = np.array([255, 255, 0])
        img[output == 2, :] = np.array([0, 255, 0])
        img[output == 3, :] = np.array([255, 0, 0])
        
        yvis = y[0, ...].sum(axis=0)
        img_y = np.zeros(output.shape+(3,), dtype=np.uint8)
        img_y[yvis == 1, :] = np.array([255, 255, 0])
        img_y[yvis == 2, :] = np.array([0, 255, 0])
        img_y[yvis == 3, :] = np.array([255, 0, 0])
        
        # Shot the inputs
        # f = plt.figure(figsize=(7, 4))
        f = plt.figure(dpi=200)
        x_img = X[0, :, :, :].detach().cpu().numpy()
        plt.subplot(131)
        plt.imshow(x_img[0, :, :], cmap='gray')
        # plt.title('T1')
        plt.axis('off')
        
#         _dir = self.save_base + '{}/{}'.format(pid, slice_no)
#         fname = _dir + '/t1.png'
#         f = (x_img[0] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
#         fname = _dir + '/t2.png'
#         f = (x_img[1] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
#         fname = _dir + '/t1ce.png'
#         f = (x_img[2] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
#         fname = _dir + '/flair.png'
#         f = (x_img[3] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
#         fname = _dir + '/pred.png'
#         f = img[:, :, ::-1]#(x_img[0] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
#         fname = _dir + '/gt.png'
#         f = img_y[:, :, ::-1]#(x_img[0] * 255.).astype(np.uint8)
#         cv2.imwrite(fname, f)
        
        
#         plt.subplot(242)
#         plt.imshow(x_img[1, :, :], cmap='gray')
#         plt.title('T2')
#         plt.axis('off')
        
#         plt.subplot(243)
#         plt.imshow(x_img[2, :, :], cmap='gray')
#         plt.title('T1CE')
#         plt.axis('off')
        
#         plt.subplot(244)
#         plt.imshow(x_img[3, :, :], cmap='gray')
#         plt.title('Flair')
#         plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(img)
        # plt.title('Prediction')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(img_y)
        # plt.title('Ground Truth')
        plt.axis('off')
        plt.show()

    def visualize(self, X, predictions, y=None, slice_no=None, vary_dim=[0], nsteps=10):
        
        # self.display_input(X)
        
        output_display_image = self.generate_grid(predictions, vary_dim, nsteps)

        if y is not None:
            _y = y[i, :, :, :]
            _y = _y.cpu().numpy().astype(np.uint8)
#                     _y = _y > 0
#                     y_img = np.zeros((H, W, 3), dtype=np.uint8)
#                     y_img[_y[0, :, :], :] = np.array([255, 255, 0])
#                     y_img[_y[1, :, :], :] = np.array([0, 255, 0])
#                     y_img[_y[2, :, :], :] = np.array([255, 0, 0])

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
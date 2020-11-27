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


class Predictor:
    
    def __init__(self, model, config, dataloader, save=True):
        
        self.config = config
        
        self.original_dim = config.original_dim
        
        self.batch_size = config.batch_size
        self.nclasses = config.nclasses
        self.zdim = config.zdim
        self.save = save
        if save:
            self.output_path = config.output_path
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            print('Saving output to {}'.format(self.output_path))

        self.model = model
        self.model = self.model.eval()
        
        self.dataloader = dataloader
                
    def predict(self, visualize=False, nsamples=1, reduce='none'):
        
        try:
        
            with torch.no_grad():

                loop = tqdm(self.dataloader, desc='Generating predictions:')

                for i, (X, affine, pid) in enumerate(loop):
                    
                    predictions = self._predict(X, visualize=visualize, nsamples=nsamples, reduce=reduce)
                    
                    # Postprocess
                    predictions = self.post_process(predictions)
                    _predictions = np.zeros_like(predictions)
                    
                    _predictions[predictions == 3] = 4
                    _predictions[predictions == 2] = 1
                    _predictions[predictions == 1] = 2
                    predictions = _predictions.copy()
                    
                    if self.save:
                        # Save
                        img = nibabel.Nifti1Image(predictions, affine)
                        img.to_filename(os.path.join(self.output_path, '{}.nii.gz'.format(pid)))

                    loop.set_description('Generating predictions: Volume {}/{}'.format(
                            i+1, len(self.dataloader)
                        )
                    )
        except Exception as e:
            traceback.print_exc()

    def _predict(self, X, y=None, visualize=False, nsamples=1, reduce='none'):

        X = X.float()

        slices, channels, G1, G2 = X.size()
        predictions = np.zeros((slices, G1, G2))
        extra_slices = slices % self.batch_size
        n_steps = (slices // self.batch_size) + (extra_slices != 0)
        for _i in range(n_steps):
            start = _i * self.batch_size
            end = start + self.batch_size
            if end >= slices:
                end = slices
            
            _slices = end - start
            _X = X[start:end, :, :, :].clone()
            out, prior_means, prior_log_vars = self.model(
                _X, nsamples=nsamples, 
                sample_from='prior', reduce=reduce
            )
            out = out.detach() > 0.5
            
            # Format output according to priority
            output = out.sum(dim=1)
            output = output.cpu().numpy()

            predictions[start:end, :, :] = output.copy()

        predictions = np.transpose(predictions, axes=(1, 2, 0))
        
        if visualize:
            self.visualize(X, predictions, y)
        
        # TODO: Cast predictions to dtype of segmentations
        return predictions
    
    def visualize(self, X, predictions, y=None):
        
        H, W, sl = predictions.shape
        for i in range(sl):
            _x = predictions[:, :, i]
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[_x == 1, :] = np.array([255, 255, 0])
            img[_x == 2, :] = np.array([0, 255, 0])
            img[_x == 3, :] = np.array([255, 0, 0])

            f = plt.figure(figsize=(7, 4))

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
            else:
                x_img = X[i, :, :, :].detach().cpu().numpy()
                plt.subplot(151)
                plt.imshow(x_img[0, :, :], cmap='gray')
                plt.subplot(152)
                plt.imshow(x_img[1, :, :], cmap='gray')
                plt.subplot(153)
                plt.imshow(x_img[2, :, :], cmap='gray')
                plt.subplot(154)
                plt.imshow(x_img[3, :, :], cmap='gray')
                plt.subplot(155)
                plt.imshow(img)
            plt.title(i)
            plt.show()
            print('-'*10)
            
    def post_process(self, predictions):
        
        # Pad with zeros
        
        H, W, slices = predictions.shape
        
        out = np.zeros(self.original_dim + (slices,))
        
        c_x, c_y = out.shape[0]//2, out.shape[1]//2
        crop_offset = H//2
        out[c_x-crop_offset:c_x+crop_offset, c_y-crop_offset:c_y+crop_offset] = predictions.copy()
        
        return out.astype(np.float64)
    
    
class PredictorSlice:
    
    def __init__(self, model, config, dataset):
        
        self.config = config
        
        self.original_dim = config.original_dim
        
        self.batch_size = config.batch_size
        self.nclasses = config.nclasses
        self.zdim = config.zdim

        self.model = model
        self.model = self.model.eval()
        
        self.dataset = dataset
                
    def predict(self, pid, slice_no, visualize=True):
        '''
        param pid: patient ID from BRATS dataset
        param slice_no: slices to visualize. int of iterable of ints.
        '''
        try:
        
            with torch.no_grad():
                
                ix = self.dataset.index[pid]
                
                X, affine, _pid = self.dataset.__getitem__(ix)
                assert pid == _pid, 'pids dont match. pid: {}, _pid: {}'.format(pid, _pid)
                
                _X = X[slice_no, :, :, :].clone()
                # _X = X[slice_no, :, :, :]
                if _X.dim() < 4:
                    _X = _X.unsqueeze(dim=0)
                
                predictions = self._predict(_X, slice_no=slice_no, visualize=visualize)

                # Postprocess
                # predictions = self.post_process(predictions)
                return predictions
                
        except Exception as e:
            traceback.print_exc()

    def _predict(self, X, y=None, slice_no=None, visualize=False):

        X = X.float()

        slices, channels, G1, G2 = X.size()
        predictions = np.zeros((slices, G1, G2))
        extra_slices = slices % self.batch_size
        n_steps = (slices // self.batch_size) + (extra_slices != 0)
        for _i in range(n_steps):
            start = _i * self.batch_size
            end = start + self.batch_size
            if end >= slices:
                end = slices
            
            _slices = end - start
            _X = X[start:end, :, :, :].clone()
            out, prior_means, prior_log_vars = self.model(
                _X, nsamples=1, 
                sample_from='prior', reduce='none'
            )

            # Format output according to priority
            
            # output = torch.LongTensor(size=(_slices, G1, G2))
            out = out.detach() > 0.5
#             output[out[:, 0, :, :]] = 1
#             output[out[:, 1, :, :]] = 2
#             output[out[:, 2, :, :]] = 3
            output = out.sum(dim=1)
            output = output.cpu().numpy()

            predictions[start:end, :, :] = output.copy()

        predictions = np.transpose(predictions, axes=(1, 2, 0))
        print(predictions.shape)
        
        if visualize:
            self.visualize(X, predictions, y, slice_no)
        
        # TODO: Cast predictions to dtype of segmentations
        return predictions
    
    def visualize(self, X, predictions, y=None, slice_no=None):
        
        H, W, sl = predictions.shape
        for i in range(sl):
            _x = predictions[:, :, i]
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[_x == 1, :] = np.array([255, 255, 0])
            img[_x == 2, :] = np.array([0, 255, 0])
            img[_x == 3, :] = np.array([255, 0, 0])

            f = plt.figure(figsize=(7, 4))

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
            else:
                x_img = X[i, :, :, :].detach().cpu().numpy()
                plt.subplot(151)
                plt.imshow(x_img[0, :, :], cmap='gray')
                plt.subplot(152)
                plt.imshow(x_img[1, :, :], cmap='gray')
                plt.subplot(153)
                plt.imshow(x_img[2, :, :], cmap='gray')
                plt.subplot(154)
                plt.imshow(x_img[3, :, :], cmap='gray')
                plt.subplot(155)
                plt.imshow(img)
            if isinstance(slice_no, list):
                title = slice_no[i]
            elif slice_no is None:
                title = i
            else:
                title = slice_no
            plt.title(title)
            plt.show()
            print('-'*10)
            
    def post_process(self, predictions):
        
        # Pad with zeros
        #pad_top = offsets[0]
        #pad_bot = 244 - offsets[1]
        #pad_left = offsets[2]
        #pad_right = 244 - offsets[3]
        #predictions = np.pad(predictions, 
        #                     ((pad_top, pad_bottom),
        #                      (pad_left, pad_right), 
        #                      (0, 0)))
        
        H, W, slices = predictions.shape
        
        out = np.zeros(self.original_dim + (slices,))
        
        c_x, c_y = out.shape[0]//2, out.shape[1]//2
        crop_offset = H//2
        out[c_x-crop_offset:c_x+crop_offset, c_y-crop_offset:c_y+crop_offset] = predictions.copy()
        
        return out.astype(np.float64)
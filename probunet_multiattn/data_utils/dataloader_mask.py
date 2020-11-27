"""
This script will create the custom DataLoader for
the BRATS2020 dataset (created using create_data.py and preprocess.py).
"""

import os
import torch
import glob
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy import ndimage

# Structure for a 2D neighbourhood
structure = np.expand_dims(np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]), axis=-1)
structure = np.concatenate([np.zeros((3, 3, 1)), structure, np.zeros((3, 3, 1))], axis=-1)


class BratsDataset(Dataset):
    def __init__(self, src_dir, input_shape=(4, 224, 224), pad=10, **kwargs):
        self.src_dir = src_dir[:-1] if src_dir[-1] == "/" else src_dir
        self.__prepare_data()
        self.pad = pad

        # {"rotate": 30, "hflip": True, "vflip": True}
        self.transform_dict = kwargs


    def __prepare_data(self):
        self.data = glob.glob(self.src_dir + "/*.npz")
        print("Found {} instances in {}".format(len(self.data), self.src_dir))


    def collate_batch(self, batch):
        inputs, outputs = zip(*batch)
        return (torch.stack(inputs), torch.stack(outputs))


    def __rotate(self, x, degrees):
        """Rotates an input tensor by `degrees`"""
        return ndimage.rotate(x, degrees, reshape=False)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, ix):
        assert ix < self.__len__(), "Index OOB"

        # Establish a hook to the data.
        sampled_slice = self.data[ix]

        npz_ = np.load(sampled_slice)

        # Access the tensor using the _slice key.
        sample = npz_['_slice']

        # assert sample.shape == (224, 224, 5), "shape mismatch"

        t1 = sample[:,:,0]
        t2 = sample[:,:,1]
        t1ce = sample[:,:,2]
        flair = sample[:,:,3]

        op = np.uint8(sample[:,:,-1])
        
        wt = np.expand_dims((op > 0).astype(np.float64), axis=-1)
        tc = np.expand_dims(np.logical_or(op == 1, op == 4).astype(np.float64), axis=-1)
        et = np.expand_dims((op == 4).astype(np.float64), axis=-1)
        
        op = np.concatenate([wt, tc, et], axis=-1)

        # assert op.shape == (3, 224, 224)

        #################### Transformations ####################

        if "rotate" in self.transform_dict.keys() and self.transform_dict["rotate"]:
            angle = np.random.choice(np.linspace(0., self.transform_dict["rotate"]))
            t1 = self.__rotate(t1, angle)
            t2 = self.__rotate(t2, angle)
            t1ce = self.__rotate(t1ce, angle)
            flair = self.__rotate(flair, angle)
            op = self.__rotate(op, angle)

        if "hflip" in self.transform_dict.keys() and self.transform_dict["hflip"]:
            # Flip with a probability.
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=1)
                t2 = np.flip(t2, axis=1)
                t1ce = np.flip(t1ce, axis=1)
                flair = np.flip(flair, axis=1)
                op = np.flip(op, axis=1)
        
        if "vflip" in self.transform_dict.keys() and self.transform_dict["vflip"]:
            # Flip with a probability.
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=0)
                t2 = np.flip(t2, axis=0)
                t1ce = np.flip(t1ce, axis=0)
                flair = np.flip(flair, axis=0)
                op = np.flip(op, axis=0)

        # Mark a rough circle around the contour
        ### TODO: Skip the circle, make a square
        labels, nb = ndimage.label(op, structure=structure)
        unique = np.unique(labels).shape[0]
        # Find centroids
        # centroids = np.array(ndimage.measurements.center_of_mass(op, labels, [i for i in range(1, unique+1)]))
        min_positions = np.array(ndimage.minimum_position(op, labels, [i for i in range(1, unique+1)]))
        max_positions = np.array(ndimage.maximum_position(op, labels, [i for i in range(1, unique+1)]))

        '''
        centers = min_positions.copy()
        centers[:, :2] = (max_positions[:, :2] + min_positions[:, :2]) // 2
        radii = np.maximum(
            np.linalg.norm(centers[:, :2] - min_positions[:, :2], axis=1), 
            np.linalg.norm(centers[:, :2] - max_positions[:, :2], axis=1)
        )
        # Draw circles
        '''
        min_positions[:, :2] -= self.pad
        max_positions[:, :2] += self.pad
        mask = np.zeros_like(op)
        idx = np.arange(op.shape[0])
        mask[
            idx, 
            min_positions[:, 0]: max_positions[:, 0], 
            min_positions[:, 1]: max_positions[:, 1], 
            min_positions[:, 2]
        ] = 1.

        op = np.transpose(op, axis=(2, 0, 1))
        mask = np.transpose(mask, axis=(2, 0, 1))
                
        # (224, 224) => (1, 224, 224)
        t1 = torch.from_numpy(t1.copy()).unsqueeze(0)
        t2 = torch.from_numpy(t2.copy()).unsqueeze(0)
        t1ce = torch.from_numpy(t1ce.copy()).unsqueeze(0)
        flair = torch.from_numpy(flair.copy()).unsqueeze(0)
        op = torch.from_numpy(op.copy())
        mask = torch.from_numpy(mask.copy())
        
        # ((4, 224, 224), (3, 224, 224))
        # Return a tuple of two elements: a tuple of inputs and the output tensor.
        return (torch.cat([t1, t2, t1ce, flair], dim=0), op, mask)

"""
This script will create the custom DataLoader for
the BRATS2020 dataset (created using create_data.py and preprocess.py).
"""

import os
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy import ndimage

import cv2
np.random.seed(42)

def random_morph(op, phase, p=0.5):
    
    if phase != 'train':
        return op
    
    # With a probability of 0.5, dilate of erode
    if np.random.random() > p:
        kernel_h = int(np.random.choice([3, 5, 7], 1))
        kernel_w = int(np.random.choice([3, 5, 7], 1))
        
        kernel = np.ones((kernel_h, kernel_w))
#         if np.random.random() > 0.5:
#             # Dilate
#             op = cv2.dilate(op, kernel)
#         else:
#             # Erode
#             op = cv2.erode(op, kernel)
        op = cv2.dilate(op, kernel)
    return op

class BratsDataset(Dataset):
    def __init__(self, src_dir, phase='train', **kwargs):
        self.src_dir = src_dir[:-1] if src_dir[-1] == "/" else src_dir
        print("Brats dataloader invoked")
        self.__prepare_data()
        
        self.phase = phase

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
        
        wt = np.uint8(op > 0)
        wt = random_morph(wt, self.phase) > 0
        tc = np.uint8(np.logical_or(op == 1, op == 4))
        tc = random_morph(tc, self.phase) > 0
        et = np.uint8(op == 4)
        et = random_morph(et, self.phase) > 0
        
        wt = np.expand_dims(wt.astype(np.float64), axis=-1)
        tc = np.expand_dims(tc.astype(np.float64), axis=-1)
        et = np.expand_dims(et.astype(np.float64), axis=-1)
        
        op_tensor = np.concatenate([wt, tc, et], axis=-1)
        op = op_tensor
        
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
        op = np.transpose(op, axes=(2, 0, 1))
                
        # (224, 224) => (1, 224, 224)
        t1 = torch.from_numpy(t1.copy()).unsqueeze(0)
        t2 = torch.from_numpy(t2.copy()).unsqueeze(0)
        t1ce = torch.from_numpy(t1ce.copy()).unsqueeze(0)
        flair = torch.from_numpy(flair.copy()).unsqueeze(0)
        op = torch.from_numpy(op.copy())
        
        # ((4, 224, 224), (3, 224, 224))
        # Return a tuple of two elements: a tuple of inputs and the output tensor.
        return (torch.cat([t1, t2, t1ce, flair], dim=0), op)

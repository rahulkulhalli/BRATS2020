"""
This script will create the custom DataLoader for
the BRATS2020 prediction. The data of one patient at a time is
returned.
"""

import os
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy import ndimage
import nibabel as nib

RESIZE_TARGET = 128

def resize(x, target=RESIZE_TARGET):
    """
    resize is actually just center crop lol got'em
    """
    c_x, c_y = x.shape[2]//2, x.shape[3]//2
    crop_offset = target//2
    
    return x[:, :, c_x-crop_offset:c_x+crop_offset, c_y-crop_offset:c_y+crop_offset]


def preprocess(x):
    
    min_val = np.nanmin(x, axis=(2, 3), keepdims=True)
    max_val = np.nanmax(x, axis=(2, 3), keepdims=True)
    
    return (x - min_val) / (max_val - min_val + 1e-8)


class BratsDataset(Dataset):
    def __init__(self, src_dir, index=False, **kwargs):
        '''
        param src_dir: source directory of data
        param index: index patient ids with location in the list for direct specific retrieval
        '''
        self.src_dir = src_dir[:-1] if src_dir[-1] == "/" else src_dir
        print("Brats predcition dataloader invoked")
        self.__prepare_data(index)

        self.transform_dict = kwargs


    def __prepare_data(self, index=False):
        self.data = glob.glob(self.src_dir + "/BraTS20*")
        print("Found {} instances in {}".format(len(self.data), self.src_dir))
        if index:
            self.index = {data.split(os.path.sep)[-1]: i for i, data in enumerate(self.data)}
        else:
            self.index = None


    def collate_batch(self, batch):
        X, affine, pid = zip(*batch)
        return X[0], affine[0], pid[0]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, ix):
        assert ix < self.__len__(), "Index OOB"

        # Establish a hook to the data.
        pid_path = self.data[ix]
        pid = pid_path.split(os.path.sep)[-1]
        
        # Get all nii gz objects
        t1_gz = nib.load(os.path.join(pid_path, '{}_t1.nii.gz'.format(pid)))
        t2_gz = nib.load(os.path.join(pid_path, '{}_t2.nii.gz'.format(pid)))
        t1ce_gz = nib.load(os.path.join(pid_path, '{}_t1ce.nii.gz'.format(pid)))
        flair_gz = nib.load(os.path.join(pid_path, '{}_flair.nii.gz'.format(pid)))
        
        # Get the affine matrix, we will need this while saving the output
        affine = t1_gz.affine
        
        # Get arrays
        t1 = np.expand_dims(t1_gz.get_fdata().astype(np.float32), axis=0)
        t2 = np.expand_dims(t2_gz.get_fdata().astype(np.float32), axis=0)
        t1ce = np.expand_dims(t1ce_gz.get_fdata().astype(np.float32), axis=0)
        flair = np.expand_dims(flair_gz.get_fdata().astype(np.float32), axis=0)
        
        # Concat
        X = np.concatenate([t1, t2, t1ce, flair], axis=0)
        # Permute so that slices are elements of a batch
        X = resize(np.transpose(X, axes=(3, 0, 1, 2)))
        
        # Preprocess
        X = preprocess(X)
        
        input_volume = torch.from_numpy(X.copy())
        
        # Return a tuple of two elements: a tuple of inputs and the output tensor.
        return (input_volume, affine, pid)

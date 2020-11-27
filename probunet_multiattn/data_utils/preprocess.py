"""
This script reads the MRI volumes (split using create_data.py)
and performs resizing, per-channel normalization, and saves the
channel-unrolled images.
"""

import os
import shutil
# from PIL import Image
import numpy as np
from scipy import ndimage
import nibabel

# Globals
ratio = 224/240
flair_format = "BraTS20_Training_{}_flair.nii.gz"
t1ce_format = "BraTS20_Training_{}_t1ce.nii.gz"
t1_format = "BraTS20_Training_{}_t1.nii.gz"
t2_format = "BraTS20_Training_{}_t2.nii.gz"
seg_format = "BraTS20_Training_{}_seg.nii.gz"
RESIZE_TARGET = 128


def resize(x, target=RESIZE_TARGET):
    """
    resize is actually just center crop lol got'em
    """
    c_x, c_y = x.shape[0]//2, x.shape[1]//2
    crop_offset = target//2
    
    return x[c_x-crop_offset:c_x+crop_offset, c_y-crop_offset:c_y+crop_offset]


def create_slices(x_src, normalize_each=True):
    
    # Convert the .nii.gz file to a numpy tensor.
    x = nibabel.load(x_src).get_fdata().astype(np.float32)

    buffer = []
    for z_dim in range(x.shape[-1]):
        resized_slice = resize(x[:,:,z_dim])

        assert resized_slice.shape == (RESIZE_TARGET, RESIZE_TARGET)

        if normalize_each:
            resized_slice = (resized_slice - np.nanmin(resized_slice))/(np.nanmax(resized_slice) - np.nanmin(resized_slice) + 1e-8)
            assert resized_slice.min() >= 0. and resized_slice.max() <= 1., "Scaling issue..."

        # Append as (RESIZE_TARGET, RESIZE_TARGET, 1)
        buffer.append(resized_slice[:,:,np.newaxis])
    
    return buffer


def create_slices_v2(list_of_slices):
    # Initial: All slices were considered
    # v2: When y==0, ignore the slices
    
    valid_ix = []
    seg = list_of_slices[-1]
    
    y = nibabel.load(seg).get_fdata().astype(np.float32)
    y_valid = []
    
    for z_dim in range(y.shape[-1]):
        if not np.all(y[:,:,z_dim] == 0.):
            
            valid_ix.append(z_dim)
            
            # (RESIZE_TARGET, RESIZE_TARGET)
            y_slice = resize(y[:,:,z_dim])
            assert y_slice.shape == (RESIZE_TARGET, RESIZE_TARGET)
            
            # (RESIZE_TARGET, RESIZE_TARGET, 1)
            y_valid.append(y_slice[..., np.newaxis])
    
    # y_valid will be (RESIZE_TARGET, RESIZE_TARGET, m)
    
    def _slice(x_src, mask):
        x = nibabel.load(x_src).get_fdata().astype(np.float32)
        
        # mask-out the irrelevant slices.
        # make x to be (RESIZE_TARGET, RESIZE_TARGET, m)
        x = x[...,mask]
        
        buffer = []
        for z_dim in range(x.shape[-1]):
            resized_slice = resize(x[:,:,z_dim])

            assert resized_slice.shape == (RESIZE_TARGET, RESIZE_TARGET)

            resized_slice = (resized_slice - np.nanmin(resized_slice))/(np.nanmax(resized_slice) - np.nanmin(resized_slice) + 1e-8)

            # Append as (RESIZE_TARGET, RESIZE_TARGET, 1)
            buffer.append(resized_slice[:,:,np.newaxis])

        return buffer
    
    
    # Not looping for sake of ease.
    t1 = _slice(list_of_slices[0], valid_ix)
    t2 = _slice(list_of_slices[1], valid_ix)
    t1ce = _slice(list_of_slices[2], valid_ix)
    flair = _slice(list_of_slices[3], valid_ix)
    
    stacked_inputs = [stack_images([t1[ix], t2[ix], \
                t1ce[ix], flair[ix], y_valid[ix]]) for ix in range(len(valid_ix))]
    
    # print([len(x) for x in [t1, t2, t1ce, flair, y_valid]])
    
    return stacked_inputs


def stack_images(slices):

    stack = np.concatenate(slices, axis=-1)

    assert stack.shape == (RESIZE_TARGET, RESIZE_TARGET, 5), "nyoope"

    return stack


def get_unique_patients(src_dir):
    return list(set([f.split("_")[2] for f in os.listdir(src_dir) if f.endswith("nii.gz")]))


def isValid(id, src_dir):
    return all([f.format(id) in os.listdir(src_dir) for f in [t1_format, t2_format, t1ce_format, \
        flair_format, seg_format]])


def export_slices(slice_list, patient_id, dst_dir):
    
    """Iterate over every slice, append a slice index to the file, and copy to dst."""
    
    # e.g. filename = "BraTS20_Training_{}_flair.nii.gz"

    for slice_ix, _slice in enumerate(slice_list):
        slice_name = patient_id + "_slice_" + ("%03d" % slice_ix) + ".npz"
        np.savez_compressed(os.path.join(dst_dir, slice_name), _slice=_slice)
    
    print("Exported {} slices for ID {}".format(len(slice_list), patient_id))


if __name__ == "__main__":

    data_root = "/home/antpc/Documents/data_mount/BRATS2020/data"
    
    # resized_target_dir = "unrolled"
    target_dir = "r128x128"

    training_IDs = get_unique_patients(os.path.join(data_root, "train"))
    print("Found {} instances in train".format(len(training_IDs)))

    testing_IDs = get_unique_patients(os.path.join(data_root, "test"))
    print("Found {} instances in test".format(len(testing_IDs)))

    for split_type in ["train", "test"]:
        
        print("{} ->".format(split_type))
        
        dir_path = os.path.join(data_root, split_type, target_dir)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print("Target directory {} created".format(target_dir))

        ids = training_IDs if split_type == "train" else testing_IDs

        for id in ids:
            # We want to normalize all inputs, not the output.

            t1_file = os.path.join(data_root, split_type, t1_format.format(id))
            t2_file = os.path.join(data_root, split_type, t2_format.format(id))
            t1ce_file = os.path.join(data_root, split_type, t1ce_format.format(id))
            flair_file = os.path.join(data_root, split_type, flair_format.format(id))
            seg_file = os.path.join(data_root, split_type, seg_format.format(id))

            # [(224, 224, 1) * 155]
            # v2: Changing the method signature to accomodate a list as an input.
            #t1_slices = create_slices(t1_file)
            #t2_slices = create_slices(t2_file)
            #t1ce_slices = create_slices(t1ce_file)
            #flair_slices = create_slices(flair_file)
            #seg_slices = create_slices(seg_file, normalize_each=False)
            
            #v2:
            stacked_inputs = create_slices_v2([t1_file, t2_file, t1ce_file, flair_file, seg_file])

            # [(224, 224, 5) * 155]
            # stacked_inputs = [stack_images([t1_slices[ix], t2_slices[ix], \
            #    t1ce_slices[ix], flair_slices[ix], seg_slices[ix]]) for ix in range(155)]

            export_slices(stacked_inputs, id, os.path.join(data_root, split_type, target_dir))

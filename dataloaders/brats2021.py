import sys
sys.path.append('./code/utils')
import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from random import sample
import random
from skimage import exposure
import SimpleITK as sitk

class ConvertToMultiChannelBasedOnBratsClassesd(object):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, sample):

        label = sample['label']
        image = sample['image']
        result = []
        # merge label 4 and label 1 to construct TC
        result.append(np.where(((label == 4) | (label == 1)), 1, 0))
            # merge labels 2, 4 and 1 to construct WT
        result.append(np.where(label>0, 1, 0))
            # label 4 is ET
        result.append(np.where(label==4, 1, 0))
        label = np.stack(result, axis=0)
        return {'image': image, 'label': label}

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape) #1,128,128,128

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class BraTS2021(Dataset):
    """ BraTS2021 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, select = None, need_brain_mask = True, if_blank_fill=True):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.select = select
        train_path = self._base_dir+'/train_path_list.txt'
        test_path = self._base_dir+'/test_path_list.txt'
        pretrain_path = self._base_dir+'/pretrain_path_list.txt'
        self.split = split
        self.need_brain_mask = need_brain_mask
        self.blank_fill = if_blank_fill
        


        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[:num]

        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'train_pretrain':
            with open(pretrain_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[:num]
        elif split == 'val_pretrain':
            with open(pretrain_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[num:]


        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}/{}h5file.h5".format(image_name, image_name), 'r')
        label = h5f['seg'][:]
        img_list = []
        if self.blank_fill:
            for i in self.select:
                if i is not None:
                    img_list.append(h5f[i][:])
                    image_shape = h5f[i][:].shape
                    if self.need_brain_mask:
                        if i == 'flair':
                            brain_mask = np.array([np.where(h5f['flair'][:]>np.min(h5f['flair'][:]),1,0)])
                else:
                    img_list.append('None')

            for i in range(len(img_list)):
                if isinstance(img_list[i],str):
                    img_list[i] = np.zeros(shape=image_shape)

        else:
            for i in self.select:
                img_list.append(h5f[i][:])

        if self.need_brain_mask:
            sample = {'image': np.stack(img_list, axis=0), 'brain': brain_mask} if 'pretrain' in self.split else  {'image': np.stack(img_list, axis=0), 'label': label.astype(np.uint8)}
        else:
            sample = {'image': np.stack(img_list, axis=0)} if 'pretrain' in self.split else  {'image': np.stack(img_list, axis=0), 'label': label.astype(np.uint8)}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class RandPatchPuzzle(object):

    def __init__(self, patch_size=[16,16,16], input_sizes=[128,128,128]):
        self.patch_size = patch_size
        self.input_sizes = input_sizes
    # 输入图像：CWHD，Numpy格式
    def __call__(self, sample):
        if 1==1:
            image = sample['image'].numpy()
            
            # pad the sample if necessary
            if (image.shape[1] < self.input_sizes[0]) or (image.shape[2] < self.input_sizes[1]) or (image.shape[3] < \
                    self.input_sizes[2]):
                
                pw = (self.input_sizes[0] - image.shape[1]) 
                ph = (self.input_sizes[1] - image.shape[2])  
                pd = (self.input_sizes[2] - image.shape[3]) 
                pw0 = pw // 2
                ph0 = ph // 2
                pd0 = pd // 2
            
                image = np.pad(image, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='minimum')
                
            (c, w, h, d) = image.shape

            patches_set = [[i,j,k] for i in np.arange(0, image.shape[1], self.patch_size[0]) for j in np.arange(0, image.shape[2], self.patch_size[1]) for k in np.arange(0, image.shape[3] ,self.patch_size[2])]
            random.shuffle(patches_set)
            sn = int(0.25 * len(patches_set))
            select_patch_t1 = patches_set[0:sn]
            select_patch_t1ce = patches_set[sn:2*sn]
            select_patch_t2 = patches_set[2*sn:3*sn]
            select_patch_flair = patches_set[3*sn:4*sn]

            w1 = self.patch_size[0]
            h1 = self.patch_size[1]
            d1 = self.patch_size[2]
            
            mix_image = np.zeros(shape=(c,w,h,d))
            
            for p in range(sn):
                mix_image[0, select_patch_t1[p][0]:select_patch_t1[p][0]+w1, select_patch_t1[p][1]:select_patch_t1[p][1]+h1, select_patch_t1[p][2]:select_patch_t1[p][2]+d1] = 1
                mix_image[1, select_patch_t1ce[p][0]:select_patch_t1ce[p][0]+w1, select_patch_t1ce[p][1]:select_patch_t1ce[p][1]+h1, select_patch_t1ce[p][2]:select_patch_t1ce[p][2]+d1] = 1
                mix_image[2, select_patch_t2[p][0]:select_patch_t2[p][0]+w1, select_patch_t2[p][1]:select_patch_t2[p][1]+h1, select_patch_t2[p][2]:select_patch_t2[p][2]+d1] = 1
                mix_image[3, select_patch_flair[p][0]:select_patch_flair[p][0]+w1, select_patch_flair[p][1]:select_patch_flair[p][1]+h1, select_patch_flair[p][2]:select_patch_flair[p][2]+d1] = 1

            mix = image * mix_image

            sample = {'image': image, 'mix_image': mix}
            
            
        return sample
    
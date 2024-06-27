import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib
import SimpleITK as sitk
import glob
import os
import h5py
import json
import random

def brain_bbox(data_t1, data_t1ce, data_t2, data_flair, gt, status="train"):
    mask = (data_t1 != 0)
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    t1_bboxed = data_t1[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    if data_t1ce is not None:
        t1ce_bboxed = data_t1ce[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        t1ce_bboxed = None
    if data_t2 is not None:
        t2_bboxed = data_t2[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        t2_bboxed = None
    if data_flair is not None:
        flair_bboxed = data_flair[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        flair_bboxed = None
    if status=='train':
        gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
        return t1_bboxed, t1ce_bboxed, t2_bboxed, flair_bboxed, gt_bboxed
    if status=='test':
        tmp_t1 = np.zeros_like(data_t1)
        tmp_t1ce = np.zeros_like(data_t1ce)
        tmp_t2 = np.zeros_like(data_t2)
        tmp_flair = np.zeros_like(data_flair)
        return t1_bboxed, t1ce_bboxed, t2_bboxed, flair_bboxed, tmp_t1, tmp_t1ce, tmp_t2, tmp_flair, [minZidx,maxZidx,minXidx,maxXidx,minYidx,maxYidx]

def volume_bounding_box(data, gt, expend=0, status="train"):
    data, gt = brain_bbox(data, gt)
    print(data.shape)
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    z, x, y = data.shape
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    minZidx_jitterd = max(minZidx - expend, 0)
    maxZidx_jitterd = min(maxZidx + expend, z)
    minXidx_jitterd = max(minXidx - expend, 0)
    maxXidx_jitterd = min(maxXidx + expend, x)
    minYidx_jitterd = max(minYidx - expend, 0)
    maxYidx_jitterd = min(maxYidx + expend, y)

    data_bboxed = data[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
    print([minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx])
    print([minZidx_jitterd, maxZidx_jitterd,
           minXidx_jitterd, maxXidx_jitterd, minYidx_jitterd, maxYidx_jitterd])

    if status == "train":
        gt_bboxed = np.zeros_like(data_bboxed, dtype=np.uint8)
        gt_bboxed[expend:maxZidx_jitterd-expend, expend:maxXidx_jitterd -
                  expend, expend:maxYidx_jitterd - expend] = 1
        return data_bboxed, gt_bboxed

    if status == "test":
        gt_bboxed = gt[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
        return data_bboxed, gt_bboxed


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out = out.astype(np.float32)
    return out


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        high_boundary = cdf[1][cdf[0] <= (1-self.percent)][-1]
        low_boundary = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, low_boundary, high_boundary)

data_base = "your_data_path"

sampleid = os.listdir(data_base)

i = 1
save_base = 'your_save_path'
ftrain = open(os.path.join(save_base, 'train_path_list.txt'), 'w')
ftest = open(os.path.join(save_base, 'test_path_list.txt'), 'w')
fval = open(os.path.join(save_base, 'val_path_list.txt'), 'w')

train_list = random.sample(sampleid,k=int(len(sampleid)*0.6))
atlas2_tmp = [x for x in sampleid if x not in train_list]
val_list = random.sample(atlas2_tmp,k=int(len(sampleid)*0.1))
test_list = [x for x in atlas2_tmp if x not in val_list]

img_for_message = sitk.ReadImage('./BraTS2021_00006/BraTS2021_00006_t1_MNI.nii.gz')
ori = img_for_message.GetOrigin()
dir = img_for_message.GetDirection()
spa = img_for_message.GetSpacing()

for p in sampleid:
    data_t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_base,p,'brain.nii.gz')))
    data_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_base, p, p+'_t1ce.nii.gz')))
    data_flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_base, p, p+'_flair.nii.gz')))
    data_t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_base, p, p+'_t2.nii.gz')))
    lab = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_base,p,'seg.nii.gz')))

    img_t1,img_t1ce,img_t2,img_flair,img_lab = brain_bbox(data_t1, data_t1ce, data_t2, data_flair, lab, 'train')
    
    img_t1 = MedicalImageDeal(img_t1, percent=0.01).valid_img
    img_t1ce = MedicalImageDeal(img_t1ce, percent=0.01).valid_img
    img_t2 = MedicalImageDeal(img_t2, percent=0.01).valid_img
    img_flair = MedicalImageDeal(img_flair, percent=0.01).valid_img

    img_t1 = itensity_normalize_one_volume(img_t1)
    img_t1ce = itensity_normalize_one_volume(data_t1ce)
    img_t2 = itensity_normalize_one_volume(data_t2)
    img_flair = itensity_normalize_one_volume(data_flair)
    

    uid = p

    if not os.path.exists(os.path.join(save_base,uid)):
        os.mkdir(os.path.join(save_base,uid))
    save_base_temp = os.path.join(save_base,uid)

    if p in train_list:
        ftrain.write(uid + "\n")
        
    if p in test_list:
        ftest.write(uid + "\n")
        
    if p in val_list:
        fval.write(uid + "\n")
        
    h5file = h5py.File(os.path.join(save_base_temp,uid+'h5file.h5'), 'w')
    h5file['t1'] = img_t1
    h5file['t1ce'] = img_t1ce
    h5file['t2'] = img_t2
    h5file['flair'] = img_flair
    h5file['seg'] = img_lab
    h5file.close()

    img_t1=sitk.GetImageFromArray(img_t1)
    img_t1.SetOrigin(ori)
    img_t1.SetDirection(dir)
    img_t1.SetSpacing(spa)
    sitk.WriteImage(img_t1, os.path.join(save_base_temp,uid+'_t1.nii.gz'))
    img_t1ce=sitk.GetImageFromArray(img_t1ce)
    img_t1ce.SetOrigin(ori)
    img_t1ce.SetDirection(dir)
    img_t1ce.SetSpacing(spa)
    sitk.WriteImage(img_t1ce, os.path.join(save_base_temp,uid+'_t1ce_MNI.nii.gz'))
    img_t2=sitk.GetImageFromArray(img_t2)
    img_t2.SetOrigin(ori)
    img_t2.SetDirection(dir)
    img_t2.SetSpacing(spa)
    sitk.WriteImage(img_t2, os.path.join(save_base_temp,uid+'_t2_MNI.nii.gz'))
    img_flair=sitk.GetImageFromArray(img_flair)
    img_flair.SetOrigin(ori)
    img_flair.SetDirection(dir)
    img_flair.SetSpacing(spa)
    sitk.WriteImage(img_flair, os.path.join(save_base_temp,uid+'_flair_MNI.nii.gz'))
    img_lab=sitk.GetImageFromArray(img_lab)
    img_lab.SetOrigin(ori)
    img_lab.SetDirection(dir)
    img_lab.SetSpacing(spa)
    sitk.WriteImage(img_lab, os.path.join(save_base_temp,uid+'_lab.nii.gz'))
    print(str(i)+':'+uid)
    i = i + 1
        
ftrain.close()
ftest.close()
fval.close()


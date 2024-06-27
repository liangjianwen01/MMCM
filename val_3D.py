import math
from glob import glob
import os
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

def ConvertToMultiChannelBasedOnBratsClassesd(label):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    
    result = []
    # merge label 4 and label 1 to construct TC
    result.append(np.where(((label == 4) | (label == 1)), 1, 0))
    # merge labels 2, 4 and 1 to construct WT
    result.append(np.where(label>0, 1, 0))
    # label 4 is ET
    result.append(np.where(label==4, 1, 0))
    label = np.stack(result, axis=0)
    return label

def one_hot_encoder(input_tensor,n_classes):
        input_tensor = torch.tensor(input_tensor).unsqueeze(0)
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=0)
        return output_tensor.float()

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, need_sig=True, need_argmax=False):
    c, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(0,0), (wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    cc, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    if need_sig:
                        logit = torch.nn.Sigmoid()
                    elif need_argmax:
                        logit = torch.nn.Softmax(dim=1)
                    else:
                        logit = torch.nn.Identity()
                    y = logit(y1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    if need_argmax:
        label_map = np.argmax(score_map,axis=0)
        label_map = one_hot_encoder(label_map,num_classes).numpy()
    else:
        label_map = np.where(score_map>=0.5, 1, 0)

    if add_pad:
        label_map = label_map[:, wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=3, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, select=['t1','t1ce','t2','flair'], val_num=None, if_blank_fill=True, need_sig=True):
    with open(os.path.join(base_dir, test_list), 'r') as f:
        image_list = f.readlines()
    if val_num is not None:
        image_list = image_list[:val_num]
    image_list = [base_dir + "/{}/{}h5file.h5".format(
        item.replace('\n', '').split(",")[0], item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = []
        if if_blank_fill:
            for i in select:
                if i is not None:
                    image.append(h5f[i][:])
                    image_shape = h5f[i][:].shape
                else:
                    image.append('None')
            for i in range(len(image)):
                if isinstance(image[i],str):
                    image[i] = np.zeros(shape=image_shape)
        else:
            for i in select:
                if i is not None:
                    image.append(h5f[i][:])

        image = np.stack(image,axis=0)
        label = h5f['seg'][:]
        label = ConvertToMultiChannelBasedOnBratsClassesd(label=label)
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, need_sig = need_sig)
        for i in range(num_classes):
            total_metric[i, :] += cal_metric(label[i,:,:,:] == 1, prediction[i,] == 1)
    print("Validation end")
    return total_metric / len(image_list)

def test_all_case_one(net, base_dir, test_list="full_test.list", num_classes=1, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, select=['t1',None,None,None], val_num=None, if_blank_fill=True, need_sig=True):
    with open(os.path.join(base_dir, test_list), 'r') as f:
        image_list = f.readlines()
    if val_num is not None:
        image_list = image_list[:val_num]
    image_list = [base_dir + "/{}/{}h5file.h5".format(
        item.replace('\n', '').split(",")[0], item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = []
        if if_blank_fill:
            for i in select:
                if i is not None:
                    image.append(h5f[i][:])
                    image_shape = h5f[i][:].shape
                else:
                    image.append('None')
            for i in range(len(image)):
                if isinstance(image[i],str):
                    image[i] = np.zeros(shape=image_shape)
        else:
            for i in select:
                if i is not None:
                    image.append(h5f[i][:])

        image = np.stack(image,axis=0)
        label = h5f['seg'][:]
        
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, need_sig = need_sig)
        for i in range(num_classes):
            total_metric[i, :] += cal_metric(label == 1, prediction[i,] == 1)
    print("Validation end")
    return total_metric / len(image_list)

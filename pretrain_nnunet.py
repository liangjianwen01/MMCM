import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import random
import shutil
import sys
import time
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2021 import (BraTS2021, RandPatchPuzzle)
from model.net import UNet_LY
from val_3D import test_all_case
import monai.transforms as monai_transforms
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data_path', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTs2021_Pre_Train', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='nnUNet_LY', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=800*450, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--sw_batch', type=int, default=2,
                    help='batch_size per sample')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=3e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--split_patch_size', type=list,  default=[16, 16, 16],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=900,
                    help='labeled data')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--resume_path', default=None, type=str, help='retrain path')

args = parser.parse_args()

def PL2Loss(input_tensor, output_tensor, brain_tensor):
    L2_calculator = nn.MSELoss()
    l2 = 0
    b,c,w,h,d = input_tensor.shape
    
    for i in range(b):   
        brain = brain_tensor[i,].repeat(1,4,1,1).reshape(16,128,128,128) #need
        l2 = l2 + L2_calculator(input_tensor[i,][torch.where(brain==1)], output_tensor[i,][torch.where(brain==1)])
    l2 = l2/b
    return l2

def train(args, snapshot_path):
    iter_num = 0
    start_epoch = 0
    loss_epoch = []
    best_loss = 1e8

    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet_LY(in_chns=4, class_num=num_classes, pretrain=True, reconstruct=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=base_lr)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    
    transform = monai_transforms.Compose(
        [   monai_transforms.RandZoomd(
                keys=["image","brain"],
                prob=0.8,
                max_zoom=1.4,
                min_zoom=0.7,
                mode=("bilinear","nearest"),
            ),
            monai_transforms.RandSpatialCropSamplesd(
                keys=["image","brain"],
                roi_size=(args.patch_size[0],args.patch_size[1],args.patch_size[2]),
                num_samples=args.sw_batch,
                random_size = False
            ),
            monai_transforms.RandRotated(
                range_x=0.5236,
                range_y=0.5236,
                range_z=0.5236,
                keys=["image","brain"],
                mode=("bilinear","nearest"),
                prob=0.8
            ),
            RandPatchPuzzle(input_sizes=args.patch_size, patch_size=args.split_patch_size),
            monai_transforms.ToTensord(keys=["image","mix_image"], track_meta=False),
        ]
    )
    db_train = BraTS2021(base_dir=train_data_path,
                         split='train_pretrain',
                         num=args.labeled_num,
                         select=['t1','t1ce','t2','flair'],
                         transform=transform,
                         need_brain_mask=True)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    if args.resume_path is not None:
        iter_num = int(args.resume_path.split('_')[9])+1
        load_dict = torch.load(args.resume_path)
        start_epoch = load_dict['epoch']
        model.load_state_dict(load_dict['model'])
        optimizer.load_state_dict(load_dict['optimizer'])
        loss_epoch = load_dict['loss']

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=800)
    elif args.lrschedule == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=0.9)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
    else:
        scheduler = None

    if (args.resume_path is not None)&(scheduler is not None):
            scheduler.step(epoch=start_epoch)
    

    model.train()

    
    reconstruct_loss = nn.MSELoss()
    
    
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    max_epoch = max_iterations // len(trainloader) + 1
    
    iterator = tqdm(range(start_epoch+1,max_epoch), ncols=70)
    
    for epoch_num in iterator:
        loss_per = 0

        for i_batch, sampled_batch in enumerate(trainloader):

            
            if isinstance(sampled_batch, list):
                mix_volume_batch = [x['mix_image'][j,] for x in sampled_batch for j in range(args.batch_size)]
                volume_batch = [x['image'][j,] for x in  sampled_batch for j in range(args.batch_size)]
                mix_volume_batch = torch.stack(mix_volume_batch, dim=0).float()
                volume_batch = torch.stack(volume_batch, dim=0).float()
                
            else:
                volume_batch, mix_volume_batch =  sampled_batch['image'], sampled_batch['mix_image']
                
            
                

            volume_batch, mix_volume_batch = volume_batch.cuda(), mix_volume_batch.cuda()
            
            
            
            outputs_reconstruct = model(mix_volume_batch)

            loss_reconstruct = 0

            
            loss_reconstruct = reconstruct_loss(outputs_reconstruct, volume_batch)
                        
            loss = loss_reconstruct 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_per += loss.item()

            logging.info(
                'iteration %d : loss : %f' % (iter_num, loss.item()))
            
            iter_num = iter_num + 1
           
            


        loss_per /= len(trainloader)
        
        loss_epoch.append(loss_per)
        
        logging.info(
                'epoch %d : loss : %f' % (epoch_num))
           
        scheduler.step()
        
            
        if epoch_num > 0 and (epoch_num + 1) % 50 == 0:
            save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_loss_{}.pth'.format(
                                                    iter_num-1, round(loss_per, 3)))
                    
            torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch_num, "loss":loss_epoch}, save_mode_path)

        if iter_num >= max_iterations:
            break


        if iter_num >= max_iterations:
            iterator.close()
            break
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "save_path/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

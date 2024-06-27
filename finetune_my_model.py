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
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2021 import (BraTS2021, ConvertToMultiChannelBasedOnBratsClassesd)
from model.net import Unet_LY
from utils import losses
from val_3D import test_all_case
import monai.transforms as monai_transforms
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data_path', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTs2021_Finetune', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='nnUNet_LY', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=210*300, help='maximum epoch number to train') 
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=120,
                    help='labeled data') 
parser.add_argument('--labeled_num_val', type=int, default=40,
                    help='labeled data') 
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--pretrain_path', type=str,
                    default='/mnt/liangjw/MMCM/pretrained_model/BraTS2021_Pretrain_Model.pth', help='Name of Experiment') 
parser.add_argument('--resume_path', type=str,
                    default=None, help='Name of Experiment') 
args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet_LY(in_chns=4, class_num=num_classes).to(device) 
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    iter_num = 0

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.resume_path is None:
        if args.pretrain_path is not None:
            model_dict = model.state_dict()
            load_model_dict = torch.load(args.pretrain_path)['model']
            for k in list(load_model_dict.keys()):
                if 'mix_num' in k:
                    del load_model_dict[k]
            model_dict.update(load_model_dict)
            model.load_state_dict(model_dict)
    else:
        load_model = torch.load(args.resume_path)
        model.load_state_dict(load_model['model'])


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    transform = monai_transforms.Compose(
        [   ConvertToMultiChannelBasedOnBratsClassesd(),
            monai_transforms.SpatialPadd(keys=["image", "label"],
                                         spatial_size=(128,128,128),
                                         mode='edge'),
            monai_transforms.RandZoomd(
                keys=["image", "label"],
                prob=0.2,
                max_zoom=1.4,
                min_zoom=0.7,
                mode=("bilinear", "nearest"),
            ),
            monai_transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128,128,128),
                pos=1,
                neg=0.5,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            monai_transforms.RandRotated(
                range_x=0.5236,
                range_y=0.5236,
                range_z=0.5236,
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.2
            ),
            monai_transforms.RandAdjustContrastd(keys="image",
                                           prob=0.3,
                                           gamma=(0.7, 1.5)),    
            monai_transforms.ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )
    db_train = BraTS2021(base_dir=train_data_path,
                         split='train',
                         num=args.labeled_num,
                         select=['t1','t1ce','t2','flair'],
                         transform=transform,
                         need_brain_mask=False,
                         if_blank_fill=True) 
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=6, pin_memory=True, worker_init_fn=worker_init_fn)
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=base_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    if args.resume_path is not None:
        optimizer.load_state_dict(load_model['optimizer'])
        iter_num = int(args.resume_path.split('_')[9].split('.')[0])
        start_epoch = int((iter_num+1) / len(trainloader))-1
    else:
        start_epoch = -1

    
    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=525)
    elif args.lrschedule == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=0.9)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
    else:
        scheduler = None

    if (args.resume_path is not None) and (args.lrschedule is not None):
        scheduler.step(start_epoch)

    

    model.train()

    bce_loss_tc = BCEWithLogitsLoss()
    bce_loss_wt = BCEWithLogitsLoss()
    bce_loss_et = BCEWithLogitsLoss()
    dice_loss = losses.DiceLoss(3,per_class=True)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    
    max_epoch = max_iterations // len(trainloader) 
    best_performance = 0.0
    iterator = tqdm(range(start_epoch+1, max_epoch), ncols=70)
    to_logits = nn.Sigmoid()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            sampled_batch = sampled_batch
            if isinstance(sampled_batch, list):
                label_batch = [x['label'].squeeze() for x in  sampled_batch]
                volume_batch = [x['image'].squeeze() for x in  sampled_batch]
                label_batch = torch.stack(label_batch, dim=0)
                volume_batch = torch.stack(volume_batch, dim=0)
            else:
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            # outputs_soft = torch.softmax(outputs, dim=1)
            outputs_sigmoid = to_logits(outputs)
            # loss_ce = ce_loss(outputs, label_batch)
            loss_dice, loss_dice_perclass = dice_loss(outputs_sigmoid, label_batch)
            loss_bce_tc = bce_loss_tc(outputs[:,0,:,:,:], label_batch[:,0,:,:,:].float())
            loss_bce_wt = bce_loss_wt(outputs[:,1,:,:,:], label_batch[:,1,:,:,:].float())
            loss_bce_et = bce_loss_et(outputs[:,2,:,:,:], label_batch[:,2,:,:,:].float())
            loss_ce = (loss_bce_tc + loss_bce_wt + loss_bce_et)/3
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            iter_num = iter_num + 1
            

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, et_ce: %f, wt_ce: %f, tc: %f, loss_dice: %f, tc: %f, wt: %f, et: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_bce_tc.item(), loss_bce_wt.item(), loss_bce_et.item(), loss_dice.item(),
                 loss_dice_perclass[0], loss_dice_perclass[1], loss_dice_perclass[2]))
           

            if (iter_num > 0) and ((iter_num+1) % 2100) == 0: 
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val_path_list.txt", num_classes=3, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64, select=['t1','t1ce','t2','flair'], val_num=args.labeled_num_val, if_blank_fill=True, need_sig=True) 
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':int((iter_num+1) / len(trainloader))-1}, save_mode_path)
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':int((iter_num+1) / len(trainloader))-1}, save_best)

                logging.info(
                    'iteration %d : dice_score : %f dice_tc : %f dice_wt : %f dice_et : %f  hd95 : %f' % (iter_num, avg_metric[:, 0].mean(), avg_metric[0, 0], avg_metric[1, 0], avg_metric[2, 0], avg_metric[:, 1].mean()))
                model.train()

            if (iter_num+1) % (50*210) == 0: 
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':int((iter_num+1) % len(trainloader))-1}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            
        scheduler.step()
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

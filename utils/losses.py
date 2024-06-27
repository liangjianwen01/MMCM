import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True, num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.num_classes = num_classes

    def forward(self, input, target):
        if self.num_classes==2:
            lg = torch.nn.LogSigmoid()
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        if self.num_classes>2:
            logpt = F.log_softmax(input, dim=1)
        elif self.num_classes==2:
            logpt = torch.stack([lg(-input),lg(input)],dim=1).squeeze(2)
        
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.6, reduction='element_wise'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1-pt) ** self.gamma * target * torch.log(pt) - \
                (1-alpha) * pt ** self.gamma * (1-target) * torch.log(1-pt)
        
        if self.reduction=='element_wise':
            loss = torch.mean(loss)
        elif self.reduction=='sum':
            loss = torch.sum(loss)
        return loss

class BCESampleBalanceLoss(torch.nn.Module):

    def __init__(self, reduction='element_wise'):
        super().__init__()
        self.reduction = reduction

    def forward(self, _input, target):
        lg = torch.nn.LogSigmoid()
        
        n_sample = (target.view(-1)).size()[0]
        n_positive_sample = torch.sum(target)
        n_neg_sample = n_sample - n_positive_sample

        alpha = n_neg_sample/n_sample if n_positive_sample>0 else 1e-5

        if alpha.type() != _input.data.type():
            alpha = alpha.type_as(_input.data)
        if n_positive_sample.type() != _input.data.type():
            n_positive_sample = n_positive_sample.type_as(_input.data)
        if n_neg_sample.type() != _input.data.type():
            n_neg_sample = n_neg_sample.type_as(_input.data)

        loss = - alpha * target * lg(_input) - \
                (1-alpha) * (1-target) * lg(-_input)
        
        if self.reduction=='element_wise':
            loss = torch.mean(loss)
        elif self.reduction=='sum':
            loss = torch.sum(loss)
        elif self.reduction=='weight_mean':
            loss = torch.sum(loss)/((alpha*n_positive_sample) + ((1-alpha)*n_neg_sample))
        return loss



class DiceLoss(nn.Module):
    def __init__(self, n_classes, per_class=False):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.per_class = per_class

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        if ignore_mask is not None:
            ignore_mask = ignore_mask.float()
            intersect = torch.sum(score * target * ignore_mask)
            y_sum = torch.sum(target * target * ignore_mask)
            z_sum = torch.sum(score * score * ignore_mask)
        else:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore_mask=None, mutil_mask=False, one_hot=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if mutil_mask:
                dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask[:,i])  
            else:
                dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask) 
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        if self.per_class:
            return loss / self.n_classes, class_wise_dice
        else:
            return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

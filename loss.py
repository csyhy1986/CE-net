import torch.nn as nn
import torch
import numpy as np

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None):
    
    input = input.squeeze(1)
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    
    #    if math.isnan(ce_loss.mean()):
    #       ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp() + 0.1).log())
     #   print(input.grad)
    #    return torch.Tensor([0]) 
            
    if weight is not None:
        ce_loss = ce_loss * weight

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()

#NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences
class clsLoss(nn.Module):
    def __init__(self):
        super(clsLoss, self).__init__()

    def loss_classi(self, pred, label):
        pos_num = torch.sum(label)
        total = torch.numel(label)
        neg_num = total - pos_num
        pos_w = neg_num / (pos_num+1)

        classi_loss = binary_cross_entropy_with_logits(pred, label, pos_weight=pos_w, reduce=True)
        
        return classi_loss

    def forward(self, pred, label):
        loss = self.loss_classi(pred, label)
        return loss

#Learning to Find Good Correspondences
class FundLoss(nn.Module):
    def __init__(self):
        super(FundLoss, self).__init__()
    
    def forward(self, pred_F, label_F):
        sum_F_squre = torch.square(pred_F + label_F)
        dif_F_squre = torch.square(pred_F - label_F)
        sum1 = torch.mean(sum_F_squre,dim=1)
        sum2 = torch.mean(dif_F_squre,dim=1)
        min_sqr = torch.minimum(sum1,sum2)
        F_loss = torch.mean(min_sqr)
        return F_loss
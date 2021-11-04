import numpy as np
import torch

def precision_difF(lbls, pred_lbls):
    pred_lbls = pred_lbls.squeeze(1)
    assert (lbls.shape == pred_lbls.shape)
    
    # sum_F_squre = torch.square(pred_F + F)
    # dif_F_squre = torch.square(pred_F - F)
    # sum_sqrue1 = torch.mean(sum_F_squre,dim=1)
    # sum_sqrue2 = torch.mean(dif_F_squre,dim=1)
    # min_sqr = torch.minimum(sum_sqrue1,sum_sqrue2)
    # dif_F = torch.mean(min_sqr)

    n_batch = lbls.shape[0]
    pt_pairs_batch = lbls.shape[1]
    mask1 = pred_lbls > 0
    pred_lbls[mask1] = 1.0
    mask2 = pred_lbls < 0
    pred_lbls[mask2] = 0.0

    dif_lbls = abs(lbls - pred_lbls)
    precision = 1 - torch.sum(dif_lbls)/(n_batch*pt_pairs_batch)

    false_mask = (pred_lbls < 0.00001) & (lbls < 0.00001)
    n_correct_false = torch.count_nonzero(false_mask)
    n_false = torch.count_nonzero(lbls < 0.00001)
    false_r = n_correct_false / n_false

    correct_mask = (pred_lbls > 0.00001) & (lbls > 0.00001)
    n_correct_true = torch.count_nonzero(correct_mask)
    n_correct = torch.count_nonzero(lbls > 0.00001)
    true_r = n_correct_true / n_correct


    return (precision, false_r, true_r)
import warnings

from torch.nn import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

import math
import torch

from Models.Bayesian_FNO import BayesLinear

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.


    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
        
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, (BayesLinear)):
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))
            
    if last_layer_only or n == 0 :
        return kl
    
    if reduction == 'mean' :
        return kl_sum/n
    elif reduction == 'sum' :
        return kl_sum
    else :
        raise ValueError(reduction + " is not valid")
        

class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            
class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of bayesian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)
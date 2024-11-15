# %%
import torch 
import torch.nn.functional as F 
from Utils.BayesianLoss import * 

def NLL(pred, target):
    """
    Negative Log Likelihood loss for probabilistic predictions
    Args:
        pred_mean: mean prediction
        pred_var: predicted variance
        target: ground truth
        
        Note: to make training more stable, we optimize
        a modified loss by having our model predict log(sigma^2)
        rather than sigma^2.
    """
    pred_mean, pred_log_var = pred[...,0:1], pred[...,1:2]
    loss = (pred_log_var + (pred_mean-target).pow(2) / torch.exp(pred_log_var)) / 2
    return loss.mean()

def ELBO(model, pred, target, batch_size):
    """
    Calculate ELBO loss for Bayesian Neural Network
    
    ELBO = E_q[log p(D|w)] - KL(q(w)||p(w))
    where:
    - q(w) is variational posterior
    - p(w) is prior distribution
    - p(D|w) is likelihood
    """
    likelihood = -F.mse_loss(pred, target, reduction='sum')
    kl_div = BKLLoss(reduction='mean', last_layer_only=True)(model)
    elbo = -(likelihood - kl_div) / batch_size
    return elbo

# %%
import torch 
import torch.nn.functional as F 

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


def elbo(pred, target, kl_div, num_batches, beta=1.0):
    """
    Evidence Lower BOund (ELBO) loss function
    Args:
        pred: model prediction
        target: ground truth
        kl_div: KL divergence term
        num_batches: number of batches in dataset (for KL scaling)
        beta: weight for KL term
    """
    likelihood = -0.5 * F.mse_loss(pred, target, reduction='sum')
    kl_term = beta * kl_div / num_batches
    return -(likelihood - kl_term)
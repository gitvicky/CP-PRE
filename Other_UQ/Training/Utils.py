import torch 

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
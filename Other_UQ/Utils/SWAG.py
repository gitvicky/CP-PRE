import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SWAG:
    def __init__(self, base_model, max_num_models=20, var_clamp=1e-6):
        self.base_model = base_model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        
        self.n_models = 0
        self.theta = None
        self.theta_sq = None
        self.D = None
                
    def _split_complex(self, tensor):
        """Split complex tensor into real and imaginary parts"""
        if torch.is_complex(tensor):
            return torch.cat([tensor.real.flatten(), tensor.imag.flatten()])
        return tensor.flatten()
    
    def _merge_complex(self, tensor, shape):
        """Merge real and imaginary parts back into complex tensor"""
        half_size = tensor.numel() // 2
        if half_size * 2 == tensor.numel():  # Complex tensor
            return (tensor[:half_size] + 1j * tensor[half_size:]).reshape(shape)
        return tensor.reshape(shape)  # Real tensor
    
    def collect_model(self, model):
        # Convert all parameters to real vectors
        w_list = []
        for p in model.parameters():
            w_list.append(self._split_complex(p.data))
        w = torch.cat(w_list).to(device)
        
        if self.theta is None:
            self.theta = w.clone()
            self.theta_sq = w ** 2
        else:
            self.n_models += 1
            self.theta = (self.theta * self.n_models + w) / (self.n_models + 1)
            self.theta_sq = (self.theta_sq * self.n_models + w ** 2) / (self.n_models + 1)
        
        deviation = w - self.theta
        if self.D is None:
            self.D = deviation.view(-1, 1)
        else:
            if self.D.shape[1] < self.max_num_models:
                self.D = torch.cat([self.D, deviation.view(-1, 1)], dim=1)
            else:
                self.D = torch.cat([self.D[:, 1:], deviation.view(-1, 1)], dim=1)

    def sample(self, scale=0.5):
        """Sample weights handling both real and complex parameters"""
        # Ensure random tensors are created on the correct device
        z1 = torch.randn(self.theta.shape[0], dtype=self.theta.dtype, device=device)
        
        # Calculate variance separately for real and imaginary parts
        variance = torch.clamp(self.theta_sq - self.theta ** 2, self.var_clamp)
        
        # Start with mean and diagonal component
        w = self.theta + scale * torch.sqrt(variance) * z1
        
        # Add low rank component if available
        if self.D is not None and self.D.shape[1] > 0:
            z2 = torch.randn(self.D.shape[1], dtype=self.theta.dtype, device=device)
            w += scale / ((self.D.shape[1] - 1) ** 0.5) * self.D.mv(z2)
        
        # Update model parameters
        curr_idx = 0
        for p in self.base_model.parameters():
            param_size = p.numel()
            if torch.is_complex(p.data):
                # For complex parameters, we need to handle twice the size
                param_size *= 2
                p.data = self._merge_complex(w[curr_idx:curr_idx + param_size], p.shape)
            else:
                # For real parameters, just reshape
                p.data = w[curr_idx:curr_idx + param_size].view(p.shape)
            curr_idx += param_size
    
    def to(self, device):
        """Move the SWAG model to the specified device"""
        device = device
        self.base_model = self.base_model.to(device)
        
        # Move SWAG statistics to device
        if self.theta is not None:
            self.theta = self.theta.to(device)
        if self.theta_sq is not None:
            self.theta_sq = self.theta_sq.to(device)
        if self.D is not None:
            self.D = self.D.to(device)
        
        return self
    
    def save(self, path):
        """Save SWAG model state"""
        torch.save({
            'base_model_state_dict': self.base_model.state_dict(),
            'n_models': self.n_models,
            'theta': self.theta,
            'theta_sq': self.theta_sq,
            'D': self.D,
            'max_num_models': self.max_num_models,
            'var_clamp': self.var_clamp,
            'device': device
        }, path)
    
    @classmethod
    def load(cls, path, base_model):
        """Load SWAG model state"""
        checkpoint = torch.load(path, map_location=device)
        
        swag = cls(base_model, 
                  max_num_models=checkpoint['max_num_models'],
                  var_clamp=checkpoint['var_clamp'])
        
        base_model.load_state_dict(checkpoint['base_model_state_dict'])
        swag.n_models = checkpoint['n_models']
        swag.theta = checkpoint['theta']
        swag.theta_sq = checkpoint['theta_sq']
        swag.D = checkpoint['D']
        swag.device = checkpoint['device']
        
        # Move model to the saved device
        swag.to(device)
        
        return swag
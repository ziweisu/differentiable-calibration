import numpy as np
import scipy.stats as st
import torch
from torch import nn
##############################################################################
# Defines scoring functions 
##############################################################################

# Continuously Ranked Probability Score
def crps(x, y, beta=1.5, epsilon=1e-6):
    # bias correction
    n = y.size(dim = 0)
    if beta == 1 or beta == 2:
        return (torch.mean(torch.abs(y[..., None] - x) ** beta) \
                - 0.5 * n / (n-1) * torch.mean(torch.abs(y[..., None] - y) ** beta))
    else:
        # add smoothing for numerical stability
        return (torch.mean(smooth_abs(y[..., None] - x, epsilon) ** beta) - 
                0.5 * n / (n-1) * torch.mean(smooth_abs(y[..., None] - y, epsilon) ** beta))
    
def smooth_abs(x, epsilon=1e-6):
    return torch.sqrt(x**2 + epsilon)

def crps_multidim(x, y, beta=1):
    n = y.size(0)
    
    # Compute pairwise differences
    diff_xy = x.unsqueeze(1) - y.unsqueeze(0)
    diff_yy = y.unsqueeze(1) - y.unsqueeze(0)
    
    # Compute Euclidean norms
    norm_xy = torch.norm(diff_xy, dim=2).pow(beta)
    norm_yy = torch.norm(diff_yy, dim=2).pow(beta)
    
    # Compute means
    mean_xy = torch.mean(norm_xy)
    mean_yy = torch.mean(norm_yy)
    
    # Compute CRPS
    crps_value = mean_xy - 0.5 * n / (n - 1) * mean_yy
    
    return crps_value

 
# Dawid-Sebastiani Score
def DS(X, Y): 
    mu_hat = torch.mean(Y)
    var_hat = torch.var(Y, unbiased = True)
    score = torch.mean(torch.square(X - mu_hat) / var_hat)  + torch.log(var_hat)
    # Scaling
    # score = score / 20
    return score

# Predictive Model Choice Criterion
def pmcc(X, Y):
    mu_hat = torch.mean(Y)
    var_hat = torch.var(Y)
    score = torch.mean(torch.square(X - mu_hat)) + var_hat
    # Scaling
    # score = score / 30
    return score

# Log Score
def LogS(X, Y):
    # Fit kernel density
    kde = st.gaussian_kde(Y, bw_method='silverman')
    kde_X = kde.pdf(X)
    kde_X = np.where(kde_X > 1e-10, kde_X, 1e-10)
    score = -torch.mean(np.log(kde_X))
    return score

# Energy Score
def ES(X, Y):
    beta = 1.5 # \beta \in (0, 2)
    score = 0
    for i in range(Y.size(dim=0)):
        score = score - torch.mean((torch.abs(Y[i] - X)) ** beta) \
            / Y.size(dim=0) + torch.sum((torch.abs(Y[i] - Y)) ** beta) / \
                (2 * Y.size(dim=0) * (Y.size(dim=0) - 1))
        
    return score

# Kernel Score with Gaussian RBF kernel
def compute_median_l2_distance(X):
    distances = torch.abs(X[..., None] - X)
    median_distance = torch.median(distances)
    return median_distance

class Gaussian_RBF(nn.Module):
    def __init__(self, sigma):
        super(Gaussian_RBF, self).__init__()
        self.sigma = sigma
    def forward(self, x, y):
        if True:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        L2_distance = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-L2_distance[None, ...] / (2 * self.sigma ** 2)).sum(dim=0).mean()
    
def KS(x, y):
    # median heuristic for bandwidth
    if True:
        median_distance = compute_median_l2_distance(x)
        sigma = torch.sqrt(median_distance ** 2 / 2)
    else:
        sigma = 1
    n = y.size(dim=0)
    return (-2 * torch.mean(torch.exp(-torch.square(y[..., None] - x) / (2 * sigma ** 2))) \
            + n / (n-1) * torch.mean(torch.exp(-torch.square(y[..., None] - y) / (2 * sigma ** 2))))

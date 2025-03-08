import torch
from scoring_rules import compute_median_l2_distance
def Riesz(x, y, beta=1):
    return torch.abs(y - x) ** beta / 2

def Gaussian(x, y):
    if True:
        median_distance = compute_median_l2_distance(x)
        sigma = torch.sqrt(median_distance ** 2 / 2)
    else:
        sigma = 1
    return torch.exp(-torch.square(y - x) / (2 * sigma ** 2))
import os
import sys
sys.path.append(os.path.join(os.path.abspath(''), './src'))
import numpy as np
import pandas as pd
import torch
from scoring_rules import crps, DS, pmcc, KS
from estimation import sgd_estimation, conf_int
from simulation_model import lindley
from plot import plot_loss
from tqdm import tqdm
from kernels import Riesz, Gaussian
import matplotlib.pyplot as plt
import seaborn as sns
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # change if needed
theta_opts = [1.2, 1.46515538, 1.84569039, 2.49797974, 3.9806058]
N = 1000  # number of draws from prior
df = {'N': [0, 0, 0, 0, 0], 'm': [0, 0, 0, 0, 0], 'MSE, posterior mean': [0, 0, 0, 0, 0],
      'CovProb, Credible Set': [0, 0, 0, 0, 0], 'Width, Credible Set': [0, 0, 0, 0, 0],
      'cpu time':[0, 0, 0, 0, 0], 'Service shape':[1, 0.8, 0.6, 0.4, 0.2]}
macro_count = 1000
m = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

index = 0
for service_shape in [1, 0.8, 0.6, 0.4, 0.2]:
    theta_opt = theta_opts[index]
    x_sample = lindley(theta_opt, replications=m, service_shape_parameter=service_shape, device=device)
    df['N'][index] = N
    df['m'][index] = m
    for k in tqdm(range(macro_count)):
        time_start = time.time()
        search_range = torch.from_numpy(np.random.uniform(theta_opt-1, theta_opt+1, N)).to(device)
        # Sample from the model
        y_samples = torch.stack([lindley(lambda_i, replications=m, device=device).to(device)
                                 for lambda_i in search_range])
        
        # Compute the distances in parallel
        d = torch.tensor([KS(x_sample, y_samples[i, :]) for i in range(N)]).to(device)
        
        # Compute the 1% quantile
        q = torch.quantile(d, 0.01)
        
        # Compute the credible set
        Theta = search_range[d <= q]
        upper_threshold = torch.quantile(Theta, 0.975)
        lower_threshold = torch.quantile(Theta, 0.025)
        credible_set = Theta[(Theta <= upper_threshold) & (Theta >= lower_threshold)]
        
        df['Width, Credible Set'][index] += (upper_threshold - lower_threshold).cpu().numpy()
        df['MSE, posterior mean'][index] += ((torch.mean(Theta) - theta_opt)**2).cpu().numpy()
        
        if torch.min(Theta) <= 1.2 and torch.max(Theta) >= 1.2:
            df['CovProb, Credible Set'][index] += 1 / macro_count
        
        if k == 0:
            # Plot the posterior draws
            plt.figure(figsize=(8, 6))
            sns.kdeplot(Theta.cpu().numpy(), fill=True, label='Posterior')
            
            # Add lines for 2.5%, 97.5% quantiles, posterior mean, posterior median, and true parameter
            plt.axvline(lower_threshold.cpu().numpy(), color='b', linestyle='--', label='2.5% quantile')
            plt.axvline(upper_threshold.cpu().numpy(), color='b', linestyle='--', label='97.5% quantile')
            plt.axvline(torch.mean(Theta).cpu().numpy(), color='g', linestyle='--', label='Posterior mean')
            plt.axvline(torch.median(Theta).cpu().numpy(), color='y', linestyle='--', label='Posterior median')
            plt.axvline(theta_opt, color='r', linestyle='-', label='True value')
            
            plt.xlabel('Service rate')
            plt.ylabel('Density')
            plt.title('Posterior of service rate')
            plt.legend()
            
            # Create the directory if it doesn't exist
            os.makedirs('./results/plots/1d_inexact_MMD_ABC', exist_ok=True)
            
            # Save the plot
            plt.savefig(f'./results/plots/1d_inexact_MMD_ABC/1d_inexact_MMD_ABC_a={service_shape}.png')
            plt.close()
        df['cpu time'][index] += (time.time() - time_start) / macro_count
    
    df['Width, Credible Set'][index] /= macro_count
    df['MSE, posterior mean'][index] /= macro_count
    index += 1


print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/1d_inexact_MMD_ABC', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/1d_inexact_MMD_ABC/results.csv', index=False)
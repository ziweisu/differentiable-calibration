import os
import sys
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.join(os.path.abspath(''), './src'))
from scoring_rules import crps, DS, pmcc, KS
from estimation import sgd_estimation, conf_int
from simulation_model import lindley
from plot import plot_loss
from tqdm import tqdm
from kernels import Riesz, Gaussian
import matplotlib.pyplot as plt
import seaborn as sns
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change if needed

N = 1000
macro_count = 1000
theta_opt = 1.2
df = {'N': [0, 0, 0, 0, 0],
    'm': [0, 0, 0, 0, 0],
    'number of empty set': [0, 0, 0, 0, 0],
    'CovProb, Eligibility Set' : [0, 0, 0, 0, 0],
    'Width, Eligibility Set' : [0, 0, 0, 0, 0],
    'CPU time':[0, 0, 0, 0, 0]}              
q = 1.36 # 95% CI
index = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for m in [10, 50, 100, 500, 1000]:
    x_sample = lindley(theta_opt, replications=m, device=device)
    df['N'][index] = N
    df['m'][index] = m
    for k in tqdm(range(macro_count)):
        time_start = time.time()
        search_range = np.random.uniform(0.5, 2.5, N)
        a = torch.empty([N, m], device=device)
        b = torch.empty([N, m], device=device)
        i = 0
        for lambda_i in search_range:
            # simulate for M/M/1 model with rate lambda_i
            y_sample = lindley(lambda_i, replications=N, device=device)
            # compute eligibility set parameters
            a[i, :] = torch.abs(torch.mean((y_sample[:, None] <= x_sample).float(), axis = 0) - torch.mean((x_sample[:, None] <= x_sample).float(), axis = 0))
            b[i, :] = torch.abs(torch.mean((y_sample[:, None] < x_sample).float(), axis = 0) - torch.mean((x_sample[:, None] < x_sample).float(), axis = 0))
            i += 1
        a_max = torch.max(a, axis = 1).values
        b_max = torch.max(b, axis = 1).values
        # Construct eligibility set
        index_set = ((a_max <= q / np.sqrt(m)) & (b_max <= q / np.sqrt(m)))
        Theta = search_range[index_set.cpu().numpy()]  
        if np.any(Theta):
            df['Width, Eligibility Set'][index] += (np.max(Theta) - np.min(Theta))
            if np.min(Theta) <= theta_opt and np.max(Theta) >= theta_opt:
                df['CovProb, Eligibility Set'][index] += 1/macro_count
        else:
            df['number of empty set'][index] += 1
        if k == 0:
            # Plot the posterior draws
            plt.figure(figsize=(8, 6))
            sns.kdeplot(Theta, fill=True, label='Eligibility Set')
            
            # Add lines for 2.5%, 97.5% quantiles, posterior mean, posterior median, and true parameter
            plt.axvline(np.min(Theta), color='b', linestyle='--', label='Set minimum')
            plt.axvline(np.max(Theta), color='b', linestyle='--', label='Set maximum')
            plt.axvline(np.mean(Theta), color='g', linestyle='--', label='Set mean')
            plt.axvline(np.median(Theta), color='y', linestyle='--', label='Set median')
            plt.axvline(theta_opt, color='r', linestyle='-', label='True value')
            
            plt.xlabel('Service rate')
            plt.ylabel('Density')
            plt.title('Eligibility Set of service rate')
            plt.legend()
            
            # Create the directory if it doesn't exist
            os.makedirs('./results/plots/1d_exact_eligibility_set', exist_ok=True)
            
            # Save the plot
            plt.savefig(f'./results/plots/1d_exact_eligibility_set/1d_exact_eligibility_set_m={m}.png')
            plt.close()
        df['CPU time'][index] += (time.time() - time_start) / macro_count
    df['Width, Eligibility Set'][index] /= (macro_count - df['number of empty set'][index]) 
    index += 1

print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/1d_exact_eligibility_set', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/1d_exact_eligibility_set/results.csv', index=False)
import os
import sys
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.join(os.path.abspath(''), './src'))
from scoring_rules import crps, DS, pmcc, KS
from estimation_2d import sgd_estimation, conf_int
from simulation_model_2d import lindley
from plot import plot_loss
from tqdm import tqdm
from kernels import Riesz, Gaussian
import matplotlib.pyplot as plt
import seaborn as sns
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change if needed

N = 1000
m = 1000
macro_count = 1000
df = {'N': [0, 0, 0, 0, 0],
    'm': [0, 0, 0, 0, 0],
    'Service shape':[0, 0, 0, 0, 0],
    'number of empty set': [0, 0, 0, 0, 0],
    'CovProb, Eligibility Set' : [0, 0, 0, 0, 0],
    'Width, Eligibility Set, Service rate' : [0, 0, 0, 0, 0],
    'Width, Eligibility Set, Arrival rate' : [0, 0, 0, 0, 0],
    'Average CPU time': [0, 0, 0, 0, 0],
    'theta_opt': [np.array([2.5, 1]), np.array([5.86548716, 2.66126376]), 
                  np.array([7.87392723, 3.02316964]), np.array([10.28224743,  2.90738165]), np.array([0.0166845 , 0.41594256])]}              
q = 1.36 # 95% CI
index = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for service_shape in [1, 0.8, 0.6, 0.4, 0.2]:
    x_sample = lindley(service_rate=2.5, arrival_rate=1, service_shape_parameter=service_shape, arrival_shape_parameter=0.5, replications=m).to(device)
    df['N'][index] = N
    df['m'][index] = m
    df['Service shape'][index] = service_shape
    theta_opt = df['theta_opt'][index]
    for k in tqdm(range(macro_count)):
        start_time = time.time()
        low = df['theta_opt'][index] - 1
        high = df['theta_opt'][index] + 1
        search_range = np.random.uniform(low=low, high=high, size=(N, 2))
        a = torch.empty([N, m]).to(device)
        b = torch.empty([N, m]).to(device)
        i = 0
        for lambda_i in search_range:
            # simulate for M/M/1 model with rate lambda_i
            y_sample = lindley(service_rate=lambda_i[0], arrival_rate=lambda_i[1], service_shape_parameter=service_shape, replications=N).to(device)
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
            df['Width, Eligibility Set, Service rate'][index] += (np.max(Theta[:, 0]) - np.min(Theta[:, 0]))
            df['Width, Eligibility Set, Arrival rate'][index] += (np.max(Theta[:, 1]) - np.min(Theta[:, 1]))
            if np.min(Theta[:, 0]) <= theta_opt[0] and np.min(Theta[:, 1]) <= theta_opt[1] and np.max(Theta[:, 0]) >= theta_opt[0] and np.max(Theta[:, 1]) >= theta_opt[1]:
                df['CovProb, Eligibility Set'][index] += 1/macro_count
        else:
            df['number of empty set'][index] += 1
        if k == 0:
            # Plot the posterior draws
            plt.figure(figsize=(8, 6))
            plt.scatter(Theta[:, 0], Theta[:, 1], alpha=0.5, label='Eligibility Set')
            plt.scatter(theta_opt[0], theta_opt[1], color='r', marker='*', s=100, label='Optimal Parameter (Gaussian)')
            plt.xlabel('Service rate')
            plt.ylabel('Arrival rate')
            plt.title('Eligibility Set of service and arrival rates')
            plt.legend()
            os.makedirs('./results/plots/2d_inexact_eligibility_set', exist_ok=True)
            plt.savefig(f'./results/plots/2d_inexact_eligibility_set/2d_inexact_eligibility_set_alpha={service_shape}.png')
            plt.close()
        df['Average CPU time'][index] += (time.time() - start_time) / macro_count
    if df['number of empty set'][index] < macro_count:
        df['Width, Eligibility Set, Service rate'][index] /= (macro_count - df['number of empty set'][index])
        df['Width, Eligibility Set, Arrival rate'][index] /= (macro_count - df['number of empty set'][index])

    index += 1
    # Create the directory if it doesn't exist
    os.makedirs('./results/dataframes/2d_inexact_eligibility_set', exist_ok=True)

    # Save the dataframe to a CSV file
    pd.DataFrame(df).to_csv(f'./results/dataframes/2d_inexact_eligibility_set/results_alpha={service_shape}.csv', index=False)



print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/2d_inexact_eligibility_set', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/2d_inexact_eligibility_set/results.csv', index=False)
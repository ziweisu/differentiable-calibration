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
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # change if needed

# Riesz kernel (Energy Score), G/G/1 example, non stationary, exact
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
theta_init = np.array([float(5), float(5)])
theta_opt = np.array([2.5, 1.0])
df = {'n': [0, 0, 0, 0, 0], 
      'm': [0, 0, 0, 0, 0],
      'epsilon': [0, 0, 0, 0, 0], 
      'MSE, Service Rate': [0, 0, 0, 0, 0], 
      'MSE, Arrival Rate': [0, 0, 0, 0, 0], 
      'Average CPU Time': [0, 0, 0, 0, 0]}
macro_count = 100
index = 0
n = 1000
m = 1000
index = 0
for epsilon in [0.01, 0.05, 0.1, 0.2, 0.5]:
    df['n'][index] = n
    df['m'][index] = m
    x_sample = lindley(service_rate=2.5, arrival_rate=1, service_shape_parameter=1, arrival_shape_parameter=0.5, replications=m).to(device)
    num_samples_to_modify = int(epsilon * len(x_sample))
    random_indices = torch.randperm(len(x_sample))[:num_samples_to_modify].to(device)
    noise = (torch.randn(num_samples_to_modify) * 0.1).to(device)  # Standard deviation of 0.1
    x_sample[random_indices] += noise
    for k in tqdm(range(macro_count)):
        start_time = time.time()
        theta_tilde, _, para_list = sgd_estimation(lindley, x_sample, KS, theta_init, service_shape_parameter=1, 
                                                   arrival_shape_parameter=0.5, simulation_replications=n, number_epochs=800, lr_init=1, device=device)
        df['MSE, Service Rate'][index] += (theta_tilde[0] - theta_opt[0]) ** 2 / macro_count
        df['MSE, Arrival Rate'][index] += (theta_tilde[1] - theta_opt[1]) ** 2 / macro_count
        
        # Plot the confidence set for k=0
        if k == 0:                     
            fig, ax = plt.subplots()

            # Plot the initial parameter
            ax.scatter(theta_init[0], theta_init[1], color='orange', marker='x', label='Initial Parameter')

            # Plot the estimated parameter
            ax.scatter(theta_tilde[0], theta_tilde[1], color='green', marker='o', label='Estimated Parameter')

            # Plot the optimal parameter
            ax.scatter(theta_opt[0], theta_opt[1], color='red', marker='o', label='Optimal Parameter')

            # Plot the convergence of parameters
            service_rates = [theta_init[0]] + [param[0] for param in para_list]
            arrival_rates = [theta_init[1]] + [param[1] for param in para_list]
            ax.plot(service_rates, arrival_rates, color='purple', linestyle='-', marker='.', markersize=2, label='Parameter Convergence')

            ax.set_xlabel('Service Rate')
            ax.set_ylabel('Arrival Rate')
            ax.set_title(f'Parameter Convergence(m={m})')
            ax.legend()
             
            # Save the plot
            os.makedirs('./results/plots/2d_inexact_stat_noise_Gaussian', exist_ok=True)
            plt.savefig(f'./results/plots/2d_inexact_stat_noise_Gaussian/2d_inexact_stat_noise_Gaussian_epsilon={epsilon}_conv.png')
            plt.close()
        
        df['Average CPU Time'][index] += (time.time() - start_time) / macro_count
    
    # Create the directory if it doesn't exist
    os.makedirs('./results/dataframes/2d_inexact_stat_noise_Gaussian', exist_ok=True)

    # Save the dataframe to a CSV file
    pd.DataFrame(df).to_csv(f'./results/dataframes/2d_inexact_stat_noise_Gaussian/results_epsilon={epsilon}.csv', index=False)
    index += 1

print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/2d_inexact_stat_noise_Gaussian', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/2d_inexact_stat_noise_Gaussian/results.csv', index=False)
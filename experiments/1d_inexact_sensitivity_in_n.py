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
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # change if needed

# Riesz kernel (Energy Score), G/G/1 example, non stationary, exact
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
theta_init = np.array([float(5)])
df = {'n': [0, 0, 0, 0, 0], 
      'm': [0, 0, 0, 0, 0],
      'Service_shape': [0, 0, 0, 0, 0], 
      'MSE, Service Rate': [0, 0, 0, 0, 0], 
      'CovProb': [0, 0, 0, 0, 0],
      'Width': [0, 0, 0, 0, 0],
      'Asymptotic variance':[0, 0, 0, 0, 0],
      'Average CPU Time': [0, 0, 0, 0, 0],
      'theta_opt':[1.2, 1.2, 1.2, 1.2, 1.2]}
macro_count = 10
index = 1
n = [2, 10, 50, 100, 200]
m = 500
index = 0
service_shape = 1
for n in [2, 10, 50, 100, 200]:
    df['n'][index] = n
    df['m'][index] = m
    df['Service_shape'][index] = service_shape
    theta_opt = df['theta_opt'][index]
    x_sample = lindley(service_rate=1.2, service_shape_parameter=service_shape, replications=m, device=device)
    for k in tqdm(range(macro_count)):
        start_time = time.time()
        theta_tilde, _, para_list = sgd_estimation(lindley, x_sample, crps, theta_init, 
                                                   simulation_replications=n, number_epochs=200, lr_init=1, device=device)
        df['MSE, Service Rate'][index] += (theta_tilde - theta_opt) ** 2 / macro_count
        ci_lower, ci_upper, width, var_hat = conf_int(lindley, x_sample, crps, theta_tilde, Riesz, device=device)
        df['Width'][index] += 1/macro_count*width
        df['Asymptotic variance'][index] += 1/macro_count*var_hat/m
        df['CovProb'][index]  += 1/macro_count*(ci_lower <= theta_opt and theta_opt <= ci_upper)
        
        # Plot the confidence set for k=0
        if k == 0:
            fig, ax = plt.subplots()

            # Plot the optimal parameter
            ax.axvline(theta_opt, color='red', linestyle='--', label='Optimal Parameter')

            # Plot the estimated parameter
            ax.axvline(theta_tilde, color='green', linestyle='--', label='Estimated Parameter')

            # Plot the upper and lower bounds of the CI
            ax.axvline(ci_lower, color='blue', linestyle='--', label='Lower Bound')
            ax.axvline(ci_upper, color='blue', linestyle='--', label='Upper Bound')

            # Plot the normal curve
            x = np.linspace(theta_tilde - 5 * np.sqrt(var_hat / m), theta_tilde + 5 * np.sqrt(var_hat / m), 1000)
            y = norm.pdf(x, loc=theta_tilde, scale=np.sqrt(var_hat / m))
            ax.plot(x, y, color='purple', label='Normal Curve')

            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Confidence Interval (Service Shape={service_shape})')
            ax.legend()

            # Save the plot
            os.makedirs('./results/plots/1d_inexact_sensitivity_in_n', exist_ok=True)
            plt.savefig(f'./results/plots/1d_inexact_sensitivity_in_n/1d_inexact_sensitivity_in_n_alpha={service_shape}_CI.png')
            plt.close()          
            
        df['Average CPU Time'][index] += (time.time() - start_time) / macro_count
    
    index += 1

print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/1d_inexact_sensitivity_in_n', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv(f'./results/dataframes/1d_inexact_sensitivity_in_n/results_alpha={service_shape}.csv', index=False)
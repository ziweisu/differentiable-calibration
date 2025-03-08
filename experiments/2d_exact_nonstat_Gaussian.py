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

# Gaussian kernel (Energy Score), G/G/1 example, non stationary, exact
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
theta_opt = np.array([1, 1])
theta_init = np.array([float(5), float(5)])
df = {'n': [0, 0, 0, 0, 0], 
      'm': [0, 0, 0, 0, 0], 
      'MSE, Service Rate': [0, 0, 0, 0, 0], 
      'MSE, Arrival Rate': [0, 0, 0, 0, 0], 
      'CovProb, Expected Score Estimate': [0, 0, 0, 0, 0],
      'Average CPU Time': [0, 0, 0, 0, 0]}
macro_count = 1000
index = 0
for m in [500, 1000]:
    n = m
    df['n'][index] = n
    df['m'][index] = m 
    x_sample = lindley(service_rate=1, arrival_rate=1, service_shape_parameter=1, arrival_shape_parameter=0.5, replications=m).to(device)
    
    for k in tqdm(range(macro_count)):
        start_time = time.time()
        theta_tilde, _, para_list = sgd_estimation(lindley, x_sample, KS, theta_init, arrival_shape_parameter=0.5, simulation_replications=n, number_epochs=600, lr_init=2, device=device)
        df['MSE, Service Rate'][index] += (theta_tilde[0] - theta_opt[0]) ** 2 / macro_count
        df['MSE, Arrival Rate'][index] += (theta_tilde[1] - theta_opt[1]) ** 2 / macro_count
        H_hat, sigma_hat = conf_int(lindley, x_sample, KS, theta_tilde, Gaussian, service_shape_parameter=1, arrival_shape_parameter=0.5, device=device, hessian_type="finite_difference")
        sigma_hat_inv = np.linalg.inv(sigma_hat)
        H_hat_inv = np.linalg.inv(H_hat)
        
        # Plot the confidence set for k=0
        if k == 0:
            C = H_hat_inv.dot(sigma_hat).dot(H_hat_inv)  # the covariance matrix
            fig, ax = plt.subplots()
            
            # Compute the confidence set ellipsoid
            chi2_val = scipy.stats.chi2.ppf(0.95, df=2)
            eigenvalues, eigenvectors = np.linalg.eigh(C/m)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(chi2_val) * np.sqrt(eigenvalues)
            
            # Plot the ellipsoid
            ellipse = Ellipse(xy=theta_tilde, width=width, height=height, angle=angle, edgecolor='blue', fc='None', lw=2)
            ax.add_patch(ellipse)
            
            # Plot the optimal parameter
            ax.scatter(theta_opt[0], theta_opt[1], color='red', marker='*', label='Optimal Parameter')
            
            # Plot the estimated parameter
            ax.scatter(theta_tilde[0], theta_tilde[1], color='green', marker='o', label='Estimated Parameter')

            ax.set_xlabel('Service Rate')
            ax.set_ylabel('Arrival Rate')
            ax.set_title(f'Confidence Set (m={m})')
            ax.legend()
            
            # Save the plot
            os.makedirs('../results/plots/2d_exact_nonstat_Gaussian', exist_ok=True)
            plt.savefig(f'../results/plots/2d_exact_nonstat_Gaussian/2d_exact_nonstat_Gaussian_m={m}_CI.png')
            plt.close()            
            
            fig, ax = plt.subplots()

            # Plot the initial parameter
            ax.scatter(theta_init[0], theta_init[1], color='orange', marker='x', label='Initial Parameter')

            # Plot the estimated parameter
            ax.scatter(theta_tilde[0], theta_tilde[1], color='green', marker='o', label='Estimated Parameter')

            # Plot the convergence of parameters
            service_rates = [theta_init[0]] + [param[0] for param in para_list]
            arrival_rates = [theta_init[1]] + [param[1] for param in para_list]
            ax.plot(service_rates, arrival_rates, color='purple', linestyle='-', marker='.', markersize=2, label='Parameter Convergence')

            ax.set_xlabel('Service Rate')
            ax.set_ylabel('Arrival Rate')
            ax.set_title(f'Parameter Convergence(m={m})')
            ax.legend()
            
            # Save the plot
            os.makedirs('../results/plots/2d_exact_nonstat_Gaussian', exist_ok=True)
            plt.savefig(f'../results/plots/2d_exact_nonstat_Gaussian/2d_exact_nonstat_Gaussian_m={m}_conv.png')
            plt.close()
        
        # Check if the confidence interval covers the true parameter
        ci_stat = np.linalg.norm((np.sqrt(m) * scipy.linalg.sqrtm(sigma_hat_inv).dot(H_hat)).dot(theta_opt - theta_tilde))
        if ci_stat ** 2 <= scipy.stats.chi2.ppf(0.95, df=2):
            df['CovProb, Expected Score Estimate'][index] += 1/macro_count
        df['Average CPU Time'][index] += (time.time() - start_time) / macro_count
    # Create the directory if it doesn't exist
    os.makedirs('./results/dataframes/2d_exact_nonstat_Gaussian', exist_ok=True)

    # Save the dataframe to a CSV file
    pd.DataFrame(df).to_csv(f'./results/dataframes/2d_exact_nonstat_Gaussian/results_m={m}.csv', index=False)
    index += 1

print(pd.DataFrame(df))

# Create the directory if it doesn't exist
os.makedirs('./results/dataframes/2d_exact_nonstat_Gaussian', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/2d_exact_nonstat_Gaussian/results.csv', index=False)
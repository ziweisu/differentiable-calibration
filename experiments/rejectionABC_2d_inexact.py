import os
import sys
sys.path.append(os.path.join(os.path.abspath(''), './src'))
import numpy as np
import pandas as pd
import torch
from scoring_rules import crps, DS, pmcc, KS
from estimation_2d import sgd_estimation, conf_int
from simulation_model_2d import lindley
from plot import plot_loss
from tqdm import tqdm
from kernels import Riesz, Gaussian
import matplotlib.pyplot as plt
import seaborn as sns
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # change if needed

N = 9000  # number of draws from prior, 1% as posterior draw
df = {'N': [0, 0, 0, 0, 0], 
      'm': [0, 0, 0, 0, 0], 
      'Service shape':[0, 0, 0, 0, 0],
      'CovProb, Credible Set': [0, 0, 0, 0, 0], 
      'Width, Credible Set, Service Rate': [0, 0, 0, 0, 0],
      'Width, Credible Set, Arrival Rate': [0, 0, 0, 0, 0],
      'average cpu time': [0, 0, 0, 0, 0],
      'theta_opt': [np.array([2.5, 1]), np.array([5.86548716, 2.66126376]), 
                  np.array([7.87392723, 3.02316964]), np.array([10.28224743,  2.90738165]), np.array([0.0166845 , 0.41594256])]}
macro_count = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index = 0
m = 1000
for service_shape in [1]:
    x_sample = lindley(service_rate=2.5, arrival_rate=1, service_shape_parameter=service_shape, arrival_shape_parameter=0.5, replications=m).to(device)
    df['N'][index] = N
    df['m'][index] = m
    df['Service shape'][index] = service_shape
    theta_opt = df['theta_opt'][index]
    for k in tqdm(range(macro_count)):
        start_time = time.time()
        low = df['theta_opt'][index] - 1
        high = df['theta_opt'][index] + 1
        search_range = torch.from_numpy(np.random.uniform(low=low, high=high, size=(N, 2))).to(device)
        # Sample from the model
        y_samples = torch.stack([lindley(service_rate=lambda_i[0], arrival_rate=lambda_i[1], service_shape_parameter=service_shape, replications=N, device=device).to(device)
                                 for lambda_i in search_range])
        
        # Compute the distances in parallel
        d = torch.tensor([KS(x_sample, y_samples[i, :]) for i in range(N)]).to(device)
        
        # Compute the 1% quantile
        q = torch.quantile(d, 0.01)
        
        # Compute the credible set
        Theta = search_range[d <= q]
        upper_threshold_service_rate = torch.quantile(Theta[:, 0], 0.9875).cpu().numpy()
        lower_threshold_service_rate = torch.quantile(Theta[:, 0], 0.0125).cpu().numpy()
        upper_threshold_arrival_rate = torch.quantile(Theta[:, 1], 0.9875).cpu().numpy()
        lower_threshold_arrival_rate = torch.quantile(Theta[:, 1], 0.0125).cpu().numpy()
        
        df['Width, Credible Set, Service Rate'][index] += (upper_threshold_service_rate - lower_threshold_service_rate) /  macro_count
        df['Width, Credible Set, Arrival Rate'][index] += (upper_threshold_arrival_rate - lower_threshold_arrival_rate) /  macro_count
        #df['MSE, posterior mean'][index] += ((torch.mean(Theta) - theta_opt)**2).cpu().numpy()
        
        if torch.min(Theta) <= 1.2 and torch.max(Theta) >= 1.2:
            df['CovProb, Credible Set'][index] += 1 / macro_count
        
        if k == 0:
            # Plot joint ETI region
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(Theta[:, 0].cpu().numpy(), Theta[:, 1].cpu().numpy(), alpha=0.1)

            # Create a rectangle patch for the joint ETI region
            eti_rect = plt.Rectangle((lower_threshold_service_rate, lower_threshold_arrival_rate), 
                                     upper_threshold_service_rate-lower_threshold_service_rate, upper_threshold_arrival_rate-lower_threshold_arrival_rate,
                         linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='Joint ETI Region')
            ax.add_patch(eti_rect)

            ax.plot(theta_opt[0], theta_opt[1], 'go', label='Optimal Parameter')
            ax.set_xlabel('Service Rate')
            ax.set_ylabel('Arrival Rate')
            ax.set_title('Joint ETI Region')
            ax.legend()

            # Save the plot
            os.makedirs('./results/plots/rejectionABC_2d_inexact', exist_ok=True)
            plt.savefig(f'./results/plots/rejectionABC_2d_inexact/rejectionABC_2d_inexact_alpha={service_shape}_CI.png')
            plt.close()   

            upper_threshold_serv = torch.quantile(Theta[:, 0], 0.975).cpu().numpy()
            lower_threshold_serv = torch.quantile(Theta[:, 0], 0.025).cpu().numpy()
            upper_threshold_arriv = torch.quantile(Theta[:, 1], 0.975).cpu().numpy()
            lower_threshold_arriv = torch.quantile(Theta[:, 1], 0.025).cpu().numpy()

            # Plot the posterior draws
            plt.figure(figsize=(8, 6))
            sns.kdeplot(Theta[:, 0].cpu().numpy(), fill=True, label='Posterior')

            # Add lines for 2.5%, 97.5% quantiles, posterior mean, posterior median, and true parameter
            plt.axvline(lower_threshold_serv, color='b', linestyle='--', label='2.5% quantile')
            plt.axvline(upper_threshold_serv, color='b', linestyle='--', label='97.5% quantile')
            plt.axvline(torch.mean(Theta[:, 0]).cpu().numpy(), color='g', linestyle='--', label='Posterior mean')
            plt.axvline(torch.median(Theta[:, 0]).cpu().numpy(), color='y', linestyle='--', label='Posterior median')
            plt.axvline(theta_opt[0], color='r', linestyle='-', label='Optimal value')
            
            plt.xlabel('Service rate')
            plt.ylabel('Density')
            plt.title('Posterior of service rate')
            plt.legend()
            
            # Save the plot
            plt.savefig(f'./results/plots/rejectionABC_2d_inexact/rejectionABC_2d_inexact_posterior_serv_alpha={service_shape}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.kdeplot(Theta[:, 1].cpu().numpy(), fill=True, label='Posterior')

            # Add lines for 2.5%, 97.5% quantiles, posterior mean, posterior median, and true parameter
            plt.axvline(lower_threshold_arriv, color='b', linestyle='--', label='2.5% quantile')
            plt.axvline(upper_threshold_arriv, color='b', linestyle='--', label='97.5% quantile')
            plt.axvline(torch.mean(Theta[:, 1]).cpu().numpy(), color='g', linestyle='--', label='Posterior mean')
            plt.axvline(torch.median(Theta[:, 1]).cpu().numpy(), color='y', linestyle='--', label='Posterior median')
            plt.axvline(theta_opt[1], color='r', linestyle='-', label='Optimal value')
            
            plt.xlabel('Arrival rate')
            plt.ylabel('Density')
            plt.title('Posterior of arrival rate')
            plt.legend()
            
            # Save the plot
            plt.savefig(f'./results/plots/rejectionABC_2d_inexact/rejectionABC_2d_inexact_posterior_arriv_alpha={service_shape}.png')
            plt.close()
        
        df['average cpu time'][index] += (time.time() - start_time) / macro_count
    df['Width, Credible Set'][index] /= macro_count
    df['MSE, posterior mean'][index] /= macro_count
    # Create the directory if it doesn't exist
    os.makedirs('./results/dataframes/rejectionABC_2d_inexact', exist_ok=True)

    # Save the dataframe to a CSV file
    pd.DataFrame(df).to_csv(f'./results/dataframes/rejectionABC_2d_inexact/results_alpha={service_shape}.csv', index=False)
    index += 1


print(pd.DataFrame(df))

# # Create the directory if it doesn't exist
os.makedirs('./results/dataframes/rejectionABC_2d_inexact', exist_ok=True)

# Save the dataframe to a CSV file
pd.DataFrame(df).to_csv('./results/dataframes/rejectionABC_2d_inexact/results.csv', index=False)
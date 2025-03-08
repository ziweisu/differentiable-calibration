import torch
import numpy as np
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import scipy.stats
import scipy.linalg
import time
from src.scoring_rules import compute_median_l2_distance

# Set CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the stochastic volatility model
def stoc_volatility(phi, kappa, sigma, total_samples=30, replications=20000, constant_seed=None, device="cpu"):
    """
    Generate samples from a stochastic volatility model.
    
    Parameters:
    phi: Persistence parameter
    kappa: Scale parameter
    sigma: Volatility parameter
    total_samples: Number of time steps
    replications: Number of replications
    constant_seed: Random seed for reproducibility
    device: Computation device
    
    Returns:
    A tensor of shape [replications, total_samples] containing the returns
    """
    if constant_seed is not None:
        torch.manual_seed(constant_seed)
    
    # Generate noise for volatility and returns
    etas = torch.randn(replications, total_samples-1, device=device) * sigma
    epsilons = torch.randn(replications, total_samples, device=device)
    
    # Initial volatility
    h_0 = torch.randn(replications, device=device) * sigma / torch.sqrt(1 - phi**2)
     
    # Compute volatility process h_t
    h_t = [h_0]
    for t in range(1, total_samples):
        h_t.append(phi * h_t[-1] + etas[:, t-1])
    
    h_t = torch.stack(h_t, dim=1)

    # Compute returns y_t
    y_t = epsilons * kappa * torch.exp(0.5 * h_t)
    
    return y_t

# Define Riesz kernel for MMD
class Riesz:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        Compute the Riesz kernel: k(x,y) = -||x-y||^alpha
        
        Parameters:
        x: First input tensor [batch_size1, dim]
        y: Second input tensor [batch_size2, dim]
        
        Returns:
        Kernel matrix [batch_size1, batch_size2]
        """
        # Compute pairwise distances
        if x.dim() == 2 and y.dim() == 3:  # For the case y_sample[..., None]
            # Here x is [batch_size1, dim] and y is [batch_size2, 1, dim]
            y = y.squeeze(1)
        
        distances = torch.cdist(x, y)
        # Apply Riesz kernel: -||x-y||^alpha
        return -torch.pow(distances + 1e-8, self.alpha)

# Define Gaussian kernel for MMD
class Gaussian:
    def __init__(self, sigma=1):
        self.sigma = sigma
    
    def __call__(self, x, y):
        """
        Compute the Gaussian kernel: k(x,y) = exp(-||x-y||^2/(2*sigma^2))
        
        Parameters:
        x: First input tensor [batch_size1, dim]
        y: Second input tensor [batch_size2, dim]
        
        Returns:
        Kernel matrix [batch_size1, batch_size2]
        """
        # Compute pairwise distances
        if x.dim() == 2 and y.dim() == 3:  # For the case y_sample[..., None]
            # Here x is [batch_size1, dim] and y is [batch_size2, 1, dim]
            y = y.squeeze(1)
        
        distances_squared = torch.cdist(x, y, p=2).pow(2)
        # Apply Gaussian kernel
        return torch.exp(-distances_squared / (2 * self.sigma))

# Define the maximum mean discrepancy
class crps_multidim:
    def __init__(self, beta=1.0):
        self.beta=beta
    def __call__(self, observations, simulations):
        """
        Compute the continuous ranked probability score (CRPS) using MMD with a Riesz kernel.
        
        Parameters:
        observations: Observed data [batch_size, time_steps]
        simulations: Simulated data [replications, time_steps]
        
        Returns:
        The CRPS value
        """
        n_obs = observations.shape[0]
        n_sim = simulations.shape[0]
        
        # Compute ||x_i - x_j||
        X_obs = observations.unsqueeze(1)  # [n_obs, 1, time_steps]
        X_sim = simulations.unsqueeze(0)   # [1, n_sim, time_steps]
        
        # Compute pairwise distances
        pairwise_obs_obs = torch.sqrt(torch.sum((X_obs - X_obs.transpose(0, 1))**2, dim=2) + 1e-8) ** self.beta
        pairwise_sim_sim = torch.sqrt(torch.sum((X_sim - X_sim.transpose(0, 1))**2, dim=2) + 1e-8) ** self.beta
        pairwise_obs_sim = torch.sqrt(torch.sum((X_obs - X_sim)**2, dim=2) + 1e-8) ** self.beta
        
        # Use Riesz kernel: k(x, y) = -||x - y||^beta
        obs_obs_term = -torch.sum(pairwise_obs_obs) / (n_obs * (n_obs - 1)) if n_obs > 1 else 0
        sim_sim_term = -torch.sum(pairwise_sim_sim) / (n_sim * (n_sim - 1)) if n_sim > 1 else 0
        obs_sim_term = -torch.sum(pairwise_obs_sim) / (n_obs * n_sim)
        
        # MMD = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]
        mmd = obs_obs_term + sim_sim_term - 2 * obs_sim_term
        
        return mmd

class mmd_multidim:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, observations, simulations):
        """
        Compute the MMD with a Gaussian kernel.
        
        Parameters:
        observations: Observed data [batch_size, time_steps]
        simulations: Simulated data [replications, time_steps]
        
        Returns:
        The CRPS value
        """

        n_obs = observations.shape[0]
        n_sim = simulations.shape[0]
        
        # Compute ||x_i - x_j||
        X_obs = observations.unsqueeze(1)  # [n_obs, 1, time_steps]
        X_sim = simulations.unsqueeze(0)   # [1, n_sim, time_steps]

        # Compute pairwise distances
        pairwise_obs_obs = torch.exp(-torch.sum((X_obs - X_obs.transpose(0, 1))**2, dim=2) / (2*self.sigma) + 1e-8)
        pairwise_sim_sim = torch.exp(-torch.sum((X_sim - X_sim.transpose(0, 1))**2, dim=2) / (2*self.sigma) + 1e-8)
        pairwise_obs_sim = torch.exp(-torch.sum((X_obs - X_sim)**2, dim=2)/(2*self.sigma) + 1e-8)
        
        # Use Riesz kernel: k(x, y) = -||x - y||
        obs_obs_term = torch.sum(pairwise_obs_obs) / (n_obs * (n_obs - 1)) if n_obs > 1 else 0
        sim_sim_term = torch.sum(pairwise_sim_sim) / (n_sim * (n_sim - 1)) if n_sim > 1 else 0
        obs_sim_term = torch.sum(pairwise_obs_sim) / (n_obs * n_sim)
        
        # MMD = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]
        mmd = obs_obs_term + sim_sim_term - 2 * obs_sim_term
        
        return mmd

# SGD estimation with mini-batching
def sgd_estimation(simulation, observations, scoring_rule,
                  simulation_parameter_start, simulation_replications=1000,
                  number_epochs=200, lr_init=0.1, device="cpu", constant_seed=None,
                  batch_size=2000):
    """
    Stochastic gradient descent optimization for parameter estimation.
    
    Parameters:
    simulation: Function that generates samples from the model
    observations: Observed data
    scoring_rule: Function to compute the discrepancy (e.g., MMD)
    simulation_parameter_start: Initial parameter values
    simulation_replications: Number of model simulations per iteration
    number_epochs: Number of optimization epochs
    lr_init: Initial learning rate
    device: Computation device
    constant_seed: Random seed for reproducibility
    batch_size: Batch size for stochastic gradient computation
    
    Returns:
    Optimized parameters, loss history, and parameter history
    """
    torch_para = torch.nn.Parameter(torch.tensor(simulation_parameter_start, device=device))
    optimizer = torch.optim.Adam([torch_para], lr=lr_init)
    loss_list = []
    para_list = []
    
    n_observations = observations.shape[0]
    
    for k in range(number_epochs):
        epoch_loss = 0
        
        # Shuffle the observations
        indices = torch.randperm(n_observations)
        
        for start_idx in range(0, n_observations, batch_size):
            optimizer.zero_grad()
            
            end_idx = min(start_idx + batch_size, n_observations)
            batch_indices = indices[start_idx:end_idx]
            batch_observations = observations[batch_indices]
            
            # Transform parameters:
            # phi is in (-1,1), kappa and sigma are positive
            phi = torch.tanh(torch_para[0]/2)
            kappa = torch.exp(torch_para[1])
            sigma = torch.exp(torch_para[2]/2)
            
            # Generate samples from the model
            y_sample = simulation(phi, kappa, sigma, replications=simulation_replications, 
                                  device=device, constant_seed=constant_seed)
            
            # Compute loss and gradients
            loss = scoring_rule(batch_observations, y_sample)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - start_idx)
        
        avg_epoch_loss = epoch_loss / n_observations
        loss_list.append(avg_epoch_loss)
        para_list.append(torch_para.detach().cpu().numpy().copy())
    
    return torch_para.detach().cpu().numpy(), loss_list, para_list

# Compute Hessian approximation
def hessian_approx(simulation, observations, scoring_rule,
                  simulation_parameter, device="cpu", constant_seed=None, 
                  replications=1000):
    """
    Approximate the Hessian matrix of the scoring rule with respect to the parameters.
    
    Parameters:
    simulation: Function that generates samples from the model
    observations: Observed data
    scoring_rule: Function to compute the discrepancy (e.g., MMD)
    simulation_parameter: Parameter values
    device: Computation device
    constant_seed: Random seed for reproducibility
    replications: Number of model simulations
    
    Returns:
    Approximated Hessian matrix
    """
    torch_para = torch.tensor(simulation_parameter, requires_grad=True, device=device)
    
    def func(theta):
        phi = torch.tanh(theta[0]/2)
        kappa = torch.exp(theta[1]) 
        sigma = torch.exp(theta[2]/2)
        y_sample = simulation(phi, kappa, sigma, replications=replications, 
                              constant_seed=constant_seed, device=device)
        return scoring_rule(observations, y_sample)
    
    # Initialize Hessian matrix
    n_params = len(simulation_parameter)
    hessian = torch.zeros((n_params, n_params), device=device)
    
    # Compute first-order gradient
    loss = func(torch_para)
    grad = torch.autograd.grad(loss, torch_para, create_graph=True)[0]
    
    # Compute Hessian by differentiating the gradient
    for i in range(n_params):
        grad_i = grad[i]
        grad_grad_i = torch.autograd.grad(grad_i, torch_para, retain_graph=(i < n_params-1))[0]
        hessian[i] = grad_grad_i
    
    return hessian.detach().cpu().numpy()

# Compute asymptotic variance using Jacobian approximation
def asymp_var_jacobian_approx(simulation, observations, simulation_parameter, kernel,
                              device="cpu", constant_seed=None, replications=5000, jacobian_type=None):
    """
    Compute the asymptotic variance of the MMD estimator using Jacobian approximation.
    
    Parameters:
    simulation: Function that generates samples from the model
    observations: Observed data
    simulation_parameter: Parameter values
    kernel: Kernel function for MMD
    device: Computation device
    constant_seed: Random seed for reproducibility
    replications: Number of model simulations
    jacobian_type: Method to compute the Jacobian
    
    Returns:
    Asymptotic variance-covariance matrix
    """
    def kernel_J(theta):
        # Transform parameters for simulation
        phi = torch.tanh(theta[0]/2)
        kappa = torch.exp(theta[1])
        sigma = torch.exp(theta[2]/2)
        
        # Generate samples
        y_sample = simulation(phi, kappa, sigma, replications=replications, 
                              device=device, constant_seed=constant_seed)
        
        # Compute kernel values between samples and observations
        K = torch.mean(kernel(y_sample[..., None], observations), dim=0)
        return K
    
    # Compute the Jacobian of the kernel
    if jacobian_type == "built-in":
        # Compute with torch.autograd.functional.jacobian (slower)
        torch_para = torch.tensor(simulation_parameter, device=device)
        J = jacobian(kernel_J, torch_para)
    else:
        # Compute with autograd (faster)
        torch_para = torch.tensor(simulation_parameter, requires_grad=True, device=device)
        
        # Transform parameters for simulation
        phi = torch.tanh(torch_para[0]/2)
        kappa = torch.exp(torch_para[1])
        sigma = torch.exp(torch_para[2]/2)
        
        # Generate samples
        y_sample = simulation(phi, kappa, sigma, replications=replications, 
                              device=device, constant_seed=constant_seed)
        
        # Compute Jacobian rows
        J = torch.zeros((len(observations), 3), device=device)
        for i in range(len(observations)):
            # Ensure observations[i] has the right dimension
            obs_i = observations[i].unsqueeze(0)  # Add batch dimension
            torch.mean(kernel(y_sample, obs_i)).backward(retain_graph=(i < len(observations)-1))
            J[i, :] = torch_para.grad.clone()
            torch_para.grad.zero_()
    
    # Compute the asymptotic variance
    sigma_hat = 4 * torch.cov(J.T)
    
    return sigma_hat.detach().cpu().numpy()

# Compute confidence intervals
def conf_int(simulation, observations, scoring_rule, simulation_parameter, kernel,
             device="cpu", constant_seed=0, replications=5000, jacobian_type=None):
    """
    Compute confidence intervals for the estimated parameters.
    
    Parameters:
    simulation: Function that generates samples from the model
    observations: Observed data
    scoring_rule: Function to compute the discrepancy (e.g., MMD)
    simulation_parameter: Estimated parameter values
    kernel: Kernel function for MMD
    device: Computation device
    constant_seed: Random seed for reproducibility
    replications: Number of model simulations
    jacobian_type: Method to compute the Jacobian
    
    Returns:
    Hessian matrix and asymptotic variance-covariance matrix
    """
    # Compute Hessian approximation
    H_hat = hessian_approx(simulation, observations, scoring_rule, simulation_parameter, 
                           device=device, constant_seed=constant_seed, replications=replications)
    
    # Compute asymptotic variance using Jacobian
    sigma_hat = asymp_var_jacobian_approx(simulation, observations, simulation_parameter, kernel,
                                          device=device, constant_seed=constant_seed, 
                                          replications=replications, jacobian_type=jacobian_type)
    
    return H_hat, sigma_hat

def compute_median_l2_distance(X):
    """
    More efficient implementation using torch.cdist
    
    Parameters:
    X: Tensor of shape [batch_size, time_steps]
    
    Returns:
    Median of pairwise distances
    """
    # Compute pairwise Euclidean distances between time series
    pairwise_distances = torch.cdist(X, X, p=2)
    
    # Get non-zero distances (upper triangular without diagonal)
    mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1) > 0
    non_zero_distances = pairwise_distances[mask]
    
    # Compute median
    if non_zero_distances.numel() > 0:
        median_distance = torch.median(non_zero_distances)
    else:
        # Fallback if no non-zero distances
        median_distance = torch.tensor(1.0, device=X.device)
    
    return median_distance

def run_experiment(experiment_name, macro_count=100):
    """
    Run a simulation study similar to the second code's structure.
    """
    if experiment_name == 'stochastic_volatility_Gaussian' or 'stochastic_volatility_Gaussian_corrected':
        kernel = Gaussian()
        distance = mmd_multidim()
    else:
        kernel = Riesz()
        distance = crps_multidim()

    # Setup random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define true parameters
    true_params = [(np.exp(0.98)-1)/(np.exp(0.98)+1), np.exp(0.65), np.exp(0.15/2)]  # [phi, kappa, sigma]
    
    # First, use a large sample to get "optimal" parameters
    x_sample_large = stoc_volatility(
        torch.tensor(true_params[0]), 
        torch.tensor(true_params[1]), 
        torch.tensor(true_params[2]), 
        replications=1000, 
        device=device
    )
    
    # Initial parameter guess (internal parameterization)
    theta_init = np.array([2*np.arctanh(0.5), np.log(1.9), 2*np.log(1)])
    
    # Estimate "optimal" parameters
    if False:
        _, _, theta_list = sgd_estimation(
            stoc_volatility, 
            x_sample_large, 
            crps_multidim, 
            theta_init,
            simulation_replications=100,
            number_epochs=300, 
            lr_init=1e-3, 
            device=device
        )
        theta_opt = np.mean(theta_list[-100:], axis=0)
    else:
        theta_opt = np.array([2*np.arctanh(true_params[0]), np.log(true_params[1]), 2*np.log(true_params[2])])
    
    # Convert to original parameterization
    phi_opt = np.tanh(theta_opt[0]/2)
    kappa_opt = np.exp(theta_opt[1])
    sigma_opt = np.exp(theta_opt[2]/2)
    
    print(f"Optimal parameters: phi={phi_opt:.4f}, kappa={kappa_opt:.4f}, sigma={sigma_opt:.4f}")
    
    # Initialize results dataframe
    df = {
        'n': [100, 100, 100, 100, 100],
        'm': [10, 50, 100, 500, 1000],
        'MSE, phi': [0, 0, 0, 0, 0],
        'MSE, kappa': [0, 0, 0, 0, 0],
        'MSE, sigma': [0, 0, 0, 0, 0],
        'CovProb': [0, 0, 0, 0, 0],
        'Average CPU Time': [0, 0, 0, 0, 0]
    }

    
    # Run experiments for different sample sizes
    for idx, m in enumerate(df['m']):
        n = df['n'][idx]
        print(f"Running experiment for m={m}, n={n}")
        
        for k in tqdm(range(macro_count), desc=f"Sample size m={m}"):
            # Set specific seed for this run
            seed = k * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate data from true model
            x_sample = stoc_volatility(
                torch.tensor(true_params[0]), 
                torch.tensor(true_params[1]), 
                torch.tensor(true_params[2]), 
                replications=m, 
                device=device
            )
            if experiment_name == 'stochastic_volatility_Gaussian' or 'stochastic_volatility_Gaussian_corrected':
                sigma = torch.sqrt(compute_median_l2_distance(x_sample)** 2 / 2)
                kernel.sigma = sigma
                distance.sigma = sigma

            # Measure CPU time
            start_time = time.time()
            
            # Estimate parameters
            theta_tilde, _, para_history = sgd_estimation(
                stoc_volatility, 
                x_sample, 
                distance, 
                theta_init,
                simulation_replications=n,
                number_epochs=500,  # Reduced for demonstration
                lr_init=1e-3, 
                device=device, 
                batch_size=min(100, m)  # Adjust batch size based on m
            )

            # Transform to original parameterization
            phi_est = np.tanh(theta_tilde[0]/2)
            kappa_est = np.exp(theta_tilde[1])
            sigma_est = np.exp(theta_tilde[2]/2)
            
            # Compute MSEs
            df['MSE, phi'][idx] += ((phi_est - phi_opt) ** 2) / macro_count
            df['MSE, kappa'][idx] += ((kappa_est - kappa_opt) ** 2) / macro_count
            df['MSE, sigma'][idx] += ((sigma_est - sigma_opt) ** 2) / macro_count
            
            # Compute confidence intervals
            try:
                H_hat, sigma_hat = conf_int(
                    stoc_volatility, 
                    x_sample, 
                    distance, 
                    theta_tilde, 
                    kernel,
                    device=device,
                    replications=n
                )
                
                # Check if the true parameter is in the confidence interval
                sigma_hat_inv = np.linalg.inv(sigma_hat)
                H_hat_inv = np.linalg.inv(H_hat)
                
                # Calculate the difference between estimated and true parameters
                param_diff = theta_tilde - theta_opt
                
                # Compute the chi-square statistic
                ci_stat = np.linalg.norm((np.sqrt(m) * scipy.linalg.sqrtm(sigma_hat_inv).dot(H_hat)).dot(param_diff))
                
                # Check if the statistic is less than the critical value
                if ci_stat ** 2 <= scipy.stats.chi2.ppf(0.95, df=3):
                    df['CovProb'][idx] += 1/macro_count
            except np.linalg.LinAlgError:
                print(f"Warning: Linear algebra error in run {k} for m={m}. Skipping confidence interval check.")
            
            # Record CPU time
            df['Average CPU Time'][idx] += (time.time() - start_time) / macro_count
            
            # Create visualization for the first run
            if k == 0 and idx == 4:  # For m=1000
                # Convert parameter history to original scale
                phi_history = np.tanh(np.array([p[0] for p in para_history])/2)
                kappa_history = np.exp(np.array([p[1] for p in para_history]))
                sigma_history = np.exp(np.array([p[2] for p in para_history])/2)
                
                # Create the plots directory
                os.makedirs(f'./results/plots/{experiment_name}', exist_ok=True)
                
                # Plot parameter convergence
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].plot(phi_history)
                axes[0].axhline(true_params[0], color='r', linestyle='--', label='True value')
                axes[0].axhline(phi_opt, color='g', linestyle='-.', label='Optimal value')
                axes[0].set_title('phi convergence')
                axes[0].set_xlabel('Epoch')
                axes[0].legend()
                
                axes[1].plot(kappa_history)
                axes[1].axhline(true_params[1], color='r', linestyle='--', label='True value')
                axes[1].axhline(kappa_opt, color='g', linestyle='-.', label='Optimal value')
                axes[1].set_title('kappa convergence')
                axes[1].set_xlabel('Epoch')
                axes[1].legend()
                
                axes[2].plot(sigma_history)
                axes[2].axhline(true_params[2], color='r', linestyle='--', label='True value')
                axes[2].axhline(sigma_opt, color='g', linestyle='-.', label='Optimal value')
                axes[2].set_title('sigma convergence')
                axes[2].set_xlabel('Epoch')
                axes[2].legend()
                
                plt.tight_layout()
                plt.savefig(f'./results/plots/{experiment_name}/parameter_convergence_m100.png')
                plt.close()
        
        # Save interim results
        os.makedirs(f'./results/dataframes/{experiment_name}', exist_ok=True)
        pd.DataFrame(df).to_csv(f'./results/dataframes/{experiment_name}/results_m={m}.csv', index=False)
    
    # Print and save the final results
    results_df = pd.DataFrame(df)
    print(results_df)
    results_df.to_csv(f'./results/dataframes/{experiment_name}/final_results.csv', index=False)
    
    # Create summary plots
    plt.figure(figsize=(15, 10))
    
    # Plot MSE for each parameter vs sample size
    plt.subplot(2, 2, 1)
    plt.plot(df['m'], df['MSE, phi'], 'o-', label='phi')
    plt.plot(df['m'], df['MSE, kappa'], 's-', label='kappa')
    plt.plot(df['m'], df['MSE, sigma'], '^-', label='sigma')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (m)')
    plt.ylabel('MSE')
    plt.title('MSE vs Sample Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot coverage probability vs sample size
    plt.subplot(2, 2, 2)
    plt.plot(df['m'], df['CovProb'], 'o-')
    plt.axhline(0.95, color='r', linestyle='--', label='Expected (95%)')
    plt.xscale('log')
    plt.xlabel('Sample Size (m)')
    plt.ylabel('Coverage Probability')
    plt.title('Coverage Probability vs Sample Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot CPU time vs sample size
    plt.subplot(2, 2, 3)
    plt.plot(df['m'], df['Average CPU Time'], 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (m)')
    plt.ylabel('Average CPU Time (s)')
    plt.title('CPU Time vs Sample Size')
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{experiment_name}/simulation_summary.png')
    plt.close()
    
    return results_df

if __name__ == "__main__":
    # Create necessary directories
    experiment_name = 'stochastic_volatility_Gaussian_corrected'
    os.makedirs(f'./results/plots/{experiment_name}', exist_ok=True)
    os.makedirs(f'./results/dataframes/{experiment_name}', exist_ok=True)
    
    # kernel and distance must match: Riesz with crps, Gaussian with mmd
    results = run_experiment(experiment_name=experiment_name, macro_count=100)
    print("Simulation study completed successfully!")
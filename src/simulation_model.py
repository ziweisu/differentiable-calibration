import torch
from torch.distributions.gamma import Gamma
from torch.distributions.exponential import Exponential
from torch.distributions.uniform import Uniform

def lindley(service_rate, service_shape_parameter=1, sample_period=50, burn_in_period=10, replications=100,
            constant_seed=None, device = "cpu", softplus=False):
    """
    Estimates waiting time with m customers, after warming up for d
    Lindley approximation for waiting time in an M/G/1 queue
    """
    if constant_seed is not None:
        torch.manual_seed(constant_seed)
    recurrsionvar = torch.zeros(replications, device=device)
    totaltime = torch.zeros(replications, device=device)
    service_times, arrival_times = pull_simulation_drivers(replications, sample_period + burn_in_period,
                                                           service_shape_parameter, device = device)
    service_ratio = service_shape_parameter / service_rate
    for i in range(sample_period + burn_in_period):  # use one loop, not two
        # Simulate from M/G/1
        if softplus is False:
            recurrsionvar = torch.nn.functional.relu(recurrsionvar + service_ratio * service_times[:, i]
                                                 - arrival_times[:, i])
        else:
            recurrsionvar = torch.nn.functional.softplus(beta=10, threshold=0.1, input = recurrsionvar + service_ratio * service_times[:, i]
                                                 - arrival_times[:, i])
        if i >= burn_in_period:
            totaltime += recurrsionvar
    return totaltime / sample_period

def queue_system(service_rates, service_shape_parameter = 1, sample_period = 50, 
                 burn_in_period = 10, replications = 100, constant_seed = None):
    """
    Simulates a queue system with one queue and two consecutive servers 
    and returns the average waiting time
    """
    if constant_seed is not None:
        torch.manual_seed(constant_seed)

    recurrsionvar_1 = torch.zeros((replications))
    recurrsionvar_2 = torch.zeros((replications))
    totaltime = torch.zeros((replications))
    service_times_1, arrival_times_1 = pull_simulation_drivers(replications, sample_period + burn_in_period + 1,
                                                           service_shape_parameter[0])
    service_times_2, _ = pull_simulation_drivers(replications, sample_period + burn_in_period, 
                                               service_shape_parameter[1])
    arrival_times_2 = torch.zeros((replications, sample_period + burn_in_period))
    service_ratio_1 = service_shape_parameter[0] / service_rates[0]
    service_ratio_2 = service_shape_parameter[1] / service_rates[1]
    for i in range(sample_period + burn_in_period):
        if i == 0:
            recurrsionvar_1 = torch.nn.functional.relu(recurrsionvar_1 + service_ratio_1 * service_times_1[:, i]
                                                 - arrival_times_1[:, i])
            arrival_times_2[:, i] = recurrsionvar_1 + service_ratio_1 * service_times_1[:, i+1] + arrival_times_1[:, i]
        else:
            recurrsionvar_1_new = torch.nn.functional.relu(recurrsionvar_1 + service_ratio_1 * service_times_1[:, i])
            arrival_times_2[:, i] = recurrsionvar_1_new - recurrsionvar_1 + service_ratio_1 * (service_times_1[:, i+1] - service_times_1[:, i]) + arrival_times_1[:, i]
            recurrsionvar_1 = recurrsionvar_1_new
        recurrsionvar_2 = torch.nn.functional.relu(recurrsionvar_2 + service_ratio_2 * service_times_2[:, i] - arrival_times_2[:, i])
        if i >= burn_in_period:
            totaltime += recurrsionvar_2
    return totaltime / sample_period

def pull_simulation_drivers(replications, total_samples, service_shape_parameter, 
                            device = "cpu"):
    service_shape_paras = torch.empty((replications, total_samples), device=device).fill_(service_shape_parameter).to(device)
    service_times = Gamma(service_shape_paras, service_shape_paras).sample().detach().to(device)
    arrival_times = Exponential(torch.ones((replications, total_samples), device=device)).sample().detach().to(device)
    return service_times, arrival_times


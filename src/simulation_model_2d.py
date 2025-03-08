import torch
from torch.distributions.gamma import Gamma
from torch.distributions.exponential import Exponential
from torch.distributions.uniform import Uniform

def lindley(service_rate, arrival_rate, service_shape_parameter=1, arrival_shape_parameter=1, 
            sample_period=10, burn_in_period=0, replications=100, constant_seed=None, device="cpu",
            softplus=False):
    # Estimate the waiting time in a G/G/1 queue with Lindley's recursion
    if constant_seed is not None:
        torch.manual_seed(constant_seed)
    recurrsionvar = torch.zeros(replications).to(device)
    totaltime = torch.zeros(replications).to(device)
    service_times, arrival_times = pull_simulation_drivers(replications, sample_period + burn_in_period,
                                                           service_shape_parameter, arrival_shape_parameter, 
                                                           device = device)
    service_ratio = service_shape_parameter / service_rate
    arrival_ratio = arrival_shape_parameter / arrival_rate
    for i in range(sample_period + burn_in_period):  # use one loop, not two
        # Simulate from G/G/1
        if softplus is False:
            recurrsionvar = torch.nn.functional.relu(recurrsionvar + service_ratio * service_times[:, i]
                                                 - arrival_ratio * arrival_times[:, i])
        else:
            recurrsionvar = torch.nn.functional.softplus(beta=10, threshold=0.1, input = recurrsionvar + service_ratio * service_times[:, i]
                                                 - arrival_ratio * arrival_times[:, i])
        if i >= burn_in_period:
            totaltime += recurrsionvar
    return totaltime / sample_period

def queue_system(service_rates, service_shape_parameter = 1, sample_period = 50, 
                 burn_in_period = 10, replications = 100, constant_seed = None):
    # Simulates a queue system with one queue and two consecutive servers and returns the average waiting time
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
                            arrival_shape_parameter, device="cpu"):
    # Simulate the Gamma arrival and service times
    service_shape_paras = torch.empty((replications, total_samples)).fill_(service_shape_parameter).to(device)
    arrival_shape_paras = torch.empty((replications, total_samples)).fill_(arrival_shape_parameter).to(device)
    service_times = Gamma(service_shape_paras, service_shape_paras).sample().to(device)
    arrival_times = Gamma(arrival_shape_paras, arrival_shape_paras).sample().to(device)
    return service_times, arrival_times


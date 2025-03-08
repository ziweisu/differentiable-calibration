import torch
import numpy as np
from torch.autograd.functional import jacobian, hessian
from scipy.stats import norm
from functools import partial

def sgd_estimation(simulation, observations, scoring_rule,
                   simulation_parameter_start, service_shape_parameter=1, 
                   arrival_shape_parameter=1, simulation_replications=1000,
                   number_epochs=200, lr_init=0.1, device="cpu", constant_seed=None):
    # Initialization
    torch_para = torch.tensor(simulation_parameter_start, requires_grad=True)
    optimizer = torch.optim.Adam([torch_para], lr=lr_init)
    loss_list = []
    para_list = []
    for k in range(number_epochs):
        optimizer.param_groups[0]['lr'] = lr_init / np.sqrt(1. + k)
        if constant_seed is None:
            y_sample = simulation(torch_para[0], torch_para[1], service_shape_parameter=service_shape_parameter,
                                  arrival_shape_parameter=arrival_shape_parameter, replications=simulation_replications, 
                                  device=device).to(device)
        else:
            y_sample = simulation(torch_para[0], torch_para[1], service_shape_parameter=service_shape_parameter,
                                  arrival_shape_parameter=arrival_shape_parameter, replications=simulation_replications, 
                                  device=device, constant_seed=constant_seed).to(device)
        optimizer.zero_grad()
        loss = scoring_rule(observations, y_sample)
        loss.backward()
        optimizer.step()
        if True:
            loss_list.append(loss.cpu().detach().numpy())
            para_list.append(torch_para.cpu().detach().numpy().copy()) 
    return torch_para.detach().cpu().numpy(), loss_list, para_list

def sgd_estimation_log(simulation, observations, scoring_rule, simulation_parameter_start, service_shape_parameter=1, arrival_shape_parameter=1, simulation_replications=1000, number_epochs=200, lr_init=0.1, device="cpu", constant_seed=None):
    # Initialization
    log_torch_para = torch.tensor(np.log(simulation_parameter_start), requires_grad=True)
    optimizer = torch.optim.Adam([log_torch_para], lr=lr_init)
    loss_list = []
    para_list = []

    for k in range(number_epochs):
        optimizer.param_groups[0]['lr'] = lr_init / np.sqrt(1. + k)
        if constant_seed is None:
            y_sample = simulation(torch.exp(log_torch_para[0]), torch.exp(log_torch_para[1]), service_shape_parameter=service_shape_parameter, arrival_shape_parameter=arrival_shape_parameter, replications=simulation_replications, device=device).to(device)
        else:
            y_sample = simulation(torch.exp(log_torch_para[0]), torch.exp(log_torch_para[1]), service_shape_parameter=service_shape_parameter, arrival_shape_parameter=arrival_shape_parameter, replications=simulation_replications, device=device, constant_seed=constant_seed).to(device)

        optimizer.zero_grad()
        loss = scoring_rule(observations, y_sample)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.cpu().detach().numpy())
        para_list.append(torch.exp(log_torch_para).cpu().detach().numpy().copy())

    return torch.exp(log_torch_para).detach().cpu().numpy(), loss_list, para_list

def hessian_approx(simulation, observations, scoring_rule,
                   simulation_parameter, kernel, service_shape_parameter=1,
                   arrival_shape_parameter=1, device="cpu", constant_seed=None, 
                   replications=5000, hessian_type="finite_difference"):
    torch_para = torch.tensor(simulation_parameter)
    if hessian_type == "built-in":
        softplus = True
    else:
        softplus = False
    def funcsimpl(theta):
        y_sample = simulation(theta[0], theta[1], 
                              service_shape_parameter=service_shape_parameter,
                              arrival_shape_parameter=arrival_shape_parameter,
                              replications=replications, constant_seed=constant_seed,
                              device=device, softplus=softplus).to(device)
        return scoring_rule(observations, y_sample) 
    if hessian_type == "built-in":
        # built-in hessian in Pytorch
        return hessian(funcsimpl, torch_para).detach()
    elif hessian_type == "finite_difference":
        # use finite difference to estimate hessian
        epsilon_1 = 0.2
        epsilon_2 = 0.2
        J_current = jacobian(funcsimpl, torch_para)
        J_epsilon_1 = jacobian(funcsimpl, torch_para + torch.tensor([epsilon_1, 0]))
        J_epsilon_2 = jacobian(funcsimpl, torch_para + torch.tensor([0, epsilon_2]))
        H = torch.zeros((2, 2))
        H[0, 0] = (J_epsilon_1[0] - J_current[0]) / epsilon_1
        H[1, 0] = H[0, 1] = (J_epsilon_2[0] - J_current[0]) / epsilon_2
        H[1, 1] = (J_epsilon_2[1] - J_current[1]) / epsilon_2
        if False:
            print(H)
        return H

def asymp_var_jacobian_approx(simulation, observations, simulation_parameter, kernel,
                              service_shape_parameter=1, arrival_shape_parameter=1,
                                device="cpu", constant_seed=None, replications=5000, jacobian_type=None):
    def kernel_J(theta):
        y_sample = simulation(theta[0], theta[1], service_shape_parameter=service_shape_parameter, 
                              arrival_shape_parameter=arrival_shape_parameter, replications=replications, 
                              device=device, constant_seed=constant_seed).to(device)
        K = torch.mean(kernel(y_sample[..., None], observations), dim = 0)
        return K
    # Compute the jacobian of the kernel
    if jacobian_type == "built-in":
       # compute with Jacobian is slower than autograd
       torch_para = torch.tensor(simulation_parameter)
       J = jacobian(kernel_J, torch_para)
    else:
        torch_para = torch.tensor(simulation_parameter, requires_grad=True)
        y_sample = simulation(torch_para[0], torch_para[1], service_shape_parameter=service_shape_parameter,
                              arrival_shape_parameter=arrival_shape_parameter, replications=replications, device=device,
                              constant_seed=constant_seed).to(device)
        J = torch.zeros((len(observations), 2))
        for i in range(len(observations)):
            torch.mean(kernel(y_sample, observations[i])).backward(retain_graph=True)
            #grad = torch.autograd.grad(torch.mean(kernel(y_sample, observations[i])), torch_para, retain_graph=True)
            J[i, :] = torch_para.grad
            torch_para.grad.zero_()
    sigma_hat = 4 * torch.cov(J.T)
    if False:
        print("The estimated asymptotic variance of the Jacobian is {}".format(sigma_hat))
    return sigma_hat

def conf_int(simulation, observations, scoring_rule, simulation_parameter, kernel, 
             service_shape_parameter=1, arrival_shape_parameter=1, device="cpu", 
             constant_seed=0, replications=5000, hessian_type="built-in", jacobian_type=None):
    # compute hessian approximation
    H_hat = hessian_approx(simulation, observations, scoring_rule, 
                           simulation_parameter, kernel,
                           service_shape_parameter=service_shape_parameter,
                           arrival_shape_parameter=arrival_shape_parameter, 
                           device=device, constant_seed=constant_seed, 
                           replications=replications, hessian_type=hessian_type)

    # compute asymptotic variance of the jacobian
    sigma_hat = asymp_var_jacobian_approx(simulation, observations, simulation_parameter, 
                                          kernel, service_shape_parameter=service_shape_parameter,
                                          arrival_shape_parameter=arrival_shape_parameter, 
                                          device=device, constant_seed=constant_seed, 
                                          replications=replications, jacobian_type=jacobian_type)
    if False:
        print("The estimated asymptotic variance of the expected score est. is {}".format(var_hat))
    # Return H_hat, sigma_hat_inv
    return H_hat.numpy(), sigma_hat

# eval Hessian matrix
def eval_hessian(loss_grad, simulation_parameter):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], simulation_parameter, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()
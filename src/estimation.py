import torch
import numpy as np
from torch.autograd.functional import jacobian, hessian
from scipy.stats import norm
#from torch.func import hessian

def sgd_estimation(simulation, observations, scoring_rule,
                   simulation_parameter_start, simulation_replications=1000,
                   number_epochs=200, lr_init=0.1, device = "cpu", constant_seed=None):
    torch_para = torch.tensor(simulation_parameter_start, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([torch_para], lr=lr_init)
    loss_list = []
    para_list = []
    for k in range(number_epochs):
        optimizer.param_groups[0]['lr'] = lr_init / np.sqrt(1. + k)
        if constant_seed is None:
            y_sample = simulation(torch_para, replications=simulation_replications, device=device).to(device)
        else:
            y_sample = simulation(torch_para, replications=simulation_replications, 
                                  device=device, constant_seed=constant_seed).to(device)
        optimizer.zero_grad()
        loss = scoring_rule(observations, y_sample)
        loss.backward()
        optimizer.step()
        if True:
            loss_list.append(loss.cpu().detach().numpy())
            para_list.append(torch_para.cpu().detach().numpy().copy()) 
    return torch_para.detach().cpu().numpy(), loss_list, para_list


def hessian_approx(simulation, observations, scoring_rule,
                   simulation_parameter, hessian_type="finite_difference", device="cpu"):
    constant_seed = torch.random.initial_seed()
    torch_para = torch.tensor(simulation_parameter, device=device)
    if hessian_type == "built-in":
        softplus = True
    else:
        softplus = False
    def funcsimpl(theta):
        replications = 5000
        y_sample = simulation(theta, replications=replications, device=device, softplus=softplus).to(device)
        return scoring_rule(observations, y_sample) 
    if hessian_type == "built-in":
        # built-in hessian in Pytorch, tricky to use
        #return hessian(funcsimpl, torch_para).detach().numpy()
        # functorch hessian
        return hessian(funcsimpl, torch_para).detach()
    elif hessian_type == "finite_difference":
        # use finite difference to estimate hessian
        epsilon = 0.1   
        H = ((jacobian(funcsimpl, torch_para + epsilon) - jacobian(funcsimpl, torch_para)) / epsilon)
        if False:
            print(H)
        return H


def asymp_var_jacobian_approx(simulation, observations, simulation_parameter, 
                              kernel, device = "cpu", replications=5000):
    constant_seed = torch.random.initial_seed()
    def kernel_J(theta):
        y_sample = simulation(theta, replications=replications).to(device)
        K = torch.mean(kernel(y_sample[..., None], observations), dim = 0)
        return K
    # Compute the derivative of the kernel
    if False:
       # compute with Jacobian is slower than autograd
       torch_para = torch.tensor(simulation_parameter)
       J = jacobian(kernel_J, torch_para)
    else:
        torch_para = torch.tensor(simulation_parameter, requires_grad=True)
        y_sample = simulation(torch_para, replications=replications).to(device)
        J = torch.zeros((len(observations)))
        for i in range(len(observations)):
            torch.mean(kernel(y_sample, observations[i])).backward(retain_graph=True)
            J[i] = torch_para.grad.numpy().item()
            torch_para.grad.zero_()
    sigma_hat = 4 * torch.var(J)

    if False:
        print("The estimated asymptotic variance of the Jacobian is {}".format(sigma_hat))

    return sigma_hat

def conf_int(simulation, observations, scoring_rule, simulation_parameter, 
             kernel, device="cpu", alpha = 0.05):
    # compute hessian approximation
    H_hat = hessian_approx(simulation, observations, scoring_rule, 
                           simulation_parameter, device=device)
    # compute asymptotic variance of the jacobian
    sigma_hat = asymp_var_jacobian_approx(simulation, observations, simulation_parameter, 
                                          kernel, device=device)
    # Compute asymptotic variance of the expected score estimator
    var_hat = sigma_hat / (H_hat ** 2)
    if False:
        print("The estimated asymptotic variance of the expected score est. is {}".format(var_hat))

    # Construct CI
    var_hat = var_hat.detach().cpu().item()
    half_width = norm.ppf(1 - alpha / 2) * np.sqrt(var_hat / len(observations))
    width = 2 * half_width
    ci_lower = simulation_parameter - half_width
    ci_upper = simulation_parameter + half_width
    if False:
        print("The CI is {}".format([ci_lower, ci_upper]))
    return ci_lower, ci_upper, width, var_hat

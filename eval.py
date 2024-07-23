import torch
import numpy as np


def Survival(model, estimate, x, steps=200):
    device = x.device
    u = torch.ones((x.shape[0],), device=device) * 0.001
    time_steps = torch.linspace(1e-4, 1, steps=steps, device=device).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    t_max = t_max_model.reshape(-1,1).repeat(1, steps)
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps), device=device)
    surv2 = torch.zeros((x.shape[0], steps), device=device)
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i], x)
        surv2[:,i] = estimate.survival(time_steps[:,i], x)
    return surv1, surv2, time_steps, t_max_model


def surv_diff(model, estimate, x, steps):
    surv1, surv2, time_steps, t_m = Survival(model, estimate, x, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
    return torch.mean(integ/t_m)


def SurvivalNDE(model, estimate, x, idx, steps=200):
    device = 'cpu'
    x_cpu = x.to('cpu')
    u = torch.ones((x.shape[0],), device=device) * 0.001
    time_steps = torch.linspace(1e-4, 1, steps=steps, device=device).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x_cpu, u)
    t_max = t_max_model.reshape(-1,1).repeat(1, steps)
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps), device=device)
    surv2 = torch.zeros((x.shape[0], steps), device=device)
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i], x_cpu)
        surv2[:,i] = torch.exp(estimate.forward_S(x, time_steps[:,i].to(x.device).reshape(-1,1), mask=0)[:,idx])
    return surv1, surv2.to('cpu'), time_steps, t_max_model


def surv_diff_NDE(model, estimate, x, idx, steps):
    surv1, surv2, time_steps, t_m = SurvivalNDE(model, estimate, x, idx, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
    return torch.mean(integ/t_m)

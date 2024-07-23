from DGP_models import Weibull_linear
from model import Weibull_log_linear
from Copula import  Clayton
from utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from eval import surv_diff


if __name__ =="__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    nf = 10
    n_train = 10000
    n_val = 5000
    n_test = 5000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, device='cpu')
    
    dgp1 = Weibull_linear(nf, alpha=17, gamma=3, device='cpu')
    dgp2 = Weibull_linear(nf, alpha=16, gamma=3, device='cpu')
    dgp3 = Weibull_linear(nf, alpha=17, gamma=4, device='cpu')


    dgp1.coeff = torch.rand((nf,), device='cpu')
    dgp2.coeff = torch.rand((nf,), device='cpu')
    dgp3.coeff = torch.rand((nf,), device='cpu')
    copula_dgp = 'clayton'
    theta_dgp = 3.0
    eps = 1e-3
    train_dict, val_dict, test_dict = \
        generate_data(x_dict, dgp1, dgp2, dgp3, device = 'cpu', copula = copula_dgp, theta = theta_dgp)
    

    plt.hist(train_dict['E'])
    plt.savefig('E.png')
    plt.cla()


    copula = Clayton(theta_dgp, eps, 'cpu')
    best_loss = loss_triple(dgp1, dgp2, dgp3, train_dict, copula)
    copula = Clayton(0.1, eps, 'cpu')
    copula.enable_grad()



    model1 = Weibull_log_linear(nf, 2,2,'cpu')
    model1.enable_grad()
    model2 = Weibull_log_linear(nf, 2,2,'cpu')
    model2.enable_grad()
    model3 = Weibull_log_linear(nf, 2,2,'cpu')
    model3.enable_grad()
    
    copula_grad = []
    train_loss_log = []
    val_loss_log = []

    optimizer = torch.optim.Adam([
        {'params': model1.parameters(), "lr": 1e-2},
        {'params': model2.parameters(), "lr": 1e-2},
        {'params': model3.parameters(), "lr": 1e-2},
        {"params":copula.parameters(), "lr":1e-2}
        ])
    
    optimizer = torch.optim.RMSprop(model1.parameters()+model2.parameters()+model3.parameters()+copula.parameters(),lr=1e-2)
    for itr in range(100000):
        optimizer.zero_grad()
        loss = loss_triple(model1, model2, model3, train_dict, copula)
        loss.backward()
        train_loss_log.append(loss.detach().clone().cpu().numpy())
        optimizer.step()
        copula_grad.append(copula.theta.grad.clamp(-0.1, 0.1).detach().clone().cpu().numpy())
        for p in copula.parameters():
            if p < 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        with torch.no_grad():
                val_loss = loss_triple(model1, model2, model3, val_dict, copula)
                val_loss_log.append(val_loss.detach().clone().cpu().numpy())
            
        if itr % 50 == 0:
            plt.plot(train_loss_log, label="train")
            plt.plot(val_loss_log, label="validation")
            plt.savefig("loss.png")
            plt.cla()
            print(itr, copula.theta.detach().cpu().numpy(), loss.detach().clone().numpy(), best_loss.detach().clone().numpy(),\
                  surv_diff(dgp1, model1, val_dict['X'], 200).detach().cpu().numpy(), surv_diff(dgp2, model2, val_dict['X'], 200).detach().cpu().numpy(), surv_diff(dgp3, model3, val_dict['X'], 200).detach().cpu().numpy(), \
                    surv_diff(dgp1, model1, test_dict['X'], 200).detach().cpu().numpy(), surv_diff(dgp2, model2, test_dict['X'], 200).detach().cpu().numpy(), surv_diff(dgp3, model3, test_dict['X'], 200).detach().cpu().numpy())
            plt.plot(copula_grad)
            plt.savefig("copula_grad.png")
            plt.cla()
        
    
from DGP_models import Weibull_linear
from Copula import  Clayton, NestedClayton
from utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from new_model import Net, weibull_log_pdf, weibull_log_survival, triple_loss_, surv_diff


if __name__ =="__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    theta_dgp = 3
    eps = 1e-3
    train_dict, val_dict, test_dict = \
        generate_data(x_dict, dgp1, dgp2, dgp3, device = 'cpu', copula = copula_dgp, theta = theta_dgp)
    
    plt.hist(train_dict['E'])
    plt.savefig('E.png')
    plt.cla()

    copula = Clayton(theta_dgp, 1e-3, 'cpu')
    #copula = NestedClayton(2, 2, 1e-3, 1e-3, 'cpu')
    
    dgp_loss_train = loss_triple(dgp1, dgp2, dgp3, train_dict, copula)
    dgp_loss_val = loss_triple(dgp1, dgp2, dgp3, val_dict, copula)
    dgp_loss_test = loss_triple(dgp1, dgp2, dgp3, test_dict, copula)
    print(dgp_loss_train, dgp_loss_val, dgp_loss_test)
    #assert 0
    param_net = Net(nf).to(DEVICE)
    #copula = Clayton(0.5, 1e-3, DEVICE)
    copula = NestedClayton(2, 2, 1e-3, 1e-3, DEVICE)
    copula.enable_grad()
    chcopula_grad = []
    pcopula_grad = []
    optimizer = torch.optim.Adam([
        {'params': param_net.parameters(), "lr": 1e-4},
        {"params":copula.parameters(), "lr":1e-3}
        ])
    for itr in range(100000):
        param_net.train()
        optimizer.zero_grad()
        loss = triple_loss_(param_net, train_dict['X'].to(DEVICE), train_dict['T'].to(DEVICE), train_dict['E'].to(DEVICE), copula, DEVICE)
        loss.backward()
        #copula.CH_clayton.theta.grad = copula.CH_clayton.theta.grad * 100
        #copula.P_clayton.theta.grad = copula.P_clayton.theta.grad * 100
        
        optimizer.step()
        chcopula_grad.append(copula.CH_clayton.theta.grad.clamp(-0.1, 0.11).detach().clone().cpu().numpy())
        pcopula_grad.append(copula.P_clayton.theta.grad.clamp(-0.1, 0.11).detach().clone().cpu().numpy())
        
        for p in copula.parameters():
            if p < 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        
        if itr %100==0:
            with torch.no_grad():
                param_net.eval()
                loss_val = triple_loss_(param_net, val_dict['X'].to(DEVICE), val_dict['T'].to(DEVICE), val_dict['E'].to(DEVICE), copula, DEVICE)
            #print("######################################################")
            print(loss, loss_val,copula.CH_clayton.theta, copula.P_clayton.theta)
            print(surv_diff(dgp1, dgp2, dgp3, param_net, val_dict['X'], 200, DEVICE))
            print("######################################################")
            
            plt.plot(chcopula_grad)
            plt.savefig('child_grad.png')
            plt.cla()
            plt.plot(pcopula_grad)
            plt.savefig('parent_grad.png')
            plt.cla()
    



        
    
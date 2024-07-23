from DGP_models import Weibull_linear
from Copula import Clayton, NestedClayton, Convex_clayton, convex_Nested
from utils import *
import numpy as np
import torch 
from new_model import Net, weibull_log_pdf, weibull_log_survival, triple_loss_, surv_diff
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)

    nf = 10
    n_train = 10000
    n_val = 5000
    n_test = 5000

    x_dict = synthetic_x(n_train, n_val, n_test, nf, 'cpu')

    dgp1 = Weibull_linear(nf, alpha=17, gamma=3, device='cpu')
    dgp2 = Weibull_linear(nf, alpha=16, gamma=3, device='cpu')
    dgp3 = Weibull_linear(nf, alpha=17, gamma=4, device='cpu')


    dgp1.coeff = torch.rand((nf,), device='cpu')
    dgp2.coeff = torch.rand((nf,), device='cpu')
    dgp3.coeff = torch.rand((nf,), device='cpu')

    copula_dgp = 'clayton'
    theta_dgp = 1.3
    eps = 1e-3

    train_dict, val_dict, test_dict = \
        generate_data(x_dict, dgp1, dgp2, dgp3, device='cpu', copula=copula_dgp, theta=theta_dgp)
    
    print(train_dict['E'].unique(return_counts=True))

    copula = Clayton(theta_dgp, eps, device='cpu')
    dgp_loss_train = loss_triple(dgp1, dgp2, dgp3, train_dict, copula)
    dgp_loss_val = loss_triple(dgp1, dgp2, dgp3, val_dict, copula)
    dgp_loss_test = loss_triple(dgp1, dgp2, dgp3, test_dict, copula)

    print(dgp_loss_train, dgp_loss_val, dgp_loss_test)

    train_dataset = TensorDataset(train_dict['X'], train_dict['T'], train_dict['E'])
    train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    param_net = Net(nf).to(DEVICE)
    copula = Clayton(1, 1e-4, DEVICE)
    copula = NestedClayton(2, 2, 1e-3, 1e-3, DEVICE)
    copula = Convex_clayton(2,2,1e-3,1e-3, DEVICE)
    copula = convex_Nested(2,2,1e-3, 1e-3, DEVICE)
    copula.enable_grad()
    optimizer = torch.optim.Adam([
        {'params': param_net.parameters(), 'lr': 1e-3},
        {'params': copula.parameters(), 'lr': 1e-2}
    ])
    min_val_loss = 1000
    patience = 500
    for itr in range(50000):
        param_net.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(DEVICE)
            T = batch[1].to(DEVICE)
            E = batch[2].to(DEVICE)
            
            loss = triple_loss_(param_net, X, T, E, copula, DEVICE)
            loss.backward()
            #copula.theta.grad = copula.theta.grad * 100
            #copula.theta.grad = copula.theta.grad.clip(-0.1,0.1)
            optimizer.step()
            for p in copula.parameters()[:-2]:
                #print(p.shape)
                if p < 0.01:
                    with torch.no_grad():
                        p[:] = torch.clamp(p, 0.01, 100)
            
        with torch.no_grad():
            param_net.eval()
            val_loss = triple_loss_(param_net, val_dict['X'].to(DEVICE), val_dict['T'].to(DEVICE), val_dict['E'].to(DEVICE), copula, DEVICE).detach().clone().cpu().numpy()
            if val_loss  < min_val_loss:
                min_val_loss = val_loss
                #best_theta = copula.theta.detach().clone().cpu().numpy()
                torch.save(param_net.state_dict(), 'best.pt')
                patience = 500
            else:
                patience = patience - 1
        if patience == 0:
            break
        if (itr % 100 == 0):    
            print(min_val_loss)
    param_net.load_state_dict(torch.load('best.pt'))
    param_net.eval()
    print(surv_diff(dgp1, dgp2, dgp3, param_net, test_dict['X'], 200, DEVICE))
    print(surv_diff(dgp1, dgp2, dgp3, param_net, val_dict['X'], 200, DEVICE))
              
            


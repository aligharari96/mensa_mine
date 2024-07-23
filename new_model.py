import torch 
from utils import safe_log


class Net(torch.nn.Module):
    def __init__(self, nf, layers=[64,64]):
        super().__init__()
        
        d_in = nf
        self.layers_ = []
        for l in layers:
            d_out = l
            self.layers_.append(torch.nn.Linear(d_in, d_out))
            self.layers_.append(torch.nn.Dropout(0.2))
            self.layers_.append(torch.nn.ReLU())
            d_in = d_out
        self.out = torch.nn.Linear(d_in + nf, 6)
        self.layers_.append(torch.nn.Dropout(0.2))
            
        
        self.network = torch.nn.Sequential(*self.layers_)
    
    def forward(self, x):

        tmp = self.network(x)
        tmp = torch.cat([tmp, x], dim=1)
        params = self.out(tmp)
        k1 = torch.exp(params[:,0])
        k2 = torch.exp(params[:,1])
        k3 = torch.exp(params[:,2])
        
        lam1 = torch.exp(params[:,3])
        lam2 = torch.exp(params[:,4])
        lam3 = torch.exp(params[:,5])
        return k1, k2, k3, lam1, lam2, lam3
    
def weibull_log_pdf(t, k, lam):
    return safe_log(k) - safe_log(lam) + (k - 1) * safe_log(t/lam) - (t/lam)**k

def weibull_log_cdf(t, k, lam):
    return safe_log(1 - torch.exp(- (t / lam) ** k))

def weibull_log_survival(t, k, lam):
    return - (t / lam) ** k

def triple_loss_(param_net, X, T, E, copula, device):
    
    k1, k2, k3, lam1, lam2, lam3 = param_net(X)
    log_pdf1 = weibull_log_pdf(T, k1, lam1)
    log_pdf2 = weibull_log_pdf(T, k2, lam2)
    log_pdf3 = weibull_log_pdf(T, k3, lam3)
    log_surv1 = weibull_log_survival(T, k1, lam1)
    log_surv2 = weibull_log_survival(T, k2, lam2)
    log_surv3 = weibull_log_survival(T, k3, lam3)
    if copula is None:
        p1 = log_pdf1 + log_surv2 + log_surv3
        p2 = log_surv1 + log_pdf2 + log_surv3
        p3 = log_surv1 + log_surv2 + log_pdf3
    else:
        S = torch.cat([torch.exp(log_surv1).reshape(-1,1), torch.exp(log_surv2).reshape(-1,1), torch.exp(log_surv3).reshape(-1,1)], dim=1)#todo: clamp removed!!!!!!!
        p1 = log_pdf1 + safe_log(copula.conditional_cdf("u", S))
        p2 = log_pdf2 + safe_log(copula.conditional_cdf("v", S))
        p3 = log_pdf3 + safe_log(copula.conditional_cdf("w", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (E == 0) * 1.0
    e2 = (E == 1) * 1.0
    e3 = (E == 2) * 1.0
    loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
    loss = -loss/E.shape[0]
    return loss


def survival(model, k, lam, x, steps=200):
    u = torch.ones((x.shape[0],)) * 0.001
    time_steps = torch.linspace(1e-4, 1, steps=steps).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    t_max = t_max_model.reshape(-1,1).repeat(1, steps)
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps))
    surv2 = torch.zeros((x.shape[0], steps))
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i], x)
        surv2[:,i] = torch.exp(weibull_log_survival(time_steps[:,i], k, lam))
    return surv1, surv2, time_steps, t_max_model


def surv_diff(model1, model2, model3, param_net, x, steps, device):
    k1, k2, k3, lam1, lam2, lam3 = param_net(x.to(device))

    k1 = k1.to('cpu')
    lam1 = lam1.to('cpu')
    surv1, surv2, time_steps, t_m = survival(model1, k1, lam1, x, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
    l1_1 =  torch.mean(integ/t_m)

    k2 = k2.to('cpu')
    lam2 = lam2.to('cpu')
    surv1, surv2, time_steps, t_m = survival(model2, k2, lam2, x, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
    l1_2 =  torch.mean(integ/t_m)

    k3 = k3.to('cpu')
    lam3 = lam3.to('cpu')
    surv1, surv2, time_steps, t_m = survival(model3, k3, lam3, x, steps)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)),dim=1)
    l1_3 =  torch.mean(integ/t_m)

    return l1_1, l1_2, l1_3



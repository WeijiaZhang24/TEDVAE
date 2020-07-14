# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:40:16 2020

@author: Weijia
"""
import torch
from torchvision import transforms
import numpy as np
import pyro.distributions as dist

def generate_data(args, alpha= 0.25, beta = 1, gamma = 1):
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    
    zc = dist.Bernoulli(0.5).sample([args.num_data])
    zt = dist.Bernoulli(0.5).sample([args.num_data])
    zy = dist.Bernoulli(0.5).sample([args.num_data])
    
    # zc = dist.Normal(0,1).sample([args.num_data])
    # zt = dist.Normal(0,1).sample([args.num_data])
    # zy = dist.Normal(0,1).sample([args.num_data])

    xc = dist.Normal(zc, 5*zc + 3*(1-zc)).sample([args.synthetic_dim]).t()
    xt = dist.Normal(zt, 2*zt + 0.5*(1-zt)).sample([args.synthetic_dim]).t()
    xy = dist.Normal(zy, 10*zy + 6*(1-zy)).sample([args.synthetic_dim]).t()
    
    x =torch.cat([xc,xt,xy], -1)
    
    t = torch.mul(dist.Bernoulli( alpha * zt + (1-alpha) * (1 - zt)).sample(), dist.Bernoulli( alpha * zt + (1-alpha) * (1 - zt)).sample())    
    
    y = dist.Normal( beta*(zc + gamma* (2*t-2)), 1).sample([1]).t().squeeze(-1) + dist.Normal( beta*zy , 1).sample([1]).t().squeeze(-1)
    

    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.], [1.]])
    
    y_t0, y_t1 = dist.Normal( beta*(zc + gamma*(2*t0_t1-2)), 1).mean + dist.Normal( beta*zy , 1).mean
    
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite


def generate_data_z(args, dim_t, dim_c, dim_y, alpha= 0.25, beta = 1, gamma = 1):
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    prob_t = torch.tensor(0.5)
    probs_t = prob_t.new_full((dim_t,), 0.5)
    prob_c = torch.tensor(0.5)
    probs_c = prob_c.new_full((dim_c,), 0.5)    
    prob_y = torch.tensor(0.5)
    probs_y = prob_y.new_full((dim_y,), 0.5)
    
    zc = dist.Bernoulli(probs_c).sample([args.num_data])
    zt = dist.Bernoulli(probs_t).sample([args.num_data])
    zy = dist.Bernoulli(probs_y).sample([args.num_data])
    
    xc = torch.tensor([])
    for row in zc.split(1,dim=1):
        temp = dist.Normal(row.squeeze(),5*row.squeeze() + 3*(1-row.squeeze())).sample([5*dim_c]).t()
        xc = torch.cat([xc,temp], 1)

    xt = torch.tensor([])
    for row in zt.split(1,dim=1):
        temp = dist.Normal(row.squeeze(),5*row.squeeze() + 3*(1-row.squeeze())).sample([5*dim_t]).t()
        xt = torch.cat([xt,temp], 1)
        

    xy = torch.tensor([])
    for row in zy.split(1,dim=1):
        temp = dist.Normal(row.squeeze(),5*row.squeeze() + 3*(1-row.squeeze())).sample([5*dim_y]).t()
        xy = torch.cat([xy,temp], 1)

    
    x =torch.cat([xc,xt,xy], -1)
    
    t = torch.mul(dist.Bernoulli( alpha * zt + (1-alpha) * (1 - zt)).sample(), dist.Bernoulli( alpha * zt + (1-alpha) * (1 - zt)).sample()).squeeze()


    y = dist.Normal( beta*(zc.squeeze() + gamma* (2*t-2)), 1).sample([1]).t().squeeze(-1) \
        + dist.Normal( beta*zy.squeeze() , 1).sample([1]).t().squeeze(-1)
    
    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.], [1.]])
    
    y_t0, y_t1 = dist.Normal( beta*(zc + gamma*(2*t0_t1-2)), 1).mean + dist.Normal( beta*zy , 1).mean
    
    # y_t0 = (y_t0- beta*(zc + (2*t-2)) - beta*zy )/1.414
    # y_t1 = ((y_t1- beta*(zc + (2*t-2)) - beta*zy )/1.414)
    # y_t0, y_t1 = dist.Bernoulli(logits= ( beta* (zc+ 2 * (2 * t0_t1 - 2)))).mean
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite



def generate_data_cevae(args, alpha=0.75,beta=3,gamma=1):
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    
    z = dist.Bernoulli(0.5).sample([args.num_data])

    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([args.feature_dim]).t()

    t = dist.Bernoulli(alpha * z + (1 - alpha) * (1 - z)).sample()
    y = dist.Bernoulli(logits= beta * (z + gamma * (2 * t - 2))).sample()

    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.], [1.]])
    y_t0, y_t1 = dist.Bernoulli(logits=beta * (z + gamma * (2 * t0_t1 - 2))).mean
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite


def ACIC2016_train(reptition = 1, cuda = True):
    path_data = "./acic2016_continuous/"
    reptition = reptition
    
    data = np.loadtxt(path_data + str(reptition) + '_train.csv', delimiter=',',skiprows=1)
    t, y = data[:, 23][:, np.newaxis], data[:, 24][:, np.newaxis]
    mu_0, mu_1, x = data[:, 27][:, np.newaxis], data[:, 28][:, np.newaxis], data[:, :23]
    true_ite = mu_1 - mu_0
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
        
    return x, t, y, true_ite

def ACIC2016_test(reptition = 1, cuda = True):
    path_data = "./acic2016_continuous/"
    reptition = reptition
    
    data = np.loadtxt(path_data + str(reptition) + '_test.csv', delimiter=',',skiprows=1)
    t, y = data[:, 23][:, np.newaxis], data[:, 24][:, np.newaxis]
    mu_0, mu_1, x = data[:, 27][:, np.newaxis], data[:, 28][:, np.newaxis], data[:, :23]
    true_ite = mu_1 - mu_0
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
    return x, t, y, true_ite


def IHDP(path = "./IHDP", reps = 1, cuda = True):
    path_data = path
    replications = reps
    # which features are binary
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
    contfeats = [i for i in range(25) if i not in binfeats]

    data = np.loadtxt(path_data + '/ihdp_npci_train_' + str(replications) + '.csv', delimiter=',',skiprows=1)
    t, y = data[:, 0], data[:, 1][:, np.newaxis]
    mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    true_ite = mu_1 - mu_0
    x[:, 13] -= 1
    # perm = binfeats + contfeats
    # x = x[:, perm]
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
    train = (x, t, y), true_ite
            
    data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(replications) + '.csv', delimiter=',',skiprows=1)
    t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]
    x_test[:, 13] -= 1
    # x_test = x_test[:, perm]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).squeeze()
    t_test = torch.from_numpy(t_test).squeeze()
    if cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        t_test = t_test.cuda()
    true_ite_test = mu_1_test - mu_0_test
    test = (x_test, t_test, y_test), true_ite_test
    return train, test, contfeats, binfeats


def Jobs(path = "./Jobs", reps = 1, cuda = True):
    path_data = path
    replications = reps
    # which features are binary
    binfeats = [0,1,2,3]
        # which features are continuous
    contfeats = [4, 5,6,7]

    data = np.loadtxt(path_data + '/Jobs_train_' + str(replications) + '.csv', delimiter=',',skiprows=1)
    # t, y = data[:, 0], data[:, 10][:, np.newaxis] # index 10: continuous outcome re78
    t, y = data[:, 0], data[:, 1][:, np.newaxis] # index 1: binary outcome

    x = data[:, 2:10]
    # perm = binfeats + contfeats
    # x = x[:, perm]
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
    train = (x, t, y)
            
    data_test = np.loadtxt(path_data + '/Jobs_test_' + str(replications) + '.csv', delimiter=',',skiprows=1)
    t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    x_test =  data_test[:, 2:10]
    # x_test = x_test[:, perm]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).squeeze()
    t_test = torch.from_numpy(t_test).squeeze()
    if cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        t_test = t_test.cuda()
    test = (x_test, t_test, y_test)
    return train, test, contfeats, binfeats
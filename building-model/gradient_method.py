import numpy as np
import cvxpy as cp
from tqdm import tqdm
import random
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import linalg 
from itertools import accumulate
import pandas as pd
from utils import solve, model
import argparse

def train(layer, true, iters=1000, choice=1, random_seed=1, show=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pn_t = torch.tensor([0.05]).double().requires_grad_(True)
    a1_t = torch.tensor([0.5]).double().requires_grad_(True)
    a3_t = torch.tensor([0.5]).double().requires_grad_(True)
    max_theta_t = torch.tensor([18.5]).double().requires_grad_(True)   
    min_theta_t = torch.tensor([18]).double().requires_grad_(True)
    max_power_t = torch.tensor([1.0]).double().requires_grad_(True)
    variables = [pn_t,a1_t,a3_t,max_theta_t,min_theta_t,max_power_t]
    
    results = []
    record_variables = []
    optimizer = torch.optim.Adam(variables, lr=0.15)
    for i in range(iters):

        pred = layer(*variables)
        if choice==1:
            loss = nn.MSELoss()(true[0], pred[0]) + nn.MSELoss()(true[1], pred[1])
        else:
            loss = nn.MSELoss()(true[0], pred[0]) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pn_t.data = torch.clamp(pn_t.data, min=0.01, max=0.1) 
            a1_t.data = torch.clamp(a1_t.data, min=0.01, max=1) 
            a3_t.data = torch.clamp(a3_t.data, min=0.01, max=1) 
            max_power_t.data = torch.clamp(max_power_t.data, min=0.1, max=10) 
        
        results.append(loss.item())
        if i % 100==0: print("(iter %d) loss: %g " % (i, results[-1]))
        if i == 50:
            optimizer.param_groups[0]["lr"] = 0.1
        if i == 200:
            optimizer.param_groups[0]["lr"] = 0.05
        if i == 800:
            optimizer.param_groups[0]["lr"] = 0.01
        if show:
            im = plt.plot(results,color='gray')
            anno = plt.annotate(f'step:{i}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        record_variables.append([v.detach().numpy().copy() for v in variables])
    
    return [v.detach().numpy().copy() for v in variables], record_variables

def experiment(layer,seed1,theta_0, price, amb, choice, seed2, show, T=24*5):
    
    np.random.seed(seed1)
    price = price_data[:T]
    amb = amb_data[:T]
    
    C_th = 10 * np.random.uniform(0.9,1.1)
    R_th = 2 * np.random.uniform(0.9,1.1)
    P_n = 5 * np.random.uniform(0.9,1.1)
    eta = 2.5 * np.random.uniform(0.9,1.1)
    theta_r = 20 * np.random.uniform(0.9,1.1)
    Delta = np.random.uniform(0.9,1.1)
    
    pn_value = 0.02 # you can change it as you like
    a1_value = round(1 - 1/(R_th*C_th),4)
    a2_value = eta*R_th
    a3_value = round((1-a1_value)*a2_value,6)
    max_theta = round(theta_r + Delta,3)
    min_theta = round(theta_r - Delta,3)
    max_power = round(P_n,3)
    params = {'pn':pn_value, 'a1':a1_value, 'a2':a2_value, 'a3': a3_value,
          'max_theta':max_theta, 'min_theta':min_theta, 'max_power':max_power}
    print(params)  
    
    true = solve(price, amb, T, pn_value, a1_value, a3_value, max_theta, min_theta, max_power, theta_0, tensor=True)

    variables, record = train(layer, true, 600, choice, seed2, show)
    
    pn_ = ((variables[0][0] - pn_value)**2)**0.5
    a1_ = ((variables[1][0] - a1_value)**2)**0.5
    a3_ = ((variables[2][0] - a3_value)**2)**0.5
    max_theta_ = ((variables[3][0] - max_theta)**2)**0.5
    min_theta_ = ((variables[4][0] - min_theta)**2)**0.5
    max_power_ = ((variables[5][0] - max_power)**2)**0.5
    print(pn_,a1_,a3_,max_theta_,min_theta_,max_power_)
    
    return [v[0] for v in variables], [pn_value,a1_value,a3_value,max_theta,min_theta,max_power], [pn_,a1_,a3_,max_theta_,min_theta_,max_power_]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="the number of experiments")
    parser.add_argument("--save", type=bool, default=False, help="whether to save the result")
    parser.add_argument("--show", type=bool, default=False, help="whether to show the real-time training loss")
    parser.add_argument("--T", type=int, default=120, help="the length of the training data")
    parser.add_argument("--seed", type=int, default=1, help="the training random seed")
    parser.add_argument("--choice", type=int, default=1, help="1 for OptNet1 and 2 or OptNet2, indicated in the paper")
    
    opts = parser.parse_args()
    
    amb_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='theta_amb')['theta_amb'])
    price_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='price')['price'])
    
    #theta_0 = 21.64671372 # according to one history sample, you can change it as you like
    theta_0 = 35.00
    layer1 = model(price_data, amb_data, theta_0, opts.T)

    record = []
    record_variable = []
    record_true = []
    for i in range(opts.num):
        try:
            r1, r2, r3 = experiment(layer1, i, theta_0, price_data, amb_data, opts.choice, opts.seed, opts.show, opts.T)
        except Exception as e:
            continue
        record_variable.append(r1.copy())
        record_true.append(r2.copy())
        record.append(r3.copy())
    estimated_p = pd.DataFrame(data=record_variable, columns=['pn','a1','a3','max_t','min_t','max_p'])
    true_p = pd.DataFrame(data=record_true, columns=['pn','a1','a3','max_t','min_t','max_p'])
    mse = pd.DataFrame(data=record, columns=['pn','a1','a3','max_t','min_t','max_p'])

    if opts.save:
        present_time = time.strftime("%m%d%H%M", time.localtime()) 
        file_name = f"result_data/opt{opts.choice}_seed{opts.seed}_{present_time}.csv"
        total_df=pd.concat([estimated_p, true_p, mse])
        total_df.to_csv(file_name)

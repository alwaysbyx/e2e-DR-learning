import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pylab
from plotly.subplots import make_subplots
from tqdm import tqdm
import random
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from numpy import linalg 
from itertools import accumulate
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import scipy.io as scio
import pandas as pd
from utils import solve, model


def experiment(layer,i):
    
    T = 24*5
    price = price_data[:T]
    amb = amb_data[:T]
    
    np.random.seed(i)
    C_th = 10 * np.random.uniform(0.9,1.1)
    R_th = 2 * np.random.uniform(0.9,1.1)
    P_n = 5 * np.random.uniform(0.9,1.1)
    eta = 2.5 * np.random.uniform(0.9,1.1)
    theta_r = 20 * np.random.uniform(0.9,1.1)
    Delta = np.random.uniform(0.9,1.1)
    
    theta_0 = 21.64671372
    pn_value = 0.02
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

    variables, record = train(layer, true, 500)
    
    pn_ = (variables[0][0] - pn_value)/pn_value
    a1_ = (variables[1][0] - a1_value)/a1_value
    a3_ = (variables[2][0] - a3_value)/a3_value
    max_theta_ = (variables[3][0] - max_theta)/max_theta
    min_theta_ = (variables[4][0] - min_theta)/min_theta
    max_power_ = (variables[5][0] - max_power)/max_power
    print(pn_,a1_,a3_,max_theta_,min_theta_,max_power_)
    
    return [pn_,a1_,a3_,max_theta_,min_theta_,max_power_], [v[0] for v in variables], [pn_value,a1_value,a3_value,max_theta,min_theta,max_power]
    
if __name__ == '__main__':
    amb_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='theta_amb')['theta_amb'])
    price_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='price')['price'])

    layer1 = model(price_data, amb_data)

    record = []
    record_variable = []
    record_true = []
    for i in range(10):
        r1, r2, r3 = experiment(layer1,i)
        record_variable.append(r2.copy())
        record.append(r1.copy())
        record_true.append(r3.copy())
    estimated_p = pd.DataFrame(data=record_variable, columns=['pn','a1','a3','max_t','min_t','max_p'])
    true_p = pd.DataFrame(data=record_true, columns=['pn','a1','a3','max_t','min_t','max_p'])
    mape = pd.DataFrame(data=record, columns=['pn','a1','a3','max_t','min_t','max_p'])
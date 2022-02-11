import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer

def solve(price, amb, T, pn, a1, a3, max_theta, min_theta, max_power, theta_0, tensor=False):
    p = cp.Variable(T)
    y = cp.Variable(T)
    theta = cp.Variable(T)
    obj = price@p + pn * cp.sum_squares(y)
    cons = [y >= 0, p >= 0]
    for i in range(T):
        if i == 0:
            cons.append(theta[i] == a1*theta_0 + (1-a1)*amb[i] - a3*p[i] )
        else:
            cons.append(theta[i] == a1*theta[i-1] + (1-a1)*amb[i] - a3*p[i] )
        cons.append(p[i] <= max_power)
        cons.append(theta[i] <= max_theta + y[i])
        cons.append(theta[i] >= min_theta - y[i])
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve()
    if tensor:
        return torch.tensor(p.value),torch.tensor(theta.value),torch.tensor(y.value)
    return p.value, theta.value, y.value


def model(price_data, amb_data):
    i = 0
    T = 24*5
    theta_0 = 21.64671372

    pn_p = cp.Parameter(1,nonneg=True)
    a1_p = cp.Parameter(1)
    a3_p = cp.Parameter(1)
    max_theta_p = cp.Parameter(1)
    min_theta_p = cp.Parameter(1)
    max_power_p = cp.Parameter(1)

    # problem formulation
    price = price_data[i*T:(i+1)*T]
    amb = amb_data[i*T:(i+1)*T]

    p_v = cp.Variable(T)
    y_v = cp.Variable(T)
    theta_v = cp.Variable(T)
    obj = price@p_v + pn_p * cp.sum_squares(y_v)
    cons = [y_v >= 0, p_v >= 0]
    for i in range(T):
        if i == 0:
            cons.append(theta_v[i] == a1_p*theta_0 + (1-a1_p)*amb[i] - a3_p*p_v[i])
        else:
            cons.append(theta_v[i] == a1_p*theta_v[i-1] + (1-a1_p)*amb[i] - a3_p*p_v[i])
        cons.append(p_v[i] <= max_power_p)
        cons.append(theta_v[i] <= max_theta_p + y_v[i])
        cons.append(theta_v[i] >= min_theta_p - y_v[i])
    problem = cp.Problem(cp.Minimize(obj), cons)
    assert problem.is_dpp()
    layer = CvxpyLayer(problem, [pn_p,a1_p,a3_p,max_theta_p,min_theta_p,max_power_p], [p_v,theta_v,y_v])
    return layer
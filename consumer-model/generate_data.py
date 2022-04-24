import numpy as np
from utils import solve
import cvxpy as cp
import time
import argparse
import scipy.io as scio
from scipy.io import savemat
import pandas as pd


def generate_dataset(T,N,K=3,noise=0,seed=0,real=False):
    np.random.seed(seed)
    u = 10 * np.random.rand(K,T)
    L = np.array([np.random.uniform(5,10,T) for _ in range(K)])
    M = np.array([10,20,16])
    data = []
    Price = []
    if real:
        price_hist = pd.read_csv("../energystorage-model/ESID_data/price.csv")
    for i in range(N):
        if real: 
            price = np.array(price_hist.RTP[i*T:(i+1)*T])
        else: price = 30*np.random.rand(T)
        x = cp.Variable((K,T))
        obj = K*cp.sum(price@x.T) - cp.sum(u@x.T)
        problem = cp.Problem(cp.Minimize(obj), [x <= L, x >= 0, cp.sum(x,axis=1) == M])
        problem.solve()
        if noise != 0:
            tmp = np.array([np.random.uniform(1-noise/10,1+noise/10)*xx for xx in x.value])
            data.append(np.sum(tmp,axis=0))
        else:
            data.append(np.sum(x.value,axis=0))
        Price.append(price)
    data = np.array(data)
    Price = np.array(Price)
    return {'u':u, 'L':L, 'M':M, 'data':data, 'price':Price}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3, help="the number of loads")
    parser.add_argument("--noise", type=int, default=0, help="the noise level, {0,1,2}")
    parser.add_argument("--N", type=int, default=50, help="the number of history samples")
    parser.add_argument("--real", type=bool, default=False, help="whether use the real-world price")
    parser.add_argument("--seed", type=int, default=0, help="the random seed")
    parser.add_argument("--T", type=int, default=12, help="time slots")
    
    opts = parser.parse_args()

    data = generate_dataset(opts.T, opts.N, opts.K, opts.noise, opts.seed, opts.real)
    file_name = f'dataset/K{opts.K}pi{opts.noise}N{opts.N}seed{opts.seed}real{opts.real}.mat'
    savemat(file_name,data)



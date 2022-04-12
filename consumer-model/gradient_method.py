import torch
import torch.nn as nn
from utils import Layer
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import scipy.io as scio
from scipy.io import savemat


def train(data, max_ite=500,  N=3, seed=0, show=False):
    L = []
    torch.manual_seed(seed)
    price_tensor = torch.from_numpy(data['price'])
    z_tensor = torch.from_numpy(data['data'])
    layer = Layer(N=N)
    opt = optim.Adam(layer.parameters(), lr=0.2)
    for ite in range(max_ite):
        if(ite == 100):
            opt.param_groups[0]["lr"] = 0.1
        if(ite == 150):
            opt.param_groups[0]["lr"] = 0.05
        dr_pred = layer(price_tensor)
        loss = nn.MSELoss()(z_tensor, dr_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
        L.append(loss.detach().numpy())
        
        layer.L.data = torch.clamp(layer.L.data, min=0.01, max=100) 
        layer.u.data = torch.clamp(layer.u.data, min=0.01, max=100) 
        
        if show:
            im = plt.plot(L,color='gray')
            anno = plt.annotate(f'step:{ite}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        if ite % 20 == 1:
            print(f'iteration{ite}: loss = {loss}')
            # print('u',layer.u.detach().numpy())
            # print('M',layer.M.detach().numpy())
            # print('L',layer.L.detach().numpy())
    plt.cla()
    return [layer.u.detach().numpy(), layer.M.detach().numpy(), layer.L.detach().numpy()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3, help="the number of loads")
    parser.add_argument("--save", type=bool, default=True, help="whether to save the result")
    parser.add_argument("--show", type=bool, default=False, help="whether to show the real-time training loss")
    parser.add_argument("--noise", type=int, default=0, help="the noise level, {0,1,2}")
    parser.add_argument("--seed", type=int, default=0, help="the training random seed")
    parser.add_argument("--max_ite", type=int, default=200, help="the max iteration")
    
    opts = parser.parse_args()
    DATA = scio.loadmat(f'dataset/K{opts.K}pi{opts.noise}N50.mat')
    u, M, L = train(DATA, opts.max_ite, opts.K, opts.seed, opts.show)
    
    if opts.save:
        present_time = time.strftime("%m%d%H%M", time.localtime()) 
        file_name = f"result_data/OPT_K{opts.K}_noise{opts.noise}_seed{opts.seed}_{present_time}.mat"
        savemat(file_name, {'u':u,'L':L,'M':M})
        
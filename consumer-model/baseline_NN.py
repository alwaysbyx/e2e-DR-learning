import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import argparse
from scipy.io import savemat
import time


class Net(nn.Module):

    def __init__(self, d0=12, d1 = 12):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(d0, d1*3)
        self.fc2 = nn.Linear(d1*3, d1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(iters,show=False):
    layer = Net()
    
    results = []
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.2)
    for i in range(iters):
        torch.manual_seed(1)
        np.random.seed(1)
               
        pred = layer(price)        
        loss = nn.MSELoss()(dr ,pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        results.append(loss.item())
        if i % 100==0: print("(iter %d) loss: %g " % (i, results[-1]))
        if i == 200:
            optimizer.param_groups[0]["lr"] = 0.1
        if i== 400:
            optimizer.param_groups[0]["lr"] = 0.01
        if show:
            im = plt.plot(results,color='gray')
            anno = plt.annotate(f'step:{i}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        
    return layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3, help="the number of loads")
    parser.add_argument("--save", type=bool, default=True, help="whether to save the result")
    parser.add_argument("--show", type=bool, default=False, help="whether to show the real-time training loss")
    parser.add_argument("--noise", type=int, default=0, help="the noise level, {0,1,2}")
    parser.add_argument("--seed", type=int, default=0, help="the training random seed")
    parser.add_argument("--max_ite", type=int, default=1000, help="the max iteration")
    
    opts = parser.parse_args()
    DATA = scio.loadmat(f'dataset/K{opts.K}pi{opts.noise}N50.mat')
    price = torch.tensor(DATA['price']).to(torch.float32)
    dr = torch.tensor(DATA['data']).to(torch.float32)
    model = train(opts.max_ite,opts.show)

    if opts.save:
        present_time = time.strftime("%m%d%H%M", time.localtime()) 
        file_name = f"result_data/NN_K{opts.K}_noise{opts.noise}_seed{opts.seed}_{present_time}.pth"
        torch.save(model.state_dict(), file_name)
        


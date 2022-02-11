import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.metrics import mean_squared_error


class Net(nn.Module):

    def __init__(self, d0=12, d1 = 12):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(d0, d1*3)
        self.fc2 = nn.Linear(d1*3, d1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(iters):
    layer = Net()
    
    results = []
    show = True
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
        if show:
            im = plt.plot(results,color='gray')
            anno = plt.annotate(f'step:{i}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        
    return layer


if __name__ == '__main__':
    dataset = scio.loadmat('dataset/K3pi2N50.mat')
    price = torch.tensor(dataset['price']).to(torch.float32)
    dr = torch.tensor(dataset['data']).to(torch.float32)
    model = train(500)

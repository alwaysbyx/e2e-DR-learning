import torch
import torch.nn as nn
from utils import Layer
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import scipy.io as scio

if __name__ == '__main__':

    DATA = scio.loadmat('dataset/K3pi2N50.mat')

    show = True
    L = []
    torch.manual_seed(0)
    price_tensor = torch.from_numpy(DATA['price'])
    z_tensor = torch.from_numpy(DATA['data'])
    layer = Layer(N=3)
    opt = optim.Adam(layer.parameters(), lr=0.2)
    for ite in range(500):
        if(ite == 100):
            opt.param_groups[0]["lr"] = 0.1
        if(ite == 400):
            opt.param_groups[0]["lr"] = 0.05
        dr_pred = layer(price_tensor)
        loss = nn.MSELoss()(z_tensor, dr_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
        L.append(loss)
        
        layer.L.data = torch.clamp(layer.L.data, min=0.01, max=100) 
        layer.u.data = torch.clamp(layer.u.data, min=0.01, max=100) 
        
        if show:
            im = plt.plot(L,color='gray')
            anno = plt.annotate(f'step:{ite}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        if ite % 100 == 1:
            print('loss = ',loss)
            print('u',layer.u.detach().numpy())
            print('M',layer.M.detach().numpy())
            print('L',layer.L.detach().numpy())
    plt.cla()
    e = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import solve


class Net(nn.Module):

    def __init__(self, d0=24*2, d1 = 24):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(d0, d1)
        self.fc2 = nn.Linear(d1, d1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train(iters):
    layer = Net()
    
    results = []
    show = True
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    for i in range(iters):
        torch.manual_seed(1)
        np.random.seed(1)
               
        pred = layer(input_tensor)        
        loss = nn.MSELoss()(output_tensor ,pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        results.append(loss.item())
        if i % 100==0: print("(iter %d) loss: %g " % (i, results[-1]))
        #if i % 100 == 1: print(variables)
        if i == 50:
            optimizer.param_groups[0]["lr"] = 0.05
        if show:
            im = plt.plot(results,color='gray')
            anno = plt.annotate(f'step:{i}\n loss={loss}', xy=(0.85, 0.9), xycoords='axes fraction',color='black')
            plt.axis("equal")
            plt.pause(0.001)
            anno.remove()
        
    return layer

if __name__ == '__main__':
    amb_data = np.array(pd.read_excel('input_data_pool.xlsx',sheet_name='theta_amb')['theta_amb'])
    price_data = np.array(pd.read_excel('input_data_pool.xlsx',sheet_name='price')['price'])

    T = 24*5
    price = price_data[:T]
    amb = amb_data[:T]
    variable = {'pn': 0.02, 'a1': 0.9521839137854551, 'a2': 5.598830495762315, 'a3': 0.2677141616859941, 'max_theta': 20.539969677000002, 'min_theta': 18.670106323, 'max_power': 5.38}
    pn_value = variable['pn']
    a1_value = variable['a1']
    a2_value = variable['a2']
    max_theta = variable['max_theta']
    min_theta = variable['min_theta']
    max_power = variable['max_power']
    theta_0 = 21.64671372
    true = solve(price, amb, T, pn_value, a1_value, a2_value, max_theta, min_theta, max_power,theta_0, tensor=True)

    input_price = torch.tensor(price_data[:24*5]).reshape(5,24)
    input_amb = torch.tensor(amb_data[:24*5]).reshape(5,24)
    input_tensor = torch.concat([input_price,input_amb],dim=1).to(torch.float32)
    output_tensor = true[0].reshape(5,24).to(torch.float32)

    model = train(500)

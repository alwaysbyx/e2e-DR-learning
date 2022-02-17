import numpy as np
import pandas as pd
from utils import data_generator

## data dimension
N_train = 110 # sum of training and valiadation set
dim = 24

## initialize parameters
c1_value = round(np.random.uniform(0, 20),2)
c2_value = round(np.random.uniform(0, 20),2)
duration = round(np.random.uniform(1, 4))
eta = round(np.random.uniform(0.8, 1),2)

paras = pd.DataFrame([[c1_value, c2_value, duration, eta]],columns=("c1", "c2", "P", "eta"))

print(
    "Generating data!",
    "P1=",
    0.5,
    "E1=",
    0.5 * duration,
    "c1 =",
    c1_value,
    "c2 =",
    c2_value,
    "eta =",
    eta,
)

## load price data
price_hist = pd.read_csv("./ESID_data/price.csv")

## generate dispatch data and save price, true parameters
df_price, df_d, df_p = data_generator(
    c1_value,
    c2_value,
    upperbound_p=0.5,
    lowerbound_p=0,
    upperbound_e=0.5*duration,
    lowerbound_e=0,
    initial_e=0.25*duration,
    efficiency=eta,
    price_hist=price_hist,
    N=N_train,
    T=dim,
)
np.savez("data", paras = paras, price = df_price, d=df_d, p=df_p)


import numpy as np
import pandas as pd
from utils import data_generator_val

N_train = 20
N_valid = 10
T=24
ite_list = [0,99,199,299,399,499]
df = pd.DataFrame()
for i in range(10):
    df_dp = np.load("./Results/data%d/data.npz"%(i+1))
    df_para = pd.read_csv("./Results/data%d/learning.csv"%(i+1))
    df_price = df_dp["price"]
    df_p = df_dp["p"]
    df_d = df_dp["d"]


    price_valid = df_price[100:]
    d_valid = df_d[100:]
    p_valid = df_p[100:]
    y_valid = p_valid - d_valid

    for j in range(6):
        ite = ite_list[j]
        d_pred, p_pred = data_generator_val(
            df_para.loc[ite]['c1'],
            df_para.loc[ite]['c2'],
            upperbound_p=0.5,
            lowerbound_p=0,
            upperbound_e=df_para.loc[ite]['E1'],
            lowerbound_e= df_para.loc[ite]['E2'],
            efficiency=df_para.loc[ite]['eta'],
            price_hist=price_valid,
            N=N_valid,
            T=T,
        )
        y_pred = p_pred - d_pred
        mse = np.square(y_pred - y_valid).mean()
        df.loc[i,j] = mse
df.columns=("1", "100", "200", "300", "400", "500")
df.to_csv("OptNet_val_loss.csv")



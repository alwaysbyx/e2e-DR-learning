import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import scipy.io as scio
from scipy.io import savemat
from utils import solve
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def MLP():
    model = Sequential()
    model.add(Dense(64, input_dim=48))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(48))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


def RNNmodel():
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(48, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=48))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

def generate_dataset(N,  T=24):
    np.random.seed(0)
    amb_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='theta_amb')['theta_amb'])
    price_data = np.array(pd.read_excel('dataset/input_data_pool.xlsx',sheet_name='price')['price'])

    C_th = 10 * np.random.uniform(0.9,1.1)
    R_th = 2 * np.random.uniform(0.9,1.1)
    P_n = 5 * np.random.uniform(0.9,1.1)
    eta = 2.5 * np.random.uniform(0.9,1.1)
    theta_r = 20 * np.random.uniform(0.9,1.1)
    Delta = np.random.uniform(0.9,1.1)
    
    pn_value = 0.02 # you can change it as you like
    a1_value = round(1 - 1/(R_th*C_th),4)
    a2_value = eta*R_th
    a3_value = round((1-a1_value)*a2_value,6)
    max_theta = round(theta_r + Delta,3)
    min_theta = round(theta_r - Delta,3)
    max_power = round(P_n,3)
    params = {'pn':pn_value, 'a1':a1_value,  'a3': a3_value,
          'max_theta':max_theta, 'min_theta':min_theta, 'max_power':max_power}

    theta_0 = 35.0
    
    p,theta,y = solve(price_data[:N*T], amb_data[:N*T], N*T, pn_value, a1_value, a3_value, max_theta, min_theta, max_power, theta_0)

    price = np.array(price_data[:N*T]).reshape(N,T)
    amb = np.array(price_data[:N*T]).reshape(N,T)
    x = np.concatenate([price, amb],axis=1)

    p = np.array(p).reshape(N,T)
    theta = np.array(theta).reshape(N,T)
    dr = np.concatenate([p, theta],axis=1)
    return x, dr


if __name__ == '__main__':
    X,dr = generate_dataset(65)
    N = 50
    train_price = X[1:1+N]
    train_dr = dr[1:1+N]
    mlp = MLP()
    mlp.fit(train_price,train_dr, epochs=500,batch_size=2)
    end = time.time()

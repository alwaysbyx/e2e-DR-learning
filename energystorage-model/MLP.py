import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.random import seed
seed(0)
tf.random.set_seed(0)

df = pd.DataFrame(columns=("1", "100", "200", "300", "400", "500"))

for i in range(10):
    df_dp = np.load("./Results/data%d/data.npz"%(i+1))
    df_price = df_dp["price"]
    p = df_dp["p"]
    d = df_dp["d"]
    y = p-d
    T = 24
    N_train = 100
    X_train = df_price[0:N_train]
    y_train = y[0:N_train]
    X_valid = df_price[N_train:]
    y_valid = y[N_train:]
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(T,)),
        tf.keras.layers.Dense(T, activation='relu'),
        tf.keras.layers.Dense(T)
    ])
    print(model.summary())

    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError())
    history = model.fit(X_train, y_train, epochs=500, validation_data=(X_valid, y_valid),batch_size=len(X_train))
    df.loc[i] = [history.history['val_loss'][0], history.history['val_loss'][99], history.history['val_loss'][199], history.history['val_loss'][299], history.history['val_loss'][399], history.history['val_loss'][499]]


df.to_csv("MLP_val_loss.csv")


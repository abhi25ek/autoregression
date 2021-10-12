import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
plt.plot(series)

T = 10
X = []
Y = []
 
for t in range(len(series)-T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
X.shape, Y.shape

i = Input(shape = (T,))
x = Dense(1)(i)
model = Model(i,x)
model.compile(
    loss = 'mse',
    optimizer = Adam(learning_rate = 0.1)
)

r = model.fit(
    X[:-N//2], Y[:-N//2], epochs = 200, validation_data = (X[-N//2:], Y[-N//2:])
)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')

validation_target = Y[-N//2:]
validation_predict = []

last_x = X[-N//2]
while len(validation_predict) < len(validation_target):
  P = model.predict(last_x.reshape(1, -1))[0,0]

  validation_predict.append(P)
 
  last_x = np.roll(last_x, -1)
  last_x[-1] = P

plt.plot(validation_predict, label = 'prediction')
plt.plot(validation_target, label = 'target')
plt.legend()
plt.show()

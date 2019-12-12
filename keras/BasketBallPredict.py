import numpy as np
from keras import Model
from keras.layers import Dense, Input, MaxPooling2D, Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt

data = np.loadtxt('../../data/successful_shots.csv', delimiter=',')


len = data.shape[0]
print("length = ", len)
train_len = int(len * 0.9)

X_train, X_test = data[0:train_len, 0], data[train_len:, 0]
Y_train, Y_test = data[0:train_len, 1], data[train_len:, 1]

model = load_model('../../model/basket_ball.h5')

predict_Test = model.predict(X_test)

print(X_test[0:5])
print(Y_test[0:5])
print(predict_Test[0:5])

# data = data[0:100]

# data.sort(axis=0)
X = data[:, 0]
Y = data[:, 1]
Y_P = model.predict(X)

plt.scatter(X, Y, s=5)
plt.scatter(X, Y_P, s=5)
plt.xlabel('distance')
plt.ylabel('force')
plt.show()
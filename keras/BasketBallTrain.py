import numpy as np
from keras import Model
from keras.layers import Dense, Input, MaxPooling2D, Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential

data = np.loadtxt('../data/successful_shots.csv', delimiter=',')

X = data[:, 0]
Y = data[:, 1]

len = data.shape[0]
train_len = int(len * 0.9)

X_train, X_test = data[0:train_len, 0], data[train_len:, 0]
Y_train, Y_test = data[0:train_len, 1], data[train_len:, 1]

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("Y_train", Y_train.shape)
print("Y_test", Y_test.shape)

EPOCHS = 100

def build_model():
    model = Sequential()
    model.add(BatchNormalization(epsilon=1e-6, weights=None, input_shape=((1,))))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

model = build_model()
history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, verbose=1)

model.save('../model/basket_ball.h5')

predict_Test = model.predict(X_test)

print(X_test[0:5])
print(Y_test[0:5])
print(predict_Test[0:5])
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

nFeatures = X_train.shape[1]

# keras models
model = Sequential()
model.add(Dense(1, input_shape=(nFeatures,)))
model.add(Activation('linear'))

# configure training process
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

# training
trainFeatures = X_train
trainLabels = Y_train
model.fit(trainFeatures, trainLabels, batch_size=4, epochs=100)

model.summary()

model.evaluate(X_test, Y_test, verbose=True)
Y_pred = model.predict(X_test)

print(Y_test[:5])
print(Y_pred[:5, 0])

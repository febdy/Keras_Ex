import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('트레이닝 데이터 수 : ', train_images.shape, train_labels.shape)
print('테스트 데이터 수 : ', test_images.shape, test_labels.shape)

classes = np.unique(train_labels)
nClasses = len(classes)
print('결과 수 : ', nClasses)
print('결과값들 : ', classes)

plt.figure(figsize=[10, 5])

# 첫번째 트레이닝 데이터
plt.subplot(121)
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title("Ground truth : {}".format(train_labels[0]))

# 첫번째 테스팅 데이터
plt.subplot(122)
plt.imshow(test_images[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))

# 데이터들을 1차원 배열로 만듦
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

# 원-핫 인코딩으로 바꾸기
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

print('원래 수 :', train_labels[0])
print('one-hot 인코딩 : ', train_labels_one_hot[0])

'''
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClasses, activation='softmax'))

# 손실 함수 종류, 트레이닝으로 추적할 측정 항목 정의
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 트레이닝
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                    validation_data=(test_data, test_labels_one_hot))

# 성능 확인
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("차이 = {}, 정확도 = {}".format(test_loss, test_acc))

plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
'''

# 위 방법은 Overfitting 됨.
# Dropout으로 해결
model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation='softmax'))

model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                            validation_data=(test_data, test_labels_one_hot))

plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['loss'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.figure(figsize=[8, 6])
plt.plot(history_reg.history['acc'], 'r', linewidth=3.0)
plt.plot(history_reg.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

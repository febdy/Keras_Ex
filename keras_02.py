import keras
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import cv2
import numpy as np


vgg_model = vgg16.VGG16(weights='imagenet')

filename = 'C:/Users/BIT-USER/Desktop/python_workplace/cat.jpg'
img = cv2.imread(filename)  # Numpy array - (height, width, channel)

# img = cv2.resize(img, None, , interpolation=cv2.INTER_CUBIC)

img_batch = np.expand_dims(img, axis=0)

img_batch = img_batch.astype('float32')
processed_img = vgg16.preprocess_input(img_batch.copy())  # == (input - 모든 이미지의 R, G, B 평균 array)
predictions = vgg_model.predict(processed_img)  # 예측 결과 얻음
label = decode_predictions(predictions)  # 사람이 볼 수 있게 결과를 바꿔줌 (class Id, class name, 확률)
print(label)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

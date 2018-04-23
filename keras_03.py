from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

train_dir = './train'
validation_dir = './validation'

nTrain = 600
nVal = 150

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

'''
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=nTrain, 3)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=shuffle
)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1 ) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break
train_featrues = np.reshape(train_features, (nTrain, 7 * 7 * 512))


from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activatino='relu', input_dim=7*7*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=optimizers,RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))
'''
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

epochs = 25
init_lr = 1e-3
bs = 32  # batch_size

data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data. append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == 'santa' else 0
    labels.append(label)
    
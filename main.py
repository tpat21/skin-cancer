import numpy as np
import os
import cv2
from keras_preprocessing.image import load_img
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras





def Dataset_loader(DIR):
    IMG = []
    for image in os.listdir(DIR):
        PATH = os.path.join(DIR, image)
        imgArray = np.array((cv2.imread(PATH, 0)))

        IMG.append(np.array(imgArray))

    return (IMG)


benign_test = np.array(Dataset_loader("/Users/tpat/PycharmProjects/skin-cancer/data/test/benign"))
malign_test = np.array(Dataset_loader("/Users/tpat/PycharmProjects/skin-cancer/data/test/malignant"))

malign_train = np.array(Dataset_loader("/Users/tpat/PycharmProjects/skin-cancer/data/train/malignant"))
benign_train = np.array(Dataset_loader("/Users/tpat/PycharmProjects/skin-cancer/data/train/benign"))





model = keras.Sequential([
    keras.layers.Flatten(input_shape=(244,244))
])
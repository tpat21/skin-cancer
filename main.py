import os
import numpy as np
import cv2
import image
import pandas as pd


def Dataset_loader(DIR):
    IMG = []
    for image in os.listdir(DIR):
        PATH = os.path.join(DIR, image)
        imgArray = ((cv2.imread(PATH,0)))
        IMG.append((imgArray/255))
    return (IMG)




benign_train = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/train/benign")))
malignant_train = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/train/malignant")))

benign_test = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/test/benign")))
malignant_test = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/test/malignant")))


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(244,244))
# ])


# Create Labels
benign_train_label = np.ones(len(benign_train))
malignant_train_label = np.zeros(len(malignant_train))

benign_test_label = np.ones(len(benign_test))
malignant_test_label = np.zeros(len(malignant_test))


# Merge data
X_train = np.concatenate((benign_train,malignant_train),axis=0)
y_train = np.concatenate((benign_train_label,malignant_test_label),axis=0)

X_test = np.concatenate((benign_test,malignant_test),axis=0)
y_test = np.concatenate((benign_test_label,malignant_test_label),axis=0)




# # Shuffle data
# s = np.arange(X_train.shape[0])
# np.random.shuffle(s)
# X_train = X_train[s]
# y_train = y_train[s]
#
# s = np.arange(X_test.shape[0])
# np.random.shuffle(s)
# X_test = X_test[s]
# y_test = y_test[s]


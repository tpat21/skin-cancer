import os
import numpy as np
import cv2
import image
import pandas as pd
import matplotlib.pyplot as plt


def Dataset_loader(DIR):
    IMG = []
    for image in os.listdir(DIR):
        PATH = os.path.join(DIR, image)
        imgArray = ((cv2.imread(PATH,1)))
        IMG.append((imgArray/255))
    return (IMG)




benign_train = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/train/benign")))
malignant_train = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/train/malignant")))

benign_test = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/test/benign")))
malignant_test = np.array(Dataset_loader(("/Users/tpat/PycharmProjects/skin-cancer/data/test/malignant")))


# print("benign_train: {}".format(len(benign_train)))
# print("malignant_train: {}".format(len(malignant_train)))
#
# print("benign_test: {}".format(len(benign_test)))
# print("malignant_test: {}".format(len(malignant_test)))



# Create Labels
benign_train_label = np.ones(len(benign_train))
malignant_train_label = np.zeros(len(malignant_train))

benign_test_label = np.ones(len(benign_test))
malignant_test_label = np.zeros(len(malignant_test))


# print("benign_train_label: {}".format(np.shape(benign_train_label)))
# print("malignant_train_label: {}".format(np.shape(malignant_train_label)))
#
# print("benign_test_label: {}".format(np.shape(benign_test_label)))
# print("malignant_test_label: {}".format(np.shape(malignant_test_label)))


# Merge data
X_train = np.concatenate((benign_train,malignant_train),axis=0)
y_train = np.concatenate((benign_train_label,malignant_train_label),axis=0)

X_test = np.concatenate((benign_test,malignant_test),axis=0)
y_test = np.concatenate((benign_test_label,malignant_test_label),axis=0)

# print("X_train: {}".format(np.shape(X_train)))
# print("y_train: {}".format(np.shape(y_train)))
#
# print("X_test: {}".format(np.shape(X_test)))
# print("y_test: {}".format(np.shape(y_test)))

# # Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]




s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]




# # Display first 15 images of moles, and how they are classified
lenght = 60
height = 40
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if y_train[i] == 0:
        ax.title.set_text('Malignant')
    else:
        ax.title.set_text('Benign')
    plt.imshow(X_train[i],cmap=plt.cm.binary,interpolation='nearest')
plt.show()




# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(244,244))
# ])
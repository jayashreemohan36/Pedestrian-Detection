import cv2
import os
import numpy as np
import cv2
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dir = '/content/drive/MyDrive/Internship/INRIA/Training_data_08.03/INRIA_data'
categories = ['pos', 'neg']

HOG_values = []
labels = []

for category in categories:
    # combines both the paths as one path
    path = os.path.join(dir, category)
    all_images = os.listdir(path)
    for img in all_images:
        # giving the path of the first image from the combined path so that image is read from that path.
        imgpath = os.path.join(path, img)
        pet_image = cv2.imread(imgpath, 0)
        # plt.imshow(pet_image)
        color_image = cv2.cvtColor(pet_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        new_image = cv2.resize(gray_image, (64, 128))
        # plt.imshow(new_image)

        hog = feature.hog(new_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L1',
                          transform_sqrt=True)

        # (hog, hog_image) = feature.hog(new_image, orientations=9,pixels_per_cell=(8,8),cells_per_block =(2,2),block_norm='L1',visualize=True,transform_sqrt=True,feature_vector=True)
        # plt.imshow(hog_image)
        # Appending all the HOG and the corresponding categories in an array
        HOG_values.append(hog)
        labels.append(category)

# Standarize the data
scaler = StandardScaler()
# scaler.fit(HOG_values)
HOG_new = scaler.fit_transform(HOG_values)

# PCA Application on HOG features
pca = PCA(n_components=200)
HOG_PCA_values = pca.fit_transform(HOG_new)

model = LinearSVC(C=0.1)
model.fit(HOG_PCA_values, labels)

pick = open('Linearsvm_model_PCA_v01.sav', 'wb')
pickle.dump(model, pick)
pick.close()

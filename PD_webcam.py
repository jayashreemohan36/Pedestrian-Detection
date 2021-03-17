import os
import cv2
import numpy as np
import joblib
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
import pickle

#Sliding window

def sliding_window(image, window_size, step_size):

    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

#Image Pyramid

def pyramid(image, scale, min_wdw_sz):
    #print('Hello')
    yield image
    while True:
        #print('Hello1')
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if image.shape[0] < min_wdw_sz[1] or image.shape[1] < min_wdw_sz[0]:
            print('inside')
            break
        yield image
        break

clf = open('Linearsvm_model_v2.sav','rb')
svm_model= pickle.load(clf)


cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = imutils.resize(frame, width = min(400, frame.shape[1]))
    #print(image.shape[1])
    #print(image.shape[0])
    #cv2.imshow('Frame',frame)
    #cv2.waitKey()
    #image = cv2.resize(image, (400,400))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 0.5

    #detection array to store the bounding boxes
    detections = []
    scale = 0
    #image pyramid downscale
    for (i, im_scaled) in enumerate(pyramid(image,downscale,min_wdw_sz)):
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):

            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            #im_window = color.rgb2gray(im_window)
            #cv2.imshow('slidingwindow',im_window)
            #cv2.waitKey()
            fd = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm='L1', transform_sqrt=True)
            fd = fd.reshape(1, -1)
            pred = svm_model.predict(fd)
            #print('One window')
            #print(pred)

            if pred == 'pos':
                #svm_model.decision_function(fd)
                #print(svm_model.decision_function(fd))
                if svm_model.decision_function(fd) > 0.5:
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), svm_model.decision_function(fd),
                         int(min_wdw_sz[0] * (downscale ** scale)),
                         int(min_wdw_sz[1] * (downscale ** scale))))

        scale = scale+1
    clone = image.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    # print ("shape, ", pick.shape)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    scale_percent = 70
    w = clone.shape[1]
    h = clone.shape[0]
    scale_percent = 70
    w = image.shape[1]
    h = image.shape[0]

    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)
    dim = (width, height)
    #print(dim)
    resized = cv2.resize(clone, dim, interpolation=cv2.INTER_AREA)
    resized = cv2.resize(image,dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('webcam before nms', image)
    cv2.imshow('webcam after nms', clone)

    if cv2.waitKey(10) == 27:  # Exc key
        break


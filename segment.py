import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

# Run the model on the contours, and the first matching contour should be the required minimum and maximum size
def filterByEvaluatingModel(contours, thresh, models):
    (numbers_model, rank_model, suit_model) = models
    filteredContours = []
    texts = []
    images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # resize image to 28x28
        img = cv2.bitwise_not(thresh[y:y + h, x:x + w])
        img = cv2.resize(img, (28, 28))
        img = np.expand_dims(img, axis=0)

        img = np.array(img)
        img = img.astype('float32')
        img  /= 255

        images.append(img)

    images_array = np.concatenate(images, axis=0)
    number_predictions = numbers_model.predict(images_array)
    rank_predictions = rank_model.predict(images_array)
    suit_predictions = suit_model.predict(images_array)
    suit_classes = ['c', 'd', 'h', '', 's']
    rank_classes = ['A', 'J', 'K',  'Q', '']

    for i in range(len(contours)):
        if 1 > np.max(rank_predictions[i]) > 0.9 and rank_classes[np.argmax(rank_predictions[i])] != '':
            filteredContours.append(contours[i])
            texts.append(rank_classes[np.argmax(rank_predictions[i])])
        elif 1 > np.max(suit_predictions[i]) >= 0.8 and suit_classes[np.argmax(suit_predictions[i])] != '':
            filteredContours.append(contours[i])
            texts.append(suit_classes[np.argmax(suit_predictions[i])])
        elif 1 > np.max(number_predictions[i]) > 0.95:
            filteredContours.append(contours[i])
            texts.append(str(np.argmax(number_predictions[i]) + 2))


    return filteredContours, texts

def filterContoursByMinSize(contours, size):
    filteredContours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > size and h > size:
            filteredContours.append(cnt)
    return filteredContours

def filterContoursByMaxSize(contours, size):
    filteredContours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < size and h < size:
            filteredContours.append(cnt)
    return filteredContours

def displayFilteredContours(img,contours, texts):
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.putText(img, texts[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img

def filterContoursByHavingPairs(contours, texts):
    # contours should have a neighbour in close proximity, above or below
    filteredContours = []
    filteredTexts = []
    paired = []

    for i in range(len(contours)):
        if i in paired:
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        for j in range(len(contours)):
            if j in paired:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])
            distance = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            heightSum = h + h2
            closeEnough = distance < heightSum
            belowEachOther = y < y2 and y + h < y2 + h2
            notFarApart = x - x2 < max(w, w2)
            notTheSame = x != x2 or y != y2

            if closeEnough and belowEachOther and notTheSame and notFarApart:
                filteredContours.append(contours[i])
                filteredTexts.append(texts[i])
                filteredContours.append(contours[j])
                filteredTexts.append(texts[j])
                paired.append(i)
                paired.append(j)
                break

    return filteredContours, filteredTexts

def findAllContours(img, min_size, max_size):
    numbers_model = tf.keras.models.load_model("ML/models/backup/number_model_2.0.h5")
    rank_model = tf.keras.models.load_model("ML/models/backup/values_model_3.0.h5")
    suit_model = tf.keras.models.load_model("ML/models/backup/signs_model_3.0.h5")

    height = 700
    width = int(img.shape[1] * height / img.shape[0])
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bkg_level = gray[int(height / 100)][int(width / 2)]
    thresh_level = bkg_level + 100
    # im_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY , 41, 10) # which is better? I have no idea
    threshold, im_thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_thresh = cv2.bitwise_not(im_thresh)

    # get contours
    contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    innermost_contours = []
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # Check if the contour has no parent (outermost)
            innermost_contours.append(contours[i])
    contours = innermost_contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours.pop(0)

    # remove contours that are inside other contour
    contours = filterContoursByMinSize(contours, min_size)
    contours = filterContoursByMaxSize(contours, max_size)
    contours, texts = filterByEvaluatingModel(contours, im_thresh, (numbers_model, rank_model, suit_model))
    contours, texts = filterContoursByHavingPairs(contours, texts)

    return displayFilteredContours(img,contours, texts)
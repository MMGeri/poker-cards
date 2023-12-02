import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

# Run the model on the contours, and the first matching contour should be the required minimum and maximum size
def filterByEvaluatingModel(contours, thresh, image):
    filteredContours = []
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
    suit_classes = ['c', 'd', 'h', 's']
    rank_classes = ['A', 'J', 'K', 'Q']

    for i in range(len(contours)):
        if 1 > np.max(number_predictions[i]) > 0.9:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.putText(image, str(np.argmax(number_predictions[i])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            filteredContours.append(contours[i])
        elif 1 > np.max(suit_predictions[i]) > 0.9:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.putText(image, suit_classes[np.argmax(suit_predictions[i])], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            filteredContours.append(contours[i])
        elif 1 > np.max(rank_predictions[i]) > 0.9:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.putText(image, rank_classes[np.argmax(rank_predictions[i])], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            filteredContours.append(contours[i])

    return filteredContours, image

def filterContoursByMinSize(contours, size):
    filteredContours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > size and h > size:
            filteredContours.append(cnt)
    return filteredContours

def displayResultAndStore(pairedContours):
    for pairs in pairedContours:
        cnt1 = pairs[0]
        cnt2 = pairs[1]
        x, y, w, h = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 3)

    # write  contours to folder
    for i in range(len(pairedContours)):
        cnt1 = pairedContours[i][0]
        cnt2 = pairedContours[i][1]
        x, y, w, h = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        cv2.imwrite(f"contours/{i}_1.png", im_thresh[y:y + h, x:x + w])
        cv2.imwrite(f"contours/{i}_2.png", im_thresh[y2:y2 + h2, x2:x2 + w2])

def displayFilteredContours(contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

img_path = "assets/test.png"
rank_model = tf.keras.models.load_model("ML/models/backup/values_model_2.0.h5")
numbers_model = tf.keras.models.load_model("ML/models/backup/number_model.h5")
suit_model = tf.keras.models.load_model("ML/models//backup/signs_model.h5")

img= cv2.imread(img_path)
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
cv2.imshow("thresh", im_thresh)
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
contours = filterContoursByMinSize(contours, 10)
contours, img = filterByEvaluatingModel(contours, im_thresh, img)

displayFilteredContours(contours)

cv2.imshow("cards.png", img)
cv2.waitKey(0)

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

numberOfImages = 2500

images = "C:\\Users\\mager\\Desktop\\poker-cards\\ML\\augment_data\\inputs\\replacements"
output = "C:\\Users\\mager\\Desktop\\poker-cards\\ML\\training\\data"
labels = ["A", "Q", "K", "J"]
f = open(f"{output}\\values_train.csv", "w")
f.write("filename,label\n")

# craete values_train dir if not exist
if not os.path.exists(f"{output}\\values_train"):
    os.makedirs(f"{output}\\values_train")

# empty string or 1 with 50% chance
def getEmptyOrOne():
    return np.random.choice(["", "2"], p=[0.5, 0.5])

for label in labels:
    for i in range(numberOfImages):
        image = cv2.imread(f"{images}\\{label}{getEmptyOrOne()}.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        cv2.imwrite(f"{output}\\values_train\\{label}_{i}.jpg", image)

        # create csv file if not exist and write filename,label and then the file name and the label


        f = open(f"{output}\\values_train.csv", "a")
        f.write(f"{label}_{i}.jpg,{label}\n")

f.close()


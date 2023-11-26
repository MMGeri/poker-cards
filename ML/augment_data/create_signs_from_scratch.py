import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

numberOfImages = 1000

images = "C:\\Users\\mager\\Desktop\\poker-cards2\\ML\\augment_data\\inputs\\replacements"
output = "C:\\Users\\mager\\Desktop\\poker-cards2\\ML\\training\\data"
labels = ["c", "s", "h", "d"]
f = open(f"{output}\\signs_train.csv", "w")
f.write("filename,label\n")

# craete signs_train dir if not exist
if not os.path.exists(f"{output}\\signs_train"):
    os.makedirs(f"{output}\\signs_train")

for j in range(1,4):
    for label in labels:
        for i in range(numberOfImages):
            image = cv2.imread(f"{images}\\{label}{j}.png", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            cv2.imwrite(f"{output}\\signs_train\\{label}_{i}_{j}.jpg", image)

            # create csv file if not exist and write filename,label and then the file name and the label
            f = open(f"{output}\\signs_train.csv", "a")
            f.write(f"{label}_{i}_{j}.jpg,{label}\n")
f.close()


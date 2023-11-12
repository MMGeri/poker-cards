import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# read the model path and image path from command line
model_path = "C:\\Users\\mager\\Desktop\\poker-cards\\ML\\models\\values_model.h5" #sys.argv[1]
image_path = "testimg.png" # sys.argv[2]

model = tf.keras.models.load_model(model_path)

image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
classes = 13

#dilate image with opencv
range = 2
negative = -1
kernel = np.ones((range, range), np.uint8)
image = negative*cv2.dilate(np.array(-1*image), kernel, iterations=1)

# input_arr = tf.keras.preprocessing.image.img_to_array(image)
test_images = np.array(image)
test_images = np.array([test_images])  # Convert single image to a batch.
# test_images = test_images.reshape(1, 784)
test_images = test_images.astype('float32')
test_images /= 255

predictions = model.predict(test_images)
plt.bar(np.arange(classes), predictions[0])
plt.show()

plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
# classes = ["clubs", "diamonds", "hearts", "spades"]
classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "king", "queen"]
print("Model prediction:", predictions[0])
print("Model prediction:", classes[np.argmax(predictions[0])])
plt.show()

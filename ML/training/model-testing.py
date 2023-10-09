import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# read the model path and image path from command line
model_path = sys.argv[1]
image_path = sys.argv[2]

# model = tf.keras.models.load_model('models/sudoscan.h5')
model = tf.keras.models.load_model(model_path)

#import hat.png
# image = tf.keras.preprocessing.image.load_img('hat.png', color_mode='grayscale', target_size=(28, 28))
image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))

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
plt.bar(np.arange(4), predictions[0])
plt.show()

plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
classes = ["clubs", "diamonds", "hearts", "spades"]
print("Model prediction:", predictions[0])
print("Model prediction:", classes[np.argmax(predictions[0])])
plt.show()

import os
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

data = "values"

df = pd.read_csv(f"data/{data}.csv")

file_paths = df['filename'].values
labels = df['label'].apply(lambda x: x.strip('.jpg')[-1]).values
# get the 4 label types
print(set(labels))

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)

ds_train = tf.data.Dataset.from_tensor_slices((file_paths, one_hot_labels))

def read_image(image_file, label):
    image = tf.io.read_file(f"data/{data}/" + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    image = tf.image.resize_with_pad(image, target_height=28, target_width=28)
    return image, label

def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=100)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=0, max_jpeg_quality=60)
    image = tf.numpy_function(lambda img: tf.keras.preprocessing.image.random_rotation(
        img, 20, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
    ), [image], tf.float32)
    image = tf.numpy_function(lambda img: tf.keras.preprocessing.image.random_zoom(
        img, (0.9, 1), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
    ), [image], tf.float32)

    return image, label

validation_fraction = 0.2

# Calculate the number of samples for validation
num_samples = len(file_paths)
num_validation_samples = int(validation_fraction * num_samples)

ds_train = ds_train.shuffle(buffer_size=num_samples)
ds_val = ds_train.take(num_validation_samples)
ds_train = ds_train.skip(num_validation_samples)

classes = 13
batch_size = 32
ds_val = ds_val.map(read_image).map(augment).batch(batch_size)
ds_train = ds_train.map(read_image).map(augment).batch(batch_size)

# image, label = next(iter(ds_train))
# _ = plt.imshow(image)
# plt.show()
# exit()

model = Sequential([

    Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', strides=(1, 1), padding='valid',
           kernel_initializer='glorot_normal'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='valid', kernel_initializer='glorot_normal'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),
    Flatten(),

    Dense(128, activation='relu', kernel_initializer='glorot_normal'),
    Dense(64, activation='relu', kernel_initializer='glorot_normal'),
    Dense(classes, activation='softmax'),
])

earlystop = EarlyStopping(monitor="val_loss", patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_accuracy', patience= 2, verbose= 1,
                                            factor= 0.5, min_lr= 0.00001)

epochs = 5
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
history = model.fit(
    ds_train,
    epochs=epochs,
    validation_data= ds_val,
    callbacks=[earlystop,learning_rate_reduction,tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch))]
)

model.save(f"../models/{data}_model.h5")

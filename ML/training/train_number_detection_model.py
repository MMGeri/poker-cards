import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

SQUARE_SIZE = 28

numbers_df = pd.read_csv(f"data/numbers.csv", dtype={'label': str})
numbers_df.head()

def data_flow(data, batch_size):
    print(data.shape)
    print(data.columns)
    return ImageDataGenerator(rotation_range=40, rescale=1./255, shear_range=0.1, zoom_range=0.2,
                              width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.4, 1.6])\
            .flow_from_dataframe(data, f"data/numbers", x_col='file', y_col='label', color_mode='grayscale',
                      target_size=(SQUARE_SIZE, SQUARE_SIZE), class_mode='categorical', batch_size= batch_size, seed=42)


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
    Dense(10, activation='softmax'),
])
########### train
train_data, dummy_data = train_test_split(numbers_df, test_size=0.20, shuffle=True, random_state=42, stratify=numbers_df['label'])
val_data, test_data    = train_test_split(dummy_data, test_size=0.50, shuffle=True, random_state=42, stratify=dummy_data['label'])

train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

batch_size = 32
train_gen  = data_flow(train_data, batch_size)
val_gen    = data_flow(val_data, batch_size)
test_gen   = data_flow(test_data, batch_size)

earlystop = EarlyStopping(monitor="val_loss", patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_accuracy', patience= 2, verbose= 1,
                                            factor= 0.5, min_lr= 0.00001)
checkpoint = ModelCheckpoint(f'../models/number_model.h5', monitor='val_accuracy', save_best_only=True,
                             save_weights_only=False, mode='auto', save_freq='epoch')

epochs = 40
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
history = model.fit(
    train_gen, epochs= epochs, validation_data= val_gen,
    callbacks= [checkpoint, earlystop, learning_rate_reduction]
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import os
import glob as gb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from zipfile import ZipFile
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")
#ZipFile("training_set.zip","r").extractall()
#ZipFile("test_set.zip", "r").extractall()
train_path = "./training_set"
test_path = "./test_set"
print('Total images in Train and Test Data Set:')
print(len(os.listdir(train_path)))
print(len(os.listdir(test_path)))
filenames = os.listdir(train_path)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    categories.append(category)
data = pd.DataFrame({
    'Image': filenames,
    'Category': categories})
print(data.head(10))
sample = random.choice(data['Image'])
plt.imshow(plt.imread(("./training_set/"+sample)))
plt.show()
size  = 150
channels = 3
batch = 128
epochs = 5
Model = Sequential([Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(size, size, channels)),
                    BatchNormalization(),
                    MaxPool2D(2, 2),
                    Dropout(0.2),

                    Conv2D(filters=64, kernel_size=(5, 5), activation="relu"),
                    BatchNormalization(),
                    MaxPool2D(3, 3),
                    Dropout(0.2),

                    Conv2D(filters=128, kernel_size=(7, 7), activation="relu"),
                    BatchNormalization(),
                    MaxPool2D(4, 4),
                    Dropout(0.2),

                    Flatten(),

                    Dense(units=4096, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.4),

                    Dense(units=1024, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(units=2, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.5),

                    Dense(units=1, activation="sigmoid"),
                    ])

Model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

Model.summary()
i = 0
for dirname, _, filenames in os.walk('./training_set'):
    for filename in filenames:
        print(os.path.join(filename))
        img = plt.imread(os.path.join(dirname, filename))
        plt.figure(i)
        plt.imshow(img)
        plt.show()
        i+=1
        if i==7:
            break

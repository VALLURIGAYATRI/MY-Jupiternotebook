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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense , Conv2D, Dropout,                                     Flatten, MaxPooling2D, BatchNormalization)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3),
                    input_shape = (28,28,1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.summary()
print(len(model.layers))
Model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
Model.summary()
print(len(Model.layers))


import numpy as np # linear algebra
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv("sample_submission.csv")
print(train.head())
print(test.head())
train_y = train['label'].astype('float32')
train_x = train.drop(['label'],axis = 1).astype('int32')
test_x = test.astype('float32')

#Printing the shape of train_x and train_y
train_x.shape, train_y.shape, test_x.shape
#Reshaping the image 
train_x = train_x.values.reshape(-1,28,28,1)
#normalisation
train_x = train_x/255.0
test_x = test_x.values.reshape(-1,28,28,1)
test_x=test_x/255.0

#checking the updated shape 
train_x.shape, test_x.shape
plt.imshow(test_x[900].reshape(28,28),cmap = matplotlib.cm.binary)
plt.show()
fig = plt.figure(figsize = (8, 8))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_x[i])
plt.show()
train_y = tf.keras.utils.to_categorical(train_y,10)
train_y.shape  #Printing the shape
print(train['label'].head())  #checking Data type
model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape=(28,28,1)),
      tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', padding = 'Same'),
      tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', padding = 'Same'),
      tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', padding = 'Same'),
      tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', padding = 'Same'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(256,activation='relu'),
      tf.keras.layers.Dropout(0.50),
      tf.keras.layers.Dense(10, activation='softmax')
                         
])
model.summary()




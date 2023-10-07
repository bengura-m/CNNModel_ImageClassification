#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading packages
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


# In[2]:


# Load the CIFAR-100 dataset
cifar100 = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()


# In[3]:


# Summarizing the loaded dataset
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
from matplotlib import pyplot

# Outputting some images
for i in range(9):
     plt.subplot(330 + 1 + i)
     plt.imshow(train_images[i])
pyplot.show()


# In[4]:


# Scaling pixels
def prep_pixels(train, test):
 # convert from integers to floats
 train_norm = train.astype('float32')
 test_norm = test.astype('float32')
 # normalize to range 0-1
 train_norm = train_norm / 255.0
 test_norm = test_norm / 255.0
 # return normalized images
 return train_norm, test_norm

 # prepare pixel data
 train_images, test_images = prep_pixels(train_images, test_images)


# In[5]:


#Creating the model
model = Sequential([
    Conv2D(16, (7, 7), strides=(1, 1), activation='relu',input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(32, (5, 5), strides=(1, 1), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),
    Dense(100, activation='softmax')  # 100 classes in CIFAR-100
])

# Build the model
model.build((None, 32, 32, 3))  # Specify the input shape


# In[6]:


# Displaying the model summary
model.summary()


# In[7]:


# Designing the learning rate schedule
initial_learning_rate = 0.001  
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000, 
    decay_rate=0.9      
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# In[8]:


# Defining early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Training the model
classification_model = model.fit(train_images, train_labels, epochs=30, batch_size=128, validation_split=0.1, callbacks=[early_stopping])


# In[9]:


# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\nTest accuracy:", test_acc)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental import preprocessing


# In[2]:


# Load the CIFAR-100 dataset

cifar100 = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()


# In[3]:


# Create a data augmentation layer
data_augmentation = Sequential([
    preprocessing.Rescaling(1./255),  # Rescale pixel values to [0, 1]
    preprocessing.RandomContrast(factor=0.2),  # Randomly adjust contrast
    preprocessing.RandomFlip(mode='horizontal'),  # Randomly flip horizontally
    preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2)  # Randomly zoom
])


# In[4]:


# summarizing loaded dataset
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
from matplotlib import pyplot

for i in range(9):
     pyplot.subplot(330 + 1 + i)
     pyplot.imshow(train_images[i])
pyplot.show()


# In[5]:


# Build the model
model = Sequential([
    data_augmentation,
    Conv2D(16, (7, 7), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(32, (5, 5), strides=(1, 1), activation='relu'),
    LayerNormalization(),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),#implementing batch normalization
    Dense(100, activation='softmax')  # 100 classes in CIFAR-100
])

#Building the model
model.build((None, 32, 32, 3))  # Specify the input shape


# In[6]:


# Displaying model summary
model.summary()


# In[7]:


# Creating a learning schedule
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


#Definining early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Training the model
classification_model = model.fit(train_images, train_labels, epochs=30, batch_size=128, validation_split=0.1, callbacks=[early_stopping])


# In[9]:


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)


# In[ ]:





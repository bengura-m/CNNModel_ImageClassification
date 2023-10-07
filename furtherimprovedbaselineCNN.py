#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Input, Average, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0


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


# summarize loaded dataset
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
from matplotlib import pyplot

for i in range(9):
     # define subplot
     pyplot.subplot(330 + 1 + i)
     # plot raw pixel data
     pyplot.imshow(train_images[i])
    # show the figure
pyplot.show()


# In[5]:


# Defining base model with EfficientNetBO
base_model =  EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Define input layer(s)
input_layer = Input(shape=(32, 32, 3))  # Adjust the input shape as needed

# Creating version 1 of task2 model
model1 = Sequential([
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

# Creating version 2 of task2 model
model2 = Sequential([
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

# Creating a base model with EfficientModelBO
model3= Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Use Global Average Pooling
    Dense(128, activation='relu'),
    Dense(100, activation='softmax')  # 100 classes in CIFAR-100
])

# Saving the output of each model
output1 = model1(input_layer)
output2 = model2(input_layer)
output3 = model3(input_layer)

# Combining the predictions each of model by averaging the output
combined_output = Average()([output1, output2, output3])

# Creating a new model with the defined input layer and the output of the models
combined_model = Model(inputs=input_layer, outputs=combined_output)

# Building the new and combined model
combined_model.build(input_shape=(None, 32, 32, 3))


# In[6]:


# Display model summary
combined_model.summary()


# In[7]:


initial_learning_rate = 0.001  # Adjust this as needed

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,  # Adjust this as needed
    decay_rate=0.9      # Adjust this as needed
)

# Compile the model
combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# In[8]:


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Train the model
classification_model = combined_model.fit(train_images, train_labels, epochs=30, batch_size=128, validation_split=0.1, callbacks=[early_stopping])


# In[9]:


# Evaluate the model
test_loss, test_acc = combined_model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)


# In[ ]:





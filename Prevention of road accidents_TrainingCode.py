#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


data_dir = "C:/Yawn_NoYawn/train/"


# In[3]:


import pathlib

data_dir = pathlib.Path(data_dir)
data_dir


# In[4]:


list(data_dir.glob('*/*.jpg'))


# In[5]:


image_count = len(list(data_dir.glob('*/*.jpg')))
image_count


# In[6]:


yawn = list(data_dir.glob('yawn/*'))
yawn[:5]


# In[7]:


noyawn = list(data_dir.glob('noyawn/*'))
noyawn[:5]


# In[8]:


PIL.Image.open(str(yawn[0]))


# In[9]:


PIL.Image.open(str(noyawn[0]))


# In[10]:


yawn_image_dict = {
    'yawn':list(data_dir.glob('yawn/*')),
    'noyawn':list(data_dir.glob('noyawn/*')),
}


# In[11]:


yawn_label_dict = {
    'noyawn': 0,
    'yawn': 1,
}


# In[12]:


str(yawn_image_dict['noyawn'][0])


# In[13]:


img = cv2.imread(str(yawn_image_dict['noyawn'][0]))
img.shape


# In[14]:


cv2.resize(img, (224,224)).shape


# In[15]:


X, y = [], []

for yawn_type, images in yawn_image_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)        
        y.append(yawn_label_dict[yawn_type])


# In[16]:


X = np.array(X)
y = np.array(y)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[18]:


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# In[19]:


num_classes = 2

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs = 30)


# In[20]:


model.evaluate(X_test_scaled,y_test)


# In[21]:


predictions = model.predict(X_test_scaled)
predictions


# In[22]:


score = tf.nn.softmax(predictions[0])
score


# In[23]:


np.argmax(score)


# In[24]:


y_test[0]


# In[25]:


img_size = 224

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_size,img_size,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])


# In[26]:


plt.imshow(X[0])


# In[27]:


plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


# In[28]:


num_classes = 2

model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs = 30)


# In[29]:


model.evaluate(X_test_scaled,y_test)


# In[30]:


model.save('yawnDetection.h5')


# In[57]:


img = cv2.imread(str('C:/EyeDetection/girl.jpg'))


# In[58]:


plt.imshow(img)


# In[59]:


test_resize = cv2.resize(img, (224,224))

test=[]
test.append(test_resize)
test = np.array(test)
test_scaled = test/255


# In[ ]:


predictions = model.predict(test_scaled)


# In[65]:


score = tf.nn.softmax(predictions[-1])
score


# In[66]:


np.argmax(score)


# In[ ]:





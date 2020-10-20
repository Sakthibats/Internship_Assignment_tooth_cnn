#!/usr/bin/env python
# coding: utf-8

# # Image Classification of Dental Xray by @Sakthi
# ### Steps
# 1. Import Packages
# 2. Load Data
# 3. Build Model
# 4. Test Model

# # Import Packages
# #### Importing relevant packages and identifying the necessary folders on computer

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pandas as pd

Datadir = r"D:\Jupy\xrays database"
Categories = ["1 root", "2 or more roots"]


# # Load Data
# #### Resize them to the same shape and size in the event they are not all in the same shape or size
# #### Prepare Training dataset

# In[2]:


training_data = []
IMG_SIZE = (130, 230)

def create_training_data():
    for category in Categories:
        path = os.path.join(Datadir, category) #path to images folder
        class_num = Categories.index(category)
        for img in os.listdir(path):#path to images
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, IMG_SIZE)
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()


# In[3]:


random.shuffle(training_data) #shuffles image dataset


# In[4]:


X= [] #Array for images that will be used to train
y= [] #Array for category values that will be used to train


# In[5]:


for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, 130, 230, 1) #Array of Images needed to train model ready
y = np.array(y) #Array of category values to use to train ready
X = X/255 #Normalise


# ## Build Model
# #### The following combinations of layers yield the best accuracy to loss ratio as analysed on Tensorboard

# In[6]:


dense_layers = [1]
layers_sizes = [128]
convo_layers = [1]

for dense_layer in dense_layers:
    for layers_size in layers_sizes:
        for convo_layer in convo_layers:
            Name = "{}-convo-{}-nodes-{}-dense-{}".format(convo_layer,layers_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir = 'logs/{}'.format(Name))
            model = Sequential()
            model.add(Conv2D(layers_size, (3, 3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2,2)))
            
            for l in range(convo_layer-1):#number of Convolutional layers
                
                model.add(Conv2D(layers_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size = (2,2)))

            model.add(Flatten())
            
            for i in range(dense_layer):#number of Dense layers 
                model.add(Dense(layers_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss = "binary_crossentropy", 
                         optimizer = "adam", 
                         metrics = ['accuracy'])
            
            model.fit(X,y, batch_size=3,epochs = 10, validation_split=0.1, callbacks = [tensorboard])


# ### Save the Model for Future Use

# In[7]:


model.save('tooth.cnn.model')


# # Test (For future use)
# ### Create a function to prepare the image for our model and create a method to accept input

# In[ ]:


model = tf.keras.models.load_model("tooth.cnn.model")

#For future folder that will be shared
Test_dir = r"D:\Jupy\xrays database\tester"

testing_imgdata = []#Array for images to be predicted
testing_catdata = []#Array for accurate category values

def test_data():
    for category in Categories:
        path = os.path.join(Test_dir, category) #path to images
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, IMG_SIZE)
                testing_imgdata.append(new_array)
                testing_catdata.append(class_num)
            except Exception as e:
                pass

test_data()

testing_imgdata = np.array(testing_imgdata).reshape(-1, 130, 230, 1)/255.0 #Array of images normalised and ready to be predicted
testing_catdata = np.array(testing_catdata) #Array of accurate category values


# In[ ]:


prediction= model.predict([testing_imgdata])
predicted_val = [int(round(p[0])) for p in prediction]


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n ", confusion_matrix(testing_catdata, predicted_val))
print("Classification report:\n ", classification_report(testing_catdata, predicted_val))


# In[ ]:





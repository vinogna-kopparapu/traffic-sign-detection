from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import cv2
import imutils
import os
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import cv2
import imutils
from keras.optimizers import Adam
from keras.models import model_from_json

files = []
labels  = []

def getLabel(type):
    name = 0
    if type == 'A':
        name = 0
    if type == 'B':
        name = 1
    if type == 'C':
        name = 2
    if type == 'D':
        name = 3
    if type == 'E':
        name = 4    
    return name    
        

directory = 'train'
list_of_files = os.listdir(directory)
for file in list_of_files:
    name = os.path.splitext(os.path.basename(file))[0]
    subfiles = os.listdir(directory+'/'+name)
    print(name)
    for sub in subfiles:
        files.append(directory+'/'+name+'/'+sub)
        labels.append(name)


x_train = np.ndarray(shape=(len(files), 28, 28), dtype=np.float32)
y_train = np.ndarray(shape=(len(files)),dtype=np.float32)
for i in range(len(files)):
    img = image.load_img(files[i], target_size = (28,28), grayscale=True)
    img = np.resize(img, (28,28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(28,28)
    x_train[i] = im2arr
    y_train[i] = getLabel(labels[i])

np.save("train.txt", x_train)
np.save("labels.txt",y_train)
print(x_train.shape)
print(y_train.shape)
print(y_train)


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_train.reshape(x_train.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
unsecure_loaded_model = Sequential()
unsecure_loaded_model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
unsecure_loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
unsecure_loaded_model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
unsecure_loaded_model.add(Dense(128, activation=tf.nn.relu))
unsecure_loaded_model.add(Dropout(0.2))
unsecure_loaded_model.add(Dense(10,activation=tf.nn.softmax))
unsecure_loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
unsecure_loaded_model.fit(x=x_train,y=y_train, epochs=10)
print(unsecure_loaded_model.summary())
unsecure_loaded_model.save_weights('model_weights.h5')
model_json = unsecure_loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

imagetest = image.load_img('A01_0093_small_0.bmp', target_size = (28,28), grayscale=True)
imagetest = image.img_to_array(imagetest)
imagetest = np.expand_dims(imagetest, axis = 0)
pred = unsecure_loaded_model.predict(imagetest.reshape(1, 28, 28, 1))
predicted = str(pred.argmax())
print(predicted)


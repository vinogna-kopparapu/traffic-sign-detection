import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
import tkinter
from tkinter import filedialog
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import imutils
import os
from keras.preprocessing import image
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json

root = tkinter.Tk()
root.title("TRAFFIC SIGN DETECTION AND RECOGNITION")
root.geometry("1000x500")

global train
global video_file
files = []
labels  = []
global lenet_model
#model = Sequential()

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

def uploadTrain():
    global train
    files.clear()
    labels.clear()
    train = filedialog.askdirectory(initialdir = ".")
    trainpath.config(text=train)
    list_of_files = os.listdir(train)
    for file in list_of_files:
        name = os.path.splitext(os.path.basename(file))[0]
        subfiles = os.listdir(train+'/'+name)
        for sub in subfiles:
            files.append(train+'/'+name+'/'+sub)
            labels.append(name)
    trainpath.config(text='Total No Of Train Images : '+str(len(files)))        

def generateModel():
    global lenet_model                    
    x_train = np.ndarray(shape=(len(files), 28, 28), dtype=np.float32)
    y_train = np.ndarray(shape=(len(files)),dtype=np.float32)
    for i in range(len(files)):
        '''
        #img = image.load_img(files[i], target_size = (28,28),grayscale=True)
        '''
        img = image.load_img(files[i], target_size = (28,28))
        img = np.resize(img, (28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(28,28)
        x_train[i] = im2arr
        y_train[i] = getLabel(labels[i])

    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lenet_model = model_from_json(loaded_model_json)
            lenet_model.load_weights("model/model_weights.h5")
            #lenet_model.make_predict_function()   
            print(lenet_model.summary())
            status.config(text='LeNet Model Generated on Train & Test Data. See black console for details')
    else:
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
        lenet_model = Sequential()
        lenet_model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        lenet_model.add(MaxPooling2D(pool_size=(2, 2)))
        lenet_model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        lenet_model.add(Dense(128, activation=tf.nn.relu))
        lenet_model.add(Dropout(0.2))
        lenet_model.add(Dense(10,activation=tf.nn.softmax))
        lenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        lenet_model.fit(x=x_train,y=y_train, epochs=10)
        print(lenet_model.summary())
        lenet_model.save_weights('model/model_weights.h5')
        model_json = lenet_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        status.config(text='LeNet Model Generated on Train & Test Data. See black console for details')

def uploadVideo():
    global video_file
    video_file = askopenfilename(initialdir = "video")
    status.config(text=video_file+' loaded')


def detection():
    video = cv2.VideoCapture(video_file)
    while(True):
        ret, frame = video.read()
        if ret == True:
            imagetest = cv2.resize(frame, (28,28))
            imagetest = cv2.cvtColor(imagetest, cv2.COLOR_BGR2GRAY)
            imagetest = image.img_to_array(imagetest)
            imagetest = np.expand_dims(imagetest, axis = 0)
            predict = lenet_model.predict_classes(imagetest)
            msg = "";
            if str(predict[0]) == '0':
                msg = 'Traffic Sign A Detected'
            if str(predict[0]) == '1':
                msg = 'Traffic Sign B Detected'
            if str(predict[0]) == '2':
                msg = 'No sign Detected'
            if str(predict[0]) == '3':
                msg = 'Diversion'
            if str(predict[0]) == '4':
                msg = 'Traffic sign C Detected'
            text_label = "{}: {:4f}".format(msg, 50)
            cv2.putText(frame, text_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

ttt = Label(root)
ttt.grid(row=0)

ttt1 = Label(root)
ttt1.grid(row=3)

font = ('times', 14, 'bold')
uploadButton = Button(root, text="Upload Traffic Sign Train Images", command=uploadTrain)
uploadButton.grid(row=0)
uploadButton.config(font=font) 

trainpath = Label(root)
trainpath.grid(row=6)
trainpath.config(font=font)

modelButton = Button(root, text="Generate LeNet Model On Train & Test Images", command=generateModel)
modelButton.grid(row=9)
modelButton.config(font=font)

tt = Label(root)
tt.grid(row=12)


videoButton = Button(root, text="Upload Test Video", command=uploadVideo)
videoButton.grid(row=15)
videoButton.config(font=font)

tt1 = Label(root)
tt1.grid(row=18)

signButton = Button(root, text="Start Traffic Sign Detection", command=detection)
signButton.grid(row=21)
signButton.config(font=font)

tt2 = Label(root)
tt2.grid(row=24)

status = Label(root)
status.grid(row=27)
status.config(font=font)
root.mainloop()

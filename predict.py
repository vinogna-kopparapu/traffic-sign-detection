from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import cv2
import imutils

with open('model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    unsecure_loaded_model = model_from_json(loaded_model_json)

unsecure_loaded_model.load_weights("model_weights.h5")
unsecure_loaded_model._make_predict_function()   
print(unsecure_loaded_model.summary())



video = cv2.VideoCapture('video0.wmv')
while(True):
    ret, frame = video.read()
    print(ret)
    if ret == True:
        #cv.imwrite("test.jpg",frame)
        #imagetest = image.load_img("test.jpg", target_size = (150,150))
        imagetest = cv2.resize(frame, (28,28))
        imagetest = cv2.cvtColor(imagetest, cv2.COLOR_BGR2GRAY)
        imagetest = image.img_to_array(imagetest)
        imagetest = np.expand_dims(imagetest, axis = 0)
        predict = unsecure_loaded_model.predict_classes(imagetest)
        print(predict[0])
        msg = "";
        if str(predict[0]) == '0':
            msg = 'A'
        if str(predict[0]) == '1':
            msg = 'B'
        if str(predict[0]) == '2':
            msg = 'No Signs detected'
        if str(predict[0]) == '3':
            msg = 'Diversion'
        if str(predict[0]) == '4':
            msg = 'No Signs detected'
        text_label = "{}: {:4f}".format(msg, 80)
        cv2.putText(frame, text_label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()

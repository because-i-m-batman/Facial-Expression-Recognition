from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import operator


#Emotion Detection Class
class EmotionDetection:

    #Load Model
    def __init__(self):
        print('Emotion Detection')
        self.model = load_model('Your Emotion Model Path')
        
    def emotionDetection(self,img):

        img = cv2.resize(img,(36,36))

        img = img.astype("float") / 255.0
        image = img_to_array(img)
        image = np.expand_dims(img, axis=0)




        pred = self.model.predict(image)

        pred = pred.tolist()
        import itertools
        flatten = itertools.chain.from_iterable
        pred = list(flatten(pred))
        index, value = max(enumerate(pred), key=operator.itemgetter(1))
        if index == 0:
            print("Angry",value)
        elif index == 1:
            print("Disgust",value)
        elif index == 2:
            print("Fear",value)
        elif index == 3:
            print("Happy",value)
        elif index == 4:
            print("Neutral",value)
        elif index == 5:
            print("Sad",value)
        elif index ==6:
            print("Surprise",value)

        return image,pred
e = EmotionDetection()


image = cv2.imread('Your Image Path')
image,emotion = e.emotionDetection(image)
plt.imshow(image)
plt.show()
print(emotion)






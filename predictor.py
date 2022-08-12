import numpy as np
import keras
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
saved_model = load_model("Tumor_classifier_model.h5")
status = True


def check(input_img):
    print("your image is : "+input_img)
    print(input_img)
    img = image.load_img("images/"+input_img,target_size=(224,224))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)
    
    input_data=np.array(img)
    input_data=input_data/225
    output = saved_model.predict(input_data)

    if output >= 0.5:
        status = True
    else:
        status = False

    print(status)
    return status
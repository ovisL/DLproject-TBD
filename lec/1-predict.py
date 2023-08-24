import tensorflow as tf 
import cv2 
import numpy as np


model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

w, h = 512, 512
num_landmarks = 106

image_path = 'image/test0.jpg'

image_origin = cv2.imread(image_path, cv2.IMREAD_COLOR)

image = cv2.resize(image_origin, (w, h))
image = image/255.0 
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

pred = model.predict(image, verbose=0)[0]
pred = pred.astype(np.float32)

def returnLankmarks(img, pred):
    h, w, _ =  img.shape
    landmarks = []

    for i in range(0, len(pred), 2):
        x, y = pred[i]*w, pred[i+1]*h
        landmarks.append([int(x),int(y)])
    return landmarks

def drawLanmarks(img, landmarks) :
    h, w, _ =  img.shape
    radius = int(h * 0.005)

    for lanmark in landmarks :
        img = cv2.circle(img, lanmark, radius, (255, 0, 0), -1)
    return img

landmarks = returnLankmarks(image_origin, pred)
image_landmark = drawLanmarks(image_origin, landmarks[51:66])
# nose range : 51:66

cv2.imshow('image with landmarks',image_landmark)
cv2.waitKey(0)
cv2.imwrite('1-image_landmark.png',image_landmark)


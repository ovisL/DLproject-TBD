import tensorflow as tf 
import pandas as pd
import cv2 
import numpy as np

def plot_lankmarks(image, landmarks):
    h, w, _ =  image.shape
    radius = int(h * 0.005)

    for i in range(0, len(landmarks), 2):
        x = int(landmarks[i] * w)
        y = int(landmarks[i+1] * h)

        image = cv2.circle(image, (x, y), radius, (255, 0, 0), -1)

    return image


model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

image_h = 512
image_w = 512
radius = int(image_h * 0.005)
num_landmarks = 106

image_path = 'image/test0.jpg'

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (image_w, image_h))
image_x = image
image = image/255.0 ## (512, 512, 3)
image = np.expand_dims(image, axis=0) ## (1, 512, 512, 3)
image = image.astype(np.float32)
# print(image_x.shape)
pred = model.predict(image, verbose=0)[0]
# print(pred)
# pred = pred
pred = pred.astype(np.float32)

image_h, image_w, _ = image_x.shape
landmarks = []
for i in range(0, len(pred), 2):
    x, y = pred[i]*image_w, pred[i+1]*image_h
    landmarks.append([int(x),int(y)])


print(pred)
print(len(pred))
print(landmarks)
# pred_landmarks = plot_lankmarks(image_x, pred)

for lanmark in landmarks[51:66] :
    # print(i+1)
    image_x = cv2.circle(image_x, lanmark, radius, (255, 0, 0), -1)
    cv2.imshow('pred',image_x)
    cv2.waitKey(0)
# cv2.imwrite('a.png',image_x)
# cv2.waitKey(0)

# cv2.imshow('pred',pred_landmarks)
# cv2.waitKey(0)
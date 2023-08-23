import tensorflow as tf 
import cv2 
import numpy as np
import math

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

model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

h, w = 512, 512
num_landmarks = 106

image_path = 'image/test0.jpg'

image_origin = cv2.imread(image_path, cv2.IMREAD_COLOR)

image = cv2.resize(image_origin, (w, h))
image = image/255.0 
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

pred = model.predict(image, verbose=0)[0]
pred = pred.astype(np.float32)

landmarks = returnLankmarks(image_origin, pred)
image_landmark = drawLanmarks(image_origin, landmarks[51:66])
# nose range : 51:66

buri_path = 'buri.png'
buri = cv2.imread(buri_path,  cv2.IMREAD_UNCHANGED)

ratio = (((landmarks[64][0]-landmarks[57][0])**2 + (landmarks[64][1]-landmarks[57][1])**2)**(1/2))/buri.shape[1]*1.1
theta = math.atan((landmarks[54][1]-landmarks[51][1])/(landmarks[54][0]-landmarks[51][0]))

buri = cv2.resize(buri, (int(buri.shape[1]*ratio), int(buri.shape[0]*ratio)))

from PIL import Image
buri_pil = Image.fromarray(buri)
buri_rotated = buri_pil.rotate(theta, expand=1)

buri_rotated = np.array(buri_rotated)
cv2.imwrite('buri_rotated.png',buri_rotated)
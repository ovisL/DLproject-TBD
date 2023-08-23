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

image_path = 'test_image/test0.jpg'

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

buri_path = 'image/buri.png'
buri = cv2.imread(buri_path,  cv2.IMREAD_UNCHANGED)

ratio = (((landmarks[64][0]-landmarks[57][0])**2 + (landmarks[64][1]-landmarks[57][1])**2)**(1/2))/buri.shape[1]*1.1
theta = math.atan((landmarks[54][1]-landmarks[51][1])/(landmarks[54][0]-landmarks[51][0]))

buri = cv2.resize(buri, (int(buri.shape[1]*ratio), int(buri.shape[0]*ratio)))

from PIL import Image
buri_pil = Image.fromarray(buri)
buri_rotated = buri_pil.rotate(theta, expand=1)

buri_rotated = np.array(buri_rotated)
# cv2.imwrite('image/buri_rotated.png',buri_rotated)
buri_coord_x, buri_coord_y = int(landmarks[52][0]-(buri.shape[1]*math.sin(theta)/2)), int(landmarks[52][1]-(buri.shape[1]*math.cos(theta)/2))
print(theta, buri_coord_x,buri_coord_y)
image_origin = cv2.circle(image_origin, (landmarks[52][0],landmarks[52][1]), 3, (0, 255, 0), -1)

image_origin = cv2.circle(image_origin, (buri_coord_x,buri_coord_y), 3, (0, 255, 0), -1)

def mergeImage(img_bg, img_fg,x,y) :
    _, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[y:y+h, x:x+w ]

    #--④ 마스크 이용해서 오려내기
    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    #--⑥ 이미지 합성
    added = masked_fg + masked_bg
    img_bg[y:y+h, x:x+w ] = added
    return img_bg
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask_inv', mask_inv)
    # cv2.imshow('masked_fg', masked_fg)
    # cv2.imshow('masked_bg', masked_bg)
    # cv2.imshow('added', added)
    # cv2.imshow('result', img_bg)
    # cv2.imwrite('res.jpg',img_bg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

img_added = mergeImage(image_origin, buri_rotated, buri_coord_x, buri_coord_y)
cv2.imwrite('img_added.jpg',img_added)
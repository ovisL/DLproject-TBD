import tensorflow as tf 
import cv2 
import numpy as np
import math
from PIL import Image

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

def rotateImage(img, theta) :
    img_pil = Image.fromarray(img)
    img_rotated = img_pil.rotate(theta, expand=1)
    img_rotated = np.array(img_rotated)

    mask = img_rotated[:,:,3]
    mask_h, mask_w = mask.shape
    crop_x = []
    crop_y = []
    for i in range(mask_w) :
        for j in range(mask_h) :
            if mask[j][i] != 0:
                crop_x.append(i) 
                crop_y.append(j)    
    crop_x.sort()
    crop_y.sort()

    img_rotated = img_rotated[crop_y[0]:,crop_x[0]:] 
    return img_rotated

def mergeImage(img_bg, img_fg,x,y) :
    _, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
    h, w = img_fg.shape[:2]
    roi = img_bg[y:y+h, x:x+w]

    masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    added = masked_fg + masked_bg
    img_bg[y:y+h, x:x+w] = added
    return img_bg

model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

h, w = 512, 512
num_landmarks = 106

image_path = 'image/test0.jpg'

image_origin = cv2.imread(image_path, cv2.IMREAD_COLOR)
cv2.imshow('original image',image_origin)
# cv2.waitKey(0)


image = cv2.resize(image_origin, (w, h))
image = image/255.0 
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

pred = model.predict(image, verbose=0)[0]
pred = pred.astype(np.float32)

landmarks = returnLankmarks(image_origin, pred)
# image_landmark = drawLanmarks(image_origin, landmarks[51:66])
# # nose range : 51:66

# cv2.imshow('image with landmarks',image_landmark)
cv2.waitKey(0)
# cv2.imwrite('3-image_landmark.jpg', image_landmark)


buri_path = 'src/buri.png'
buri = cv2.imread(buri_path,  cv2.IMREAD_UNCHANGED)
cv2.imshow('buri image',buri)
cv2.waitKey(0)


ratio = ( ( (landmarks[64][0]-landmarks[57][0])**2 + (landmarks[64][1]-landmarks[57][1])**2 )**(1/2) ) / buri.shape[1]
theta = math.atan((landmarks[54][1]-landmarks[51][1])/(landmarks[54][0]-landmarks[51][0]))

buri = cv2.resize(buri, (int(buri.shape[1]*ratio), int(buri.shape[0]*ratio)))
    
buri_rotated = rotateImage(buri, theta)
cv2.imshow('rotated buri',buri_rotated)
cv2.waitKey(0)
cv2.imwrite('3-buri_rotated.png',buri_rotated)

buri_coord_x = int(landmarks[52][0]-(buri.shape[1]*math.sin(theta)/2))
buri_coord_y = int(landmarks[52][1]-(buri.shape[1]*math.cos(theta)/2))


img_merged = mergeImage(image_origin, buri_rotated, buri_coord_x, buri_coord_y)
cv2.imshow('image with buri',img_merged)
cv2.waitKey(0)
cv2.imwrite('3-img_merged.jpg',img_merged)


brow_path = 'src/brow.png'
brow = cv2.imread(brow_path,  cv2.IMREAD_UNCHANGED)
cv2.imshow('brow image',brow)
cv2.waitKey(0)


ratio_brow = ( ( (landmarks[0][0]-landmarks[32][0])**2 + (landmarks[0][1]-landmarks[32][1])**2 )**(1/2) ) / brow.shape[1]
theta_brow = math.atan((landmarks[0][1]-landmarks[32][1])/(landmarks[0][0]-landmarks[32][0]))

brow = cv2.resize(brow, (int(brow.shape[1]*ratio_brow), int(brow.shape[0]*ratio_brow)))
    
brow_rotated = rotateImage(brow, theta_brow)
cv2.imshow('rotated brow',brow_rotated)
cv2.waitKey(0)
cv2.imwrite('3-brow_rotated.png',brow_rotated)

brow_coord_x = int(landmarks[0][0]-(brow.shape[1]/15))
brow_coord_y = int(landmarks[0][1]-(brow.shape[1]*14.5/5.3))


img_merged_brow = mergeImage(img_merged, brow_rotated, brow_coord_x, brow_coord_y)
cv2.imshow('image with brow',img_merged_brow)
cv2.waitKey(0)
cv2.imwrite('3-img_merged_brow.jpg',img_merged_brow)
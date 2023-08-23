import cv2
import numpy as np

img_fg = cv2.imread('image/buri_rotated.png', cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread('test_image/test0.jpg')

# img_fg = cv2.resize(img_fg, (95, 299))
# img_bg = cv2.resize(img_bg,(600,800))


_, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
h, w = img_fg.shape[:2]
roi = img_bg[1599:1599+h, 1382:1382+w ]

#--④ 마스크 이용해서 오려내기
masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#--⑥ 이미지 합성
added = masked_fg + masked_bg
img_bg[1599:1599+h, 1382:1382+w ] = added

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('masked_fg', masked_fg)
cv2.imshow('masked_bg', masked_bg)
cv2.imshow('added', added)
cv2.imshow('result', img_bg)
cv2.imwrite('res.jpg',img_bg)
cv2.waitKey()
cv2.destroyAllWindows()

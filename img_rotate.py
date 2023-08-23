import cv2

# src = cv2.imread("apple.jpg", cv2.IMREAD_COLOR)

# height, width, channel = src.shape
# matrix = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
# dst = cv2.warpAffine(src, matrix, (width, height))

# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import imutils
# rotated = imutils.rotate_bound(src, 30)
# cv2.imshow('totated',rotated)
# cv2.waitKey(0)
# cv2.imwrite('rotated_buri.png',rotated)



# Pillow 라이브러리 불러오기
from PIL import Image

# 고양이 이미지 불러와서 img라는 변수에 입력
img = Image.open('buri.png')
# 이미지 90도 회전
img_rotated = img.rotate(30, expand=1)
print(img_rotated)
# 회전한 이미지 출력
img_rotated.show()

a = cv2.resize(img_rotated,(100,100))
import cv2
import numpy as np
from PIL import Image

buri_path = 'image/buri.png'
buri = cv2.imread(buri_path,  cv2.IMREAD_UNCHANGED)


buri_pil = Image.fromarray(buri)
buri_rotated = buri_pil.rotate(30, expand=1)
buri_rotated = np.array(buri_rotated)
# print(buri_rotated.shape)
# cv2.imshow('d',buri_rotated)
# cv2.waitKey(0)
mask = buri_rotated[:,:,3]
h,w = mask.shape

file_name = 'text.txt'

# with open(file_name, 'w+') as file:
#     for i in mask :
#         for j in i :
#             file.write(str(j))
#         file.write('\n')# '\n' 대신 ', '를 사용하면 줄바꿈이 아닌 ', '를 기준으로 문자열 구분함
crop_x = []
crop_y = []
print(mask)
for i in range(w) :
    key = False
    for j in range(h) :
        # print(mask[j][i])
        if mask[j][i] != 0:
            crop_x.append(i) 
            crop_y.append(j)
            # print(i,j)
    #         key = True
    #         break
    # if key == True : break    
crop_x.sort()
crop_y.sort()
# print(crop_y,crop_x)


buri_rotated = buri_rotated[crop_y[0]:,crop_x[0]:] 
cv2.imwrite('aaaa.png',buri_rotated)

import numpy as np
import cv2

imageSet = 6
index = 1

temp = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
h, w, _ = temp.shape
newimg = np.zeros((h, w, 3), dtype = np.uint8)

for index  in range(0, 5):
    img = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
    indices1 = (img != 0)
    indices2 = (newimg != 0)
    overlapMask = np.logical_and(indices1, indices2)
    nonOverlapMask = np.logical_not(np.logical_and(indices1, indices2))
    newimg[nonOverlapMask] += img[nonOverlapMask]
    newimg[overlapMask] = img[overlapMask]
    cv2.imwrite('outputs/l' + str(imageSet) + '/' + 'blended.png', newimg)
    print(index, imageSet)




cv2.imwrite('outputs/l' + str(imageSet) + '/' + 'blended.png', newimg)
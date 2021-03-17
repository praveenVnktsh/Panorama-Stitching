import cv2
import numpy as np

depth = 6

imageSet = 6
index = 2

img1 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index + 1) + '.png')

mask1 = img1[:, :, 0] != 0 
mask1 = np.logical_and(img1[:, :, 1] != 0, mask1)
mask1 = np.logical_and(img1[:, :, 2] != 0, mask1)

mask2 = img2[:, :, 0] != 0 
mask2 = np.logical_and(img2[:, :, 1] != 0, mask2)
mask2 = np.logical_and(img2[:, :, 2] != 0, mask2)

imgMask2 = np.zeros((img1.shape[0], img2.shape[0]), dtype = np.float)
imgMask2[mask2] = 1.0
imgMask2 = cv2.blur(imgMask2, (50, 50))
imgMask2 = cv2.merge((imgMask2, imgMask2, imgMask2))

imgMask1 = np.zeros((img1.shape[0], img2.shape[0]), dtype = np.float)
imgMask1[mask1] = 1.0
imgMask1 = cv2.blur(imgMask1, (50, 50))
imgMask1 = cv2.merge((imgMask1, imgMask1, imgMask1))

# finalMask = np.zeros((img1.shape[0], img2.shape[0]), dtype = np.float)
# finalMask[mask2] = 1.0
# finalMask[mask1 & mask2] = commonMask[mask1 & mask2]

cv2.imshow('mask1', np.array(imgMask1 * img1, dtype= np.uint8))
cv2.waitKey(0)

cv2.imshow('mask2', np.array(imgMask2 * img2, dtype= np.uint8))
cv2.waitKey(0)

cv2.imwrite('output.png', np.array(imgMask2 * img2 + imgMask1 * img1, dtype= np.uint8))
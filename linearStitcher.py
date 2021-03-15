import cv2
import numpy as np

depth = 6

imageSet = 6
index = 3

img1 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index + 1) + '.png')

mask1 = img1[:, :, 0] != 0 
mask1 = np.logical_and(img1[:, :, 1] != 0, mask1)
mask1 = np.logical_and(img1[:, :, 2] != 0, mask1)

mask2 = img2[:, :, 0] != 0 
mask2 = np.logical_and(img2[:, :, 1] != 0, mask2)
mask2 = np.logical_and(img2[:, :, 2] != 0, mask2)

netMaskIndices = np.logical_and(mask1, mask2),

mask = np.zeros((img1.shape[0], img1.shape[1]), dtype = np.uint8)

mask[netMaskIndices] = 255
overlapMask = cv2.merge((mask, mask, mask))

cv2.namedWindow('im', flags= cv2.WINDOW_GUI_NORMAL)
# cv2.imshow('im', overlapMask) 
# cv2.waitKey(0)

ret, contours,_ = cv2.findContours(mask.astype(np.uint8), 1, 1) 
contours = sorted(contours, key=cv2.contourArea, reverse=True)
rect = cv2.minAreaRect(contours[0])
(x,y),(w,h), a = rect 
box = cv2.boxPoints(rect).astype(np.int)
print(box)

w = np.max(box[:, 0]) - np.min(box[:, 0]) + 10
h = np.max(box[:, 1]) - np.min(box[:, 1]) + 10

minx = np.min(box[:, 0])
miny = np.min(box[:, 1])


print(w, h)
powe= 1 
# horizontal
lins = np.power(np.linspace(0, 1, int(w)), powe)
tile = np.tile(lins, (int(h), 1))[:, :, np.newaxis]
print(tile.shape)
mask1t = np.repeat(tile, 3, axis=2)
newMask1 = np.zeros_like(overlapMask, dtype= np.float)
print(mask1t.shape)

gradientBox = np.zeros_like(mask1, dtype=bool)
gradientBox[miny : miny + h , minx : minx + w] = True


noOverlapMask = np.logical_or(mask1, mask2)
newMask1[noOverlapMask] = 1.0
newMask1[miny : miny + h , minx : minx + w] =  mask1t
newMask1[np.logical_not(mask1 + (mask1 & gradientBox))] = 0.0
newMask1[(gradientBox.astype(np.float32) - mask2.astype(np.float32)).astype(np.bool)] = 1.0
lins = np.power(np.linspace(1, 0, int(w)), powe)
tile = np.tile(lins, (int(h), 1))[:, :, np.newaxis]
print(tile.shape)
mask1t = np.repeat(tile, 3, axis=2)
newMask2 = np.zeros_like(overlapMask, dtype= np.float)

print(mask1t.shape)

noOverlapMask = np.logical_or(mask1, mask2)
newMask2[noOverlapMask] = 1.0
newMask2[miny : miny + h , minx : minx + w] =  mask1t
newMask2[np.logical_not(mask2 + (mask2 & gradientBox))] = 0.0
newMask2[(gradientBox.astype(np.float32) - mask1.astype(np.float32)).astype(np.bool)] = 1.0
print(np.max(newMask1), np.max(newMask2))


cv2.imshow('im', np.array( newMask1))
cv2.waitKey(0)

cv2.imshow('im', np.array( newMask2))
cv2.waitKey(0)
cv2.imshow('im', np.array(img2 * newMask2 , dtype=np.uint8))
cv2.waitKey(0)

cv2.imshow('im', np.array(img2 * newMask2  + img1 * newMask1 , dtype=np.uint8))
cv2.waitKey(0)

finalImg = np.zeros_like(img1, dtype= np.uint8)
finalImg += np.array(img1 * newMask1 , dtype=np.uint8)
# finalImg = np.zeros_like(img1, dtype= np.uint8)
finalImg += np.array(img2 * newMask2 , dtype=np.uint8)


cv2.imwrite('output.png', finalImg)
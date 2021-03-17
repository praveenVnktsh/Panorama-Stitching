import cv2
import numpy as np
global  imageSet   
opencv = False
imageSet = 6
if opencv:
    st = 'opencv'
else:
    st = ''
img1 = cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(0) + '.png')
cv2.namedWindow('mask2', flags=cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('mask1', flags=cv2.WINDOW_GUI_NORMAL)
finalImg = np.zeros_like(img1, dtype= np.uint8)
for index in range(1, 5):
    if index != 1:
        img1 = finalImg
    img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(index ) + '.png')
    print('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(index ) + '.png')

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

    ret, contours,_ = cv2.findContours(mask.astype(np.uint8), 1, 1) 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(contours[0])
    (x,y),(w,h), a = rect 
    box = cv2.boxPoints(rect).astype(np.int)

    w = np.max(box[:, 0]) - np.min(box[:, 0]) + 10
    h = np.max(box[:, 1]) - np.min(box[:, 1]) + 10

    minx = np.min(box[:, 0])
    miny = np.min(box[:, 1])


    powe= 1 
    # horizontal
    lins = np.power(np.linspace(1, 0, int(w)), powe)
    tile = np.tile(lins, (int(h), 1))[:, :, np.newaxis]
    mask1t = np.repeat(tile, 3, axis=2)
    newMask1 = np.zeros_like(overlapMask, dtype= np.float)

    gradientBox = np.zeros_like(mask1, dtype=bool)
    gradientBox[miny : miny + h , minx : minx + w] = True


    noOverlapMask = np.logical_or(mask1, mask2)
    newMask1[noOverlapMask] = 1.0
    newMask1[miny : miny + h , minx : minx + w] =  mask1t
    
    newMask1[(gradientBox.astype(np.float32) - mask2.astype(np.float32)).astype(np.bool) & mask1] = 1.0
    newMask1[np.logical_not(mask1 + (mask1 & gradientBox))] = 0.0
    lins = np.power(np.linspace(0, 1, int(w)), powe)
    tile = np.tile(lins, (int(h), 1))[:, :, np.newaxis]
    mask1t = np.repeat(tile, 3, axis=2)
    newMask2 = np.zeros_like(overlapMask, dtype= np.float)


    noOverlapMask = np.logical_or(mask1, mask2)
    newMask2[noOverlapMask] = 1.0
    newMask2[miny : miny + h , minx : minx + w] =  mask1t
    newMask2[np.logical_not(mask2 + (mask2 & gradientBox))] = 0.0
    newMask2[(gradientBox.astype(np.float32) - mask1.astype(np.float32)).astype(np.bool) & mask2] = 1.0

    finalImg = np.array(img1 * newMask1 , dtype=np.uint8)
    finalImg += np.array(img2 * newMask2 , dtype=np.uint8)

    if opencv:
        cv2.imwrite('outputs/l' + str(imageSet) + '/' + st + 'output.png', finalImg)
    else:
        cv2.imwrite('outputs/l' + str(imageSet) + '/' +  st + 'output.png', finalImg)

    cv2.imshow('mask2', newMask2)
    cv2.imshow('mask1', newMask1)
    
    if cv2.waitKey(0) == ord('q'):
        exit()
    
    
    cv2.imshow('mask2', np.array(newMask2 * img2, dtype = np.uint8))
    cv2.imshow('mask1', np.array(newMask1 * img1, dtype = np.uint8))
    if cv2.waitKey(0) == ord('q'):
        exit()

cv2.imshow('mask1', finalImg)
if cv2.waitKey(0) == ord('q'):
    exit()
    
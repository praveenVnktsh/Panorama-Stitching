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

  
    gradientBox = np.zeros_like(mask1, dtype=bool)
    gradientBox[mask1 & mask2] = True
    gradientBox[np.logical_not(mask1)] = False

    

    newMask2 = np.zeros_like(overlapMask, dtype= np.float)
    newMask1 = np.zeros_like(overlapMask, dtype= np.float)

    
    

    newMask1[mask1 ^ (mask1 & gradientBox)] = 1.0
    # newMask1 = cv2.dilate(newMask1, (200, /200))
    cv2.imshow('mask', newMask1)
    cv2.waitKey(0)
    newMask1 =  cv2.blur(newMask1, (40, 40))
    cv2.imshow('mask', newMask1)
    cv2.waitKey(0)
    ret, newMask1 = cv2.threshold(newMask1, 0.02, 1.0, cv2.THRESH_BINARY)
    
    cv2.imshow('mask', newMask1)
    cv2.waitKey(0)
    # print(np.max(newMask1))
    threshIndices = newMask1 == 1.0

    # newMask1[gradientBox] = 0.5
    newMask1[mask1 ^ (mask1 & gradientBox)] = 0.0
    newMask1 =  cv2.blur(newMask1, (20, 20))
    # newMask1 = cv2.erode(newMask1, (50, 50), iterations = 20)
    cv2.imshow('mask', newMask1)
    cv2.waitKey(0)
    newMask1[mask1 ^ (mask1 & gradientBox)] = 1.0
    cv2.imshow('mask', newMask1)
    cv2.waitKey(0)
    # newMask1 = cv2.erode(newMask1, (50, 50), iterations = 20)
    # newMask1[threshIndices] = 1.0
    

    newMask2[mask2] = 1.0
    newMask2[gradientBox] = 1.0 - newMask1[gradientBox]




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
    
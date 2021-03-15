import glob
import cv2
import numpy as np
import os

imageSet = 4
imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
os.makedirs('outputs/l' + str(imageSet) + '/', exist_ok = True)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def detectFeaturesAndMatch(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)


    cv2.imshow('im', img3)
    cv2.waitKey(1)
   
    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    return src, dst


for i in range(1, len(imagePaths)):
    print('Comparing', imagePaths[i], 'and', imagePaths[i-1])
    img1 = cv2.imread(imagePaths[i-1])
    img2 = cv2.imread(imagePaths[i])
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.resize(img2, (600, 400))

    src, dst = detectFeaturesAndMatch(img1, img2)
    H, subsetMatches = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    warpedImg2, offset = transformImage(img2, H)
    cv2.imwrite('outputs/l' + str(imageSet) + '/warped.png', warpedImg2)


    stitched = blendImages(img1, warpedImg2, offset, subsetMatches, H)
    cv2.imwrite('outputs/l' + str(imageSet) + '/stitched.png', stitched)


    exit()
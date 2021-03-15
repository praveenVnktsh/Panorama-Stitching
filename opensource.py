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


def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img


for i in range(1, len(imagePaths)):
    print('Comparing', imagePaths[i], 'and', imagePaths[i-1])
    img1 = cv2.imread(imagePaths[i-1])
    img2 = cv2.imread(imagePaths[i])
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.resize(img2, (600, 400))

    src, dst = detectFeaturesAndMatch(img1, img2)
    H, subsetMatches = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    warpedImg2, offset = warpImages(img1, img2, H)
    cv2.imwrite('outputs/l' + str(imageSet) + '/warped.png', warpedImg2)


    stitched = blendImages(img1, warpedImg2, offset, subsetMatches, H)
    cv2.imwrite('outputs/l' + str(imageSet) + '/stitched.png', stitched)


    exit()
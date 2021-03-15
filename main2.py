import glob
import cv2
import numpy as np
import os
from tqdm import tqdm



def detectFeaturesAndMatch(img1, img2, nFeaturesReturn = 30):
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    correspondences = []
    for match in matches:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    return np.array(correspondences[:nFeaturesReturn])

def getHomography(matches):
    
    A = np.zeros((2*len(matches), 9))
    for i, match in enumerate(matches):
        src = match[0]
        dst = match[1]
        x1 = src[0]
        x2 = dst[0]
        y1 = src[1]
        y2 = dst[1]
        A[2*i] = np.array([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A[2*i+1] = np.array([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
    
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3) )
    return H

def getBestHomographyRANSAC(matches, trials = 10000, threshold = 10):
    finalH = None
    nMaxInliers = 0
    randomSample = None
    # while nMaxInliers < 25:
    for trialIndex in tqdm(range(trials)):
        inliers = []
        randomSample = matches[np.random.choice(len(matches), size=4, replace=False)]
        H = getHomography(randomSample)
        for match in matches:
            src = np.append(match[0], 1).T
            dst = np.append(match[1], 1).T
            transformed = np.dot(H, src)
            transformed /= transformed[2]
            if np.linalg.norm(transformed - dst) < threshold:
                inliers.append(match)
        if len(inliers) > nMaxInliers:
            nMaxInliers = len(inliers)
            finalH = H
    print('Max inliers = ', nMaxInliers)
    threshold += 1
    return finalH, randomSample

def transformPoint(i, j, H):
    transformed = np.dot(H, np.array([i, j, 1]))
    transformed /= transformed[2]
    transformed = transformed.astype(np.int)[:2]
    return np.array(transformed)

def transformImage(img, H, forward = False):
    global offset
    h, w, _ = img.shape
    if forward:
        coords = np.indices((w, h)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)    
        transformedPoints = np.dot(H, coords)
        yo, xo = coords[1, :], coords[0, :]
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)
        transformedImage[yt + offset[1], xt + offset[0]] = img[yo, xo]
    else:

        Hinv = np.linalg.inv(H)
        topLeft = transformPoint(0, 0, H) 
        topRight = transformPoint(w-1, 0, H) 
        bottomLeft = transformPoint(0, h-1, H) 
        bottomRight = transformPoint(w-1, h-1, H)
        box = np.array([topLeft, topRight, bottomLeft, bottomRight])
        minX = np.min(box[:, 0])
        maxX = np.max(box[:, 0])
        minY = np.min(box[:, 1])
        maxY = np.max(box[:, 1])

        coords = np.indices((maxX - minX, maxY - minY)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)   

        print(coords)
        print(minX, maxX, minY, maxY) 

        coords[0, :] += minX
        coords[1, :] += minY
        transformedPoints = np.dot(Hinv, coords)
        yo, xo = coords[1, :], coords[0, :]
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)

        indices = np.where((yt >= 0) & (yt < h) & (xt >= 0) & (xt < w))

        xt = xt[indices]
        yt = yt[indices]

        xo = xo[indices]
        yo = yo[indices]
        transformedImage[yo + offset[1], xo + offset[0]] = img[yt, xt]


    



    


imageSet = 4
imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
os.makedirs('outputs/l' + str(imageSet) + '/', exist_ok = True)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

shape = (2400, 1600)

offsets = [(0, 0)]
prevH = np.eye(3)

mid = int(len(imagePaths)/2)
nextImage = mid + 1


transformedImage = np.zeros((15000, 15000, 3))

threshold = 5
trials = 5000
offset = [5000, 5000]

for index in range(mid, 0, -1):
    img2 = cv2.imread(imagePaths[index-1])
    img1 = cv2.imread(imagePaths[index])
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2,shape)
    
    matches = detectFeaturesAndMatch(img2, img1)

    H, subsetMatches = getBestHomographyRANSAC(matches, trials = trials, threshold = threshold)
    prevH = np.dot(prevH, H)
    transformImage(img2, prevH)
    # print('Warped image size = ',warpedImg2.shape)
    cv2.imwrite(str(imageSet) + 'output.png', transformedImage)
    # warpedImages.append(warpedImg2)
    # offsets.append(offset)

prevH = np.eye(3)
for index in range(mid + 1, len(imagePaths)):
    img1 = cv2.imread(imagePaths[index-1])
    img2 = cv2.imread(imagePaths[index])
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2,shape)
    
    matches = detectFeaturesAndMatch(img2, img1)

    H, subsetMatches = getBestHomographyRANSAC(matches, trials = trials, threshold = threshold)
    prevH = np.dot(prevH, H)
    transformImage(img2, prevH)
    # print('Warped image size = ',warpedImg2.shape)
    cv2.imwrite(str(imageSet) + 'output.png', transformedImage)
    # warpedImages.append(warpedImg2)
    # offsets.append(offset)
    

import glob
import cv2
import numpy as np
from numba import jit

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def detectFeaturesAndMatch(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
    cv2.imshow('im', img3)
    correspondences = []
    for match in matches:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    return np.array(correspondences)

@jit
def getHomography(matches):
    
    A = np.zeros((2*len(matches), 8))
    for i, match in enumerate(matches):
        src = match[0]
        dst = match[1]
        A[2*i] = np.array([src[0], src[1], 1, 0, 0, 0, -src[0]*dst[0], -dst[0]*src[1]])
        A[2*i+1] = np.array([0, 0, 0, src[0], src[1], 1, -src[0]*dst[1], -dst[1]*src[1]])
    
    U, S, V = np.linalg.svd(A)
    h = np.append(V[-1], 1)
    H = np.reshape(h, (3, 3) )
    return H

@jit
def getBestHomographyRANSAC(matches, trials = 100000, threshold = 1000):
    finalH = None
    nMaxInliers = 0
    for trialIndex in range(trials):
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

        if trialIndex % int(trials/10) == 0:
            print(trialIndex, 'samples tested', nMaxInliers, 'inliers')
    print(finalH)
    return finalH
            

    


imageSet = 4

imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
for i in range(1, len(imagePaths)):
    print('Comparing', imagePaths[i], 'and', imagePaths[i-1])
    img1 = cv2.imread(imagePaths[i])
    img2 = cv2.imread(imagePaths[i-1])
   
    matches = detectFeaturesAndMatch(img1, img2)
    
    getBestHomographyRANSAC(matches)

    if cv2.waitKey(0) == ord('q'):
        break

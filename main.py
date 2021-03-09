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
    # cv2.imshow('im', img3)
    correspondences = []
    for match in matches:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    return np.array(correspondences)

def getHomography(matches):
    
    A = np.zeros((2*len(matches), 9))
    for i, match in enumerate(matches):
        src = match[0]
        dst = match[1]
        x1 = src[0]
        x1dash = dst[0]
        y1 = src[1]
        y1dash = dst[1]
        A[2*i] = np.array([x1, y1, 1, 0, 0, 0, -x1*x1dash, -y1*x1dash, -x1dash])
        A[2*i+1] = np.array([0, 0, 0, x1, y1, 1, -x1*y1dash, -y1*y1dash, -y1dash])
    
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3) )
    return H

def getBestHomographyRANSAC(matches, trials = 10000, threshold = 10):
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

def transformPoint(i, j, H):
    point = np.array([i, j, 1])
    transformed = np.dot(H, point)
    transformed /= transformed[2]
    transformed = transformed.astype(np.int)[:2]
    return np.array(transformed)

def transformImage(img, H):
    h, w, _ = img.shape
    topLeft = transformPoint(0, 0, H)
    topRight = transformPoint(w-1, 0, H)
    bottomLeft = transformPoint(0, h-1, H)
    bottomRight = transformPoint(w-1, h-1, H)

    box = np.array([topLeft, topRight, bottomLeft, bottomRight])
    offsets = np.array([np.min(box[:, 0]), np.min(box[:, 1])])
    
    offsetBox = box - offsets
    width = np.max(offsetBox[:, 0])
    height = np.max(offsetBox[:, 1])
    transformedImage = np.zeros((height, width, 3))
    Hinv = np.linalg.inv(H)
    for i in range(width):
        for j in range(height):
            
            transformed = transformPoint(i + offsets[0], j + offsets[1], Hinv)
            try:
                if transformed[0] > 0 and transformed[1] > 0:
                    transformedImage[j, i] = img[transformed[1], transformed[0]]
            except:
                continue
            
    
    return transformedImage, offsets


    



for i in range(1, len(imagePaths)):
    print('Comparing', imagePaths[i], 'and', imagePaths[i-1])
    img1 = cv2.imread(imagePaths[i])
    img2 = cv2.imread(imagePaths[i-1])
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.resize(img2, (600, 400))
    w, h, _ = img1.shape
    matches = detectFeaturesAndMatch(img1, img2)
    
    
    H = getBestHomographyRANSAC(matches, trials = 500, threshold = 50)
    stitched, offset = transformImage(img1, H)
    cv2.imwrite('outputs/l' + str(imageSet) + '/output.png', stitched)


    exit()
    # if cv2.waitKey(0) == ord('q'):
    #     break


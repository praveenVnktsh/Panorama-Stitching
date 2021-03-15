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
    for trialIndex in tqdm(range(trials)):
        inliers = []
        randomSample = matches[np.random.choice(len(matches), size=10, replace=False)]
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
    return finalH, randomSample

def transformPoint(i, j, H):
    transformed = np.dot(H, np.array([i, j, 1]))
    transformed /= transformed[2]
    transformed = transformed.astype(np.int)[:2]
    return np.array(transformed)

def transformImageForward(img, H):
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
    for i in range(w):
        for j in range(h):
            
            transformed = transformPoint(i,j, H)
            transformedImage[j, i] = img[transformed[1] - offsets[0], transformed[0] - offsets[1]]
            try:
                if transformed[0] > 0 and transformed[1] > 0:
                    transformedImage[j, i] = img[transformed[1], transformed[0]]
            except:
                continue
            
    return transformedImage, offsets

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



def blendImages(images, offsets):
    shape = np.array([0, 0, 3])
    maxHeight = 0
    for im in images:
        shape[1] += im.shape[1]
        maxHeight = max(maxHeight, im.shape[0])
    
    shape[0] = maxHeight*2
    print(offsets)
    print('Image shape final = ', shape)
    stitched = np.zeros(shape, dtype=np.uint8)
    cumulativeOffset = np.array([0, maxHeight//4 ])
    for index, image in enumerate(images):
        h, w , _ = image.shape
        image = image.astype(np.uint8)
        if index != 0:
            cumulativeOffset += offsets[index] - offsets[index - 1]
        

        indices = (image[:, :, 1] != 0)
        indices = np.logical_and(indices, image[:, :, 0] != 0)
        indices = np.logical_and(indices, image[:, :, 2] != 0)
        
        temp = stitched[cumulativeOffset[1]: h + cumulativeOffset[1], cumulativeOffset[0]: w + cumulativeOffset[0]]
        print(index, temp.shape, image.shape, indices.shape, cumulativeOffset)
        temp[indices] = image[indices]
        stitched[cumulativeOffset[1]: h + cumulativeOffset[1], cumulativeOffset[0]: w + cumulativeOffset[0]] = temp
        cv2.imwrite('outputs/l' + str(imageSet) + '/stitched.png', stitched)
        

    

    return stitched

    
imageSet = 4
imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
os.makedirs('outputs/l' + str(imageSet) + '/', exist_ok = True)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

shape = (300, 200)

offsets = [(0, 0)]
prevH = np.eye(3)

mid = int(len(imagePaths)/2)
nextImage = mid + 1

def execute(img1, img2):
    global prevH
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2,shape)
    
    matches = detectFeaturesAndMatch(img2, img1)

    H, subsetMatches = getBestHomographyRANSAC(matches, trials = 10000, threshold = 5)
    prevH = np.dot(prevH, H)
    warpedImg2, offset = transformImageForward(img2, prevH)
    print('Warped image size = ',warpedImg2.shape)

    warpedImages.append(warpedImg2)
    offsets.append(offset)
        

warpedImages = [cv2.resize(cv2.imread(imagePaths[mid]), shape)]


for index in range(1, len(imagePaths)):
    img1 = cv2.imread(imagePaths[index-1])
    img2 = cv2.imread(imagePaths[index])
    execute(img1, img2)
    
stitched = blendImages(warpedImages, offsets)
cv2.imwrite('outputs/l' + str(imageSet) + '/stitched.png', stitched)
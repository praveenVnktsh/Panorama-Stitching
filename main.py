from blending import Blender
import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

def detectFeaturesAndMatch(img1, img2, nFeaturesReturn = 30):
    '''
    takes in two images, and returns a set of correspondences between the two images matched using ORB features, sorted from best to worst match using an L2 norm distance.
    '''
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    correspondences = []
    for match in matches:
        correspondences.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    src = np.float32([ m[0] for m in correspondences[:nFeaturesReturn] ]).reshape(-1,1,2)
    dst = np.float32([ m[1] for m in correspondences[:nFeaturesReturn] ]).reshape(-1,1,2)
    # tempImg = cv2.drawMatches(img2, kp1, img2, kp2, matches, None, flags = 2)
    # cv2.imwrite('matches.png', tempImg)
    # exit()
    return np.array(correspondences[:nFeaturesReturn]), src, dst

def getHomography(matches):
    '''
    Takes in the points of correspondences and returns the homography matrix by 
    solving for the best fit transform using SVD.
    '''
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

def getBestHomographyRANSAC(matches, trials = 10000, threshold = 10, toChooseSamples = 4):
    '''
    Applies the RANSAC algorithm and tries out different homography matrices to compute the best matrix.
    '''
    finalH = None
    nMaxInliers = 0
    randomSample = None
    for trialIndex in tqdm(range(trials)):
        inliers = []
        # randomly sample from the correspondences, and then compute homography matrix
        # after finding homography, see if the number of inliers is the best so far. If yes, we take that homography.
        # the number of correspondences for which we can compute the homography is a parameter.
        randomSample = matches[np.random.choice(len(matches), size=toChooseSamples, replace=False)] 
        H = getHomography(randomSample)
        for match in matches:
            src = np.append(match[0], 1).T
            dst = np.append(match[1], 1).T
            transformed = np.dot(H, src)
            transformed /= transformed[2]
            if np.linalg.norm(transformed - dst) < threshold:
                inliers.append(match)

        # best match => store 
        if len(inliers) > nMaxInliers:
            nMaxInliers = len(inliers)
            finalH = H
    print('Max inliers = ', nMaxInliers)
    return finalH, randomSample

def transformPoint(i, j, H):
    '''
    Helper function that simply transforms the point according to a given homography matrix
    '''
    transformed = np.dot(H, np.array([i, j, 1]))
    transformed /= transformed[2]
    transformed = transformed.astype(np.int)[:2]
    return np.array(transformed)

def transformImage(img, H, dst, forward = False, offset = [0, 0]):
    '''
    Helper function that computes the transformed image after applying homography.
    '''
    h, w, _ = img.shape
    if forward:
        # direct conversion from image to warped image without gap filling.
        coords = np.indices((w, h)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)    
        transformedPoints = np.dot(H, coords)
        yo, xo = coords[1, :], coords[0, :]
        # projective transform. Output's 3rd index should be one to convert to cartesian coords.
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)
        dst[yt + offset[1], xt + offset[0]] = img[yo, xo]
    else:
        # applies inverse sampling to prevent any aliasing and hole effects in output image.
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
        # instead of iterating through the pixels, we take indices and do
        # H.C, where C = coordinates to get the transformed pixels.
        coords = np.indices((maxX - minX, maxY - minY)).reshape(2, -1)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int)   

        coords[0, :] += minX
        coords[1, :] += minY
        # here, we use the inverse transformation from the transformed bounding box to compute the pixel value of the transformed image.
        transformedPoints = np.dot(Hinv, coords)
        yo, xo = coords[1, :], coords[0, :]

        # projective transform. Output's 3rd index should be one to convert to cartesian coords.
        yt = np.divide(np.array(transformedPoints[1, :]),np.array(transformedPoints[2, :])).astype(np.int)
        xt = np.divide(np.array(transformedPoints[0, :]),np.array(transformedPoints[2, :])).astype(np.int)


        # to prevent out of range errors
        indices = np.where((yt >= 0) & (yt < h) & (xt >= 0) & (xt < w))

        xt = xt[indices]
        yt = yt[indices]

        xo = xo[indices]
        yo = yo[indices]

        # assign pixel values!
        dst[yo + offset[1], xo + offset[0]] = img[yt, xt]

def execute(index1, index2, prevH):
    '''
    Function that, for a given pair of indices, computes the best homography and saves the warped images to disk.
    '''
    warpedImage = np.zeros((4192, 8192, 3))
    img1 = cv2.imread(imagePaths[index1])
    img2 = cv2.imread(imagePaths[index2])
    print('Original image size = ', img1.shape)
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2,shape)
    matches, src, dst = detectFeaturesAndMatch(img2, img1)
    H, subsetMatches = getBestHomographyRANSAC(matches, trials = trials, threshold = threshold)
    prevH = np.dot(prevH, H)
    transformImage(img2, prevH, dst = warpedImage, offset = offset)
   
    cv2.imwrite('outputs/l' + str(imageSet) + '/custom/warped_' + str(index2) +  '.png', warpedImage)
    
    return prevH


if __name__ == "__main__":


    imageSet = 4


    imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
    os.makedirs('outputs/l' + str(imageSet) + '/custom/', exist_ok = True)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shape = (600, 400) # resize in order to improve speed and relax size constraints.
    mid = len(imagePaths)//2
    
    threshold = 2
    trials = 5000
    offset = [2300, 800]
       
    prevH = np.eye(3)
    prevH = execute(2, 1, prevH)
    prevH = execute(1, 0, prevH)

    prevH = np.eye(3)
    prevH = execute(2, 2, prevH) # this is wasteful, but gets the job done.

    prevH = np.eye(3)
    prevH = execute(2, 3, prevH)
    prevH = execute(3, 4, prevH)

    if imageSet == 1:
        prevH = execute(4, 5, prevH)

    # WARPING COMPLETE. BLENDING START
    
    b = Blender() # This blender object is written in blender.py. Its been encapsulated in a class to improve ease of use.
    finalImg =  cv2.imread('outputs/l' + str(imageSet) + '/custom/'  + 'warped_' + str(0) + '.png')
    if imageSet == 1:
        length = 6
    else:
        length = 5
    for index in range(1, length):
        print('blending', index)
        img2 = cv2.imread('outputs/l' + str(imageSet) + '/custom/' + 'warped_' + str(index) + '.png')
        # print('blending', index, 'outputs/l' + str(imageSet) + '/custom/' + 'warped_' + str(index) + '.png')
        finalImg, mask1truth, mask2truth = b.blend(finalImg, img2)
        mask1truth = mask1truth + mask2truth
        cv2.imwrite('outputs/l' + str(imageSet) + '/custom/' 'FINALBLENDED.png', finalImg)

    

    

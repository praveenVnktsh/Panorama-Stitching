from blending import Blender
import cv2
import glob
import cv2
import numpy as np
import os
from tqdm import tqdm


def detectFeaturesAndMatch(img1, img2, nFeaturesReturn=30):
    '''
    takes in two images, and returns a set of correspondences between the two images matched using ORB features, sorted from best to worst match using an L2 norm distance.
    '''
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    correspondences = []
    for match in matches:
        correspondences.append(
            (kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    print('Totally', len(correspondences), 'matches')
    src = np.float32(
        [m[0] for m in correspondences[:nFeaturesReturn]]).reshape(-1, 1, 2)
    dst = np.float32(
        [m[1] for m in correspondences[:nFeaturesReturn]]).reshape(-1, 1, 2)
    return np.array(correspondences[:nFeaturesReturn]), src, dst


def execute(index1, index2, prevH):
    '''
    Function that, for a given pair of indices, computes the best homography and saves the warped images to disk.
    '''

    warpedImage = np.zeros((3000, 3000, 3))
    img1 = cv2.imread(imagePaths[index1])
    img2 = cv2.imread(imagePaths[index2])
    print('Original image size = ', img1.shape)
    img1 = cv2.resize(img1, shape)
    img2 = cv2.resize(img2, shape)
    matches, src, dst = detectFeaturesAndMatch(img2, img1)
    H, subsetMatches = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    prevH = np.dot(prevH, H)
    warpedImage = cv2.warpPerspective(
        img2, prevH, (warpedImage.shape[1], warpedImage.shape[0]))

    cv2.imwrite('outputs/l' + str(imageSet) + '/opencv/warped_' +
                str(index2) + '.png', warpedImage)
    return prevH


if __name__ == "__main__":

    # image 6,  1500, 1500, (500, 350)
    # img 5, 1500, 3000

    # BEGIN WARPING

    imageSet = 4
    imagePaths = sorted(glob.glob('dataset/I' + str(imageSet) + '/*'))
    os.makedirs('outputs/l' + str(imageSet) + '/opencv/', exist_ok=True)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shape = (600, 400)

    threshold = 5
    trials = 10000
    offset = [1000, 500]
    offsetMatrix = np.array([[1, 0, offset[0]],
                      [0, 1, offset[1]],
                      [0, 0, 1]])
    prevH = offsetMatrix.copy()
    prevH = execute(2, 1, prevH)
    prevH = execute(1, 0, prevH)

    prevH = offsetMatrix.copy()
    prevH = execute(2, 2, prevH)  # this is wasteful, but alright :\

    prevH = offsetMatrix.copy()
    prevH = execute(2, 3, prevH)
    prevH = execute(3, 4, prevH)

    if imageSet == 1:
        prevH = execute(4, 5, prevH)


    finalImg =  cv2.imread('outputs/l' + str(imageSet) + '/opencv/'  + 'warped_' + str(0) + '.png')


    if imageSet == 1:
        length = 6
    else:
        length = 5



    # for index in range(1, length):
    #     img2 = cv2.imread('outputs/l' + str(imageSet) + '/opencv' + 'warped_' + str(index) + '.png')
    #     stitcher = cv2.createStitcher(True)
    #     finalImg = stitcher.stitch((finalImg, img2))
    #     cv2.imwrite('outputs/l' + str(imageSet) + '/opencv/' 'FINALBLENDED.png', finalImg)

    b = Blender() # This blender object is written in blender.py. Its been encapsulated in a class to improve ease of use.
    finalImg =  cv2.imread('outputs/l' + str(imageSet) + '/opencv/'  + 'warped_' + str(0) + '.png')
    if imageSet == 1:
        length = 6
    else:
        length = 5
    for index in range(1, length):
        print('blending', index)
        img2 = cv2.imread('outputs/l' + str(imageSet) + '/opencv/' + 'warped_' + str(index) + '.png')
        finalImg, mask1truth, mask2truth = b.blend(finalImg, img2)
        mask1truth = mask1truth + mask2truth
        cv2.imwrite('outputs/l' + str(imageSet) + '/opencv/' 'FINALBLENDED.png', finalImg)
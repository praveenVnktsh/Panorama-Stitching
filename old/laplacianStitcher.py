import numpy as np
import cv2

depth = 6

imageSet = 6
index = 0

img1 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index + 1) + '.png')


mask = np.zeros_like(img1, dtype = np.float32)
indices = img1 == 0 
# indices = np.logical_or(img2 == 0, indices)
mask[indices] = 1
cv2.namedWindow('mask', flags=cv2.WINDOW_GUI_NORMAL)
cv2.imshow('mask', mask)

# indices = (img2[:, :, 0] == 0) &   (img2[:, :, 1] == 0) &  (img2[:, :, 2] == 0)
# mask[indices] = 255
# mask = 255 - mask
# dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
# dist_transform /= np.max(dist_transform)
# mask = 1.0 - dist_transform
# mask = cv2.merge((mask, mask, mask))


G1 = img1.copy()
G2 = img2.copy()
GM = mask.copy()
gp1 = [G1]
gp2 = [G2]
gpM = [GM]
for i in range(depth):
    G1 = cv2.pyrDown(G1)
    G2 = cv2.pyrDown(G2)
    GM = cv2.pyrDown(GM)
    gp1.append(np.float32(G1))
    gp2.append(np.float32(G2))
    gpM.append(np.float32(GM))

# generate Laplacian Pyramids for A,B and masks
lp1  = [gp1[depth-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
lp2  = [gp2[depth-1]]
gpMr = [gpM[depth-1]]

cv2.namedWindow('1', flags=cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('2', flags=cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('3', flags=cv2.WINDOW_GUI_NORMAL)
for i in range(depth-1,0,-1):
# Laplacian: subtarct upscaled version of lower level from current level
# to get the high frequencies
    size = (gp1[i-1].shape[1], gp1[i-1].shape[0])
    L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i], dstsize = size))
    L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i], dstsize = size))
    cv2.imshow('1', L1)
    cv2.imshow('2', L2)
    cv2.waitKey(0)
    lp1.append(L1)
    lp2.append(L2)
    gpMr.append(gpM[i-1]) # also reverse the masks

# Now blend images according to mask in each level
LS = []
for l1,l2,gm in zip(lp1,lp2,gpMr):
    ls = l2 * gm + l1 * (1.0 - gm)
    cv2.imshow('3', ls)
    # cv2.waitKey(0)
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1,depth):
    size = (LS[i].shape[1], LS[i].shape[0])
    ls_ = cv2.pyrUp(ls_.astype(np.float32), dstsize = size).astype(np.float32)
    ls_ = cv2.add(ls_.astype(np.float32), LS[i].astype(np.float32))

cv2.imwrite('outputs/l' + str(imageSet) + '/' + 'blended.png', ls_)
import cv2
import numpy as np
from numpy.lib.function_base import average
from typing_extensions import final
class Blender():

    def __init__(self, depth = 6):
        self.depth = depth
        
    def getGaussianPyramid(self, img):
        pyra = [img]
        for i in range(self.depth - 1):
            down = cv2.pyrDown(pyra[i])
            pyra.append(down)
        return pyra

    def getLaplacianPyramid(self, img):
        pyra = []

        for i in range(self.depth-1):
            nextImg = cv2.pyrDown(img)
            size = (img.shape[1], img.shape[0])
            up = cv2.pyrUp(nextImg, dstsize=size)
            sub =  img.astype(float) - up.astype(float)
            pyra.append(sub)
            img = nextImg
            
        pyra.append(img)

        return pyra

    def getBlendingPyramid(self, lpa, lpb, gpm):
        pyra = []
        for i, mask in enumerate(gpm):
            maskNet = cv2.merge((mask, mask, mask))
            blended = lpa[i]*maskNet + lpb[i]*(1 - maskNet)
            pyra.append(blended)
        
        return pyra

    def reconstruct(self, lp):
        img = lp[-1]    
        for i in range(len(lp) - 2, -1, -1):
            laplacian = lp[i]
            size = laplacian.shape[:2][::-1]

            img = cv2.pyrUp(img, dstsize = size).astype(float)
            img += laplacian.astype(float)

        return img

    def getMask(self, img):
        mask = img[:, :, 0] != 0 
        mask = np.logical_and(img[:, :, 1] != 0, mask)
        mask = np.logical_and(img[:, :, 2] != 0, mask)

        maskImg  = np.zeros(img.shape[:2], dtype = float)
        maskImg[mask] = 1.0
        return maskImg, mask

    def blend(self, img1, img2):
        lp1 = self.getLaplacianPyramid(img1)
        lp2 = self.getLaplacianPyramid(img2)
        _, mask1truth = self.getMask(img1)
        _, mask2truth = self.getMask(img2)

        yi, xi = np.where(mask1truth & mask2truth)

        splitPoint = (np.min(xi) + np.max(xi))//2

        
        # finalMask = np.zeros(img1.shape[:2])
        # finalMask[mask1truth] = 1.0
        overlap = mask1truth & mask2truth
        # finalMask[overlap] = 0.0
        # finalMask[:, 0:splitPoint] = 1.0

        tempMask = np.zeros(img1.shape[:2])
        # tempMask[mask1truth] = 1.0
        # tempMask[mask2truth] = 1.0
        # tempMask[overlap] = 1

        # ret, contours,_ = cv2.findContours(tempMask.astype(np.uint8), 1, 1) 
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # rect = cv2.minAreaRect(contours[0])
        # (x,y),(w,h), a = rect 
        # box = cv2.boxPoints(rect).astype(np.int)
        # print(box)
        yb, xb = np.where(overlap)
        minx = np.min(xb)
        maxx = np.max(xb)
        miny = np.min(yb)
        maxy = np.max(yb)
        h, w = tempMask.shape
        
        # finalMask = np.zeros(img1.shape[:2])


        # finalMask = cv2.fillConvexPoly(finalMask,np.array([
        #     [
        #         [minx, miny], 
        #         [maxx, maxy], 
        #         [maxx, h], 
        #         [0, h], 
        #         [0,0],
        #         [minx, 0]
        #     ]
        #     ]), True,50)
        # print(finalMask)
        # split = finalMask == 1.0
        finalMask = np.zeros(img1.shape[:2])
        finalMask[:, :(minx + maxx)//2] = 1.0
        tempMask = finalMask.copy()
        # tempMask[mask1truth] = 0.3
        # tempMask[mask2truth] = 0.7
        
        # finalMask[np.logical_not(mask1truth)] = 0.0
        

        gpm = self.getGaussianPyramid(finalMask)

        blendPyra = self.getBlendingPyramid(lp1, lp2, gpm)

        finalImg = self.reconstruct(blendPyra)
        return finalImg, mask1truth, mask2truth




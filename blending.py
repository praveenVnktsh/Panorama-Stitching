import cv2
import numpy as np


class Blender():
    '''
    Class that performs blending operations on images using pyramids.
    '''
    def __init__(self, depth = 6):
        self.depth = depth
        
    def getGaussianPyramid(self, img):
        # For the image, downscale the image, and return an array.
        pyra = [img]
        for i in range(self.depth - 1):
            down = cv2.pyrDown(pyra[i])
            pyra.append(down)
        return pyra

    def getLaplacianPyramid(self, img):
        # for each image, downscale the image, and then subtract with the appropriate gaussian pyramid image to get the laplacian.
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
        # Blends the pyramid stages at each level according to the mask.
        # since the boundary of the mask changes at each downscaling, we need to get the pyramid for the mask as well
        pyra = []
        for i, mask in enumerate(gpm):
            maskNet = cv2.merge((mask, mask, mask))
            blended = lpa[i]*maskNet + lpb[i]*(1 - maskNet)
            pyra.append(blended)
        
        return pyra

    def reconstruct(self, lp):
        # for each stage in the laplacian pyramid, reconstruct by adding (inverse of what we did when downscaling)
        img = lp[-1]    
        for i in range(len(lp) - 2, -1, -1):
            laplacian = lp[i]
            size = laplacian.shape[:2][::-1]

            img = cv2.pyrUp(img, dstsize = size).astype(float)
            img += laplacian.astype(float)

        return img

    def getMask(self, img):
        # gets the mask of a particular image. Simply a helper function

        mask = img[:, :, 0] != 0 
        mask = np.logical_and(img[:, :, 1] != 0, mask)
        mask = np.logical_and(img[:, :, 2] != 0, mask)

        maskImg  = np.zeros(img.shape[:2], dtype = float)
        maskImg[mask] = 1.0
        return maskImg, mask

    def blend(self, img1, img2, strategy = 'STRAIGHTCUT'):
        '''
        Blends the two images by getting the pyramids and blending appropriately.
        '''

        # compupte the required pyramids
        lp1 = self.getLaplacianPyramid(img1)
        lp2 = self.getLaplacianPyramid(img2)


        # get the masks of both images.
        _, mask1truth = self.getMask(img1)
        _, mask2truth = self.getMask(img2)


        # using the overlaps of both the images, we compute the bounding boxes.
        yi, xi = np.where(mask1truth & mask2truth)
        overlap = mask1truth & mask2truth
        tempMask = np.zeros(img1.shape[:2])
        yb, xb = np.where(overlap)
        minx = np.min(xb)
        maxx = np.max(xb)
        miny = np.min(yb)
        maxy = np.max(yb)
        h, w = tempMask.shape

        finalMask = np.zeros(img1.shape[:2])
        if strategy == 'STRAIGHTCUT':
            # simple strategy if there is only left -> right panning.
            finalMask[:, :(minx + maxx)//2] = 1.0
        elif strategy == 'DIAGONAL':
            # Strategy that allows for slight variations in vertical movement also
            finalMask = cv2.fillConvexPoly(finalMask,np.array([
            [
                [minx, miny], 
                [maxx, maxy], 
                [maxx, h], 
                [0, h], 
                [0,0],
                [minx, 0]
            ]
            ]), True,50)

        

        gpm = self.getGaussianPyramid(finalMask)

        blendPyra = self.getBlendingPyramid(lp1, lp2, gpm)

        finalImg = self.reconstruct(blendPyra)

        
        return finalImg, mask1truth, mask2truth




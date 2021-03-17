import cv2
import numpy as np
import scipy.misc
from skimage.draw import line
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
        # gauss = self.getGaussianPyramid(img)
        pyra = []

        for i in range(self.depth-1):
            nextImg = cv2.pyrDown(img)
            size = (img.shape[1], img.shape[0])
            up = cv2.pyrUp(nextImg, dstsize=size)
            sub =  img.astype(float) - up.astype(float)
            # sub = cv2.subtract(img, up)
            pyra.append(sub)
            img = nextImg
            
        pyra.append(img)
        # for i in range(self.depth-1, 0, -1):
        #     size = gauss[i-1].shape[:2][::-1]
            
        #     up = cv2.pyrUp(gauss[i], dstsize = size)
        #     # sub = cv2.subtract(gauss[i-1], up)
        #     sub = gauss[i-1] -  up
        #     pyra.append(sub)
            
        return pyra

    def getBlendingPyramid(self, lpa, lpb, gpm):
        pyra = []
        # gpm.reverse()
        for i, mask in enumerate(gpm):
            maskNet = cv2.merge((mask, mask, mask))
            blended = lpa[i]*maskNet + lpb[i]*(1 - maskNet)
            pyra.append(blended)
        
        return pyra

    def reconstruct(self, lp):
        # lp.reverse()
        
        img = lp[-1]    
        for i in range(len(lp) - 2, -1, -1):
            laplacian = lp[i]
            size = laplacian.shape[:2][::-1]

            img = cv2.pyrUp(img, dstsize = size).astype(float)
            img += laplacian.astype(float)
            print(np.max(img), np.min(laplacian), np.max(laplacian))

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
        _, mask2truth  = self.getMask(img2)

        yi, xi = np.where(mask1truth & mask2truth)

        splitPoint = (np.min(xi) + np.max(xi))//2

        finalMask = np.zeros(img1.shape[:2])
        
        # finalMask[mask1truth] = 1.0
        # overlap = mask1truth & mask2truth
        # finalMask[overlap] = 0.0
        finalMask[:, 0:splitPoint] = 1.0
        # finalMask[np.logical_not(mask1truth)] = 0.0
        

        gpm = self.getGaussianPyramid(finalMask)

        blendPyra = self.getBlendingPyramid(lp1, lp2, gpm)

        finalImg = self.reconstruct(blendPyra)
        return finalImg


if __name__ == "__main__":
    # cv2.namedWindow('down', flags=cv2.WINDOW_GUI_NORMAL)
    b = Blender()
    imageSet = 6
    st = ''
    finalImg =  cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(0) + '.png')
    r = b.getLaplacianPyramid(finalImg)
    finalImg = b.reconstruct(r)
    cv2.imwrite('output.png', finalImg.astype(np.uint8))
    
    for index in range(1, 5):
        
        img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(index) + '.png')
        print(index)
        print(finalImg is None, img2 is None)
        finalImg = b.blend(finalImg, img2)
        
        # finalImg = np.clip(finalImg, 0, 255)
        cv2.imwrite('output.png', finalImg)
        
        


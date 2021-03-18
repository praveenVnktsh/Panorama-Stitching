import cv2
import numpy as np

opencv = False
imageSet = 6

if opencv:
    st = 'opencv'
else:
    st = ''





finalImg = cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(0) + '.png')
for index in range(1, 5):
    
    img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + st + 'warped_' + str(index) + '.png')
    shape = img2.shape
    mask1 = finalImg[:, :, 0] != 0 
    mask1 = np.logical_and(finalImg[:, :, 1] != 0, mask1)
    mask1 = np.logical_and(finalImg[:, :, 2] != 0, mask1)

    mask2 = img2[:, :, 0] != 0 
    mask2 = np.logical_and(img2[:, :, 1] != 0, mask2)
    mask2 = np.logical_and(img2[:, :, 2] != 0, mask2)


    maskImg1= np.zeros((shape[0], shape[1]), dtype = float)
    maskImg1[mask1] = 1.0

    maskImg2= np.zeros((shape[0], shape[1]), dtype = float)
    maskImg2[mask2] = 1.0

    Ga = finalImg.copy()
    Gb = img2.copy()
    Gma = maskImg1.copy()
    Gmb = maskImg1.copy()

    gpa = [Ga]
    gpb = [Gb]
    gpma = [Gma]
    gpmb = [Gmb]

    for i in range(6):
        Ga = cv2.pyrDown(Ga)
        Gb = cv2.pyrDown(Gb)
        Gma = cv2.pyrDown(Gma)
        Gmb = cv2.pyrDown(Gmb)
        gpa.append(Ga)
        gpb.append(Gb)
        gpma.append(Gma)
        gpmb.append(Gmb)

    lpa = [Ga]
    lpb = [Gb]
    for i in range(5, 0, -1):
        size = gpb[i-1].shape[1], gpb[i-1].shape[0]
        Gua = cv2.pyrUp(gpa[i], dstsize = size)
        La = cv2.subtract(gpa[i-1], Gua)
        lpa.append(La)

        Gub = cv2.pyrUp(gpb[i], dstsize = size)
        
        Lb = cv2.subtract(gpb[i-1], Gub)
        lpb.append(Lb)

    yi, xi = np.where(mask1 & mask2)

        
    splitPoint = (np.min(xi) + np.max(xi))//2
    LS = []
    for la, lb in zip(lpa, lpb):
        r, c, _ = la.shape
        ls = np.hstack((la[:, 0:splitPoint] , lb[:, splitPoint: ]))
        LS.append(ls)

    recon = LS[0]
    for i in range(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        recon = cv2.pyrUp(recon)
        
        recon = cv2.add(recon, LS[i-1])
    
    finalImg = recon
    cv2.imwrite('output.png', finalImg)
    break
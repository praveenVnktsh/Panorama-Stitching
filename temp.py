import cv2
import numpy as np
def blend(images, masks, n=5):
    """
    Image blending using Image Pyramids. We calculate Gaussian Pyramids using OpenCV.add()
    Once we have the Gaussian Pyramids, we take their differences to find Laplacian Pyramids
    or DOG(Difference of Gaussians). Then we add all the Laplacian Pyramids according to the
    seam/edge of the overlapping image. Finally we upscale all the Laplasian Pyramids to
    reconstruct the final image.
    images: array of all the images to be blended
    masks: array of corresponding alpha mask of the images
    n: max level of pyramids to be calculated.
    [NOTE: that image size should be a multiple of 2**n.]
    """

    assert(images[0].shape[0] % pow(2, n) ==
           0 and images[0].shape[1] % pow(2, n) == 0)

    # Defining dictionaries for various pyramids
    g_pyramids = {}
    l_pyramids = {}

    H, W, C = images[0].shape

    # Calculating pyramids for various images before hand
    for i in range(len(images)):

        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for _ in range(n):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    # Blending Pyramids
    common_mask = masks[0].copy()
    common_image = images[0].copy()
    common_pyramids = [l_pyramids[0][i].copy()
                       for i in range(len(l_pyramids[0]))]

    ls_ = None
    # We take one image, blend it with our final image, and then repeat for
    # n images
    for i in range(1, len(images)):

        # To decide which is left/right
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        # To check if the two pictures need to be blended are overlapping or not
        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max-x_min)/2 + x_min)/W

            # Finally we add the pyramids
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = np.hstack(
                    (la[:, 0:int(split*cols)], lb[:, int(split*cols):]))
                LS.append(ls)

        else:
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = la + lb
                LS.append(ls)

        # Reconstructing the image
        ls_ = LS[0]
        for j in range(1, n+1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        # Preparing the commong image for next image to be added
        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return ls_


img1 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index) + '.png')
img2 = cv2.imread('outputs/l' + str(imageSet) + '/' + 'warped_' + str(index + 1) + '.png')

depth = 6

imageSet = 6
index = 0
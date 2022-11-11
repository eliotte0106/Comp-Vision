import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental_matrix

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!
    
    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    panorama = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    imgA_cv = cv.imread(imageA)
    imgA_bw = cv.cvtColor(imgA_cv, cv.COLOR_BGR2GRAY)

    imgB_cv = cv.imread(imageB)
    imgB_bw = cv.cvtColor(imgB_cv, cv.COLOR_BGR2GRAY)

    final_W = imgB_cv.shape[1] + imgA_cv.shape[1]
    final_H = imgB_cv.shape[0]

    sift = cv.xfeatures2d.SIFT_create()

    kpA, desA = sift.detectAndCompute(imgA_bw, None)
    kpB, desB = sift.detectAndCompute(imgB_bw, None)

    match = cv.BFMatcher().knnMatch(desA, desB, k=2) 

    useful_matches = []
    for m in match:
        if m[1].distance * 0.5 > m[0].distance:         
            useful_matches.append(m)
            
    matches = np.asarray(useful_matches)

    src_list = []
    dest_list = []
    for match in matches[:,0]:
        src_list.append(kpA[match.queryIdx].pt)
        dest_list.append(kpB[match.trainIdx].pt)
    
    src = np.float32(src_list).reshape(-1, 1, 2)
    dest = np.float32(dest_list).reshape(-1, 1, 2)

    homog_M = cv.findHomography(src, dest, cv.RANSAC, 5.0)[0]

    index_p = np.mgrid[0:final_W, 0:final_H].reshape(2, -1).T
    padded = np.pad(index_p, [(0, 0), (0, 1)], constant_values=1)
    p = np.dot(np.linalg.inv(homog_M,), padded.T).T
    
    mapped_p = (p / p[:, 2].reshape(-1, 1))[:, 0:2].reshape(final_W, final_H, 2).astype(np.float32)

    panorama = cv.remap(imgA_cv, mapped_p, None, cv.INTER_CUBIC).transpose(1, 0, 2)
    panorama[0:imgB_cv.shape[0], 0:imgB_cv.shape[1]] = imgB_cv

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama

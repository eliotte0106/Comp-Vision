#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    k = X.shape[0]
    r = feature_width // 2
    D = feature_width ** 2
    fvs = np.zeros((k,D))
    for i in range (k):
        x = Y[i]
        y = X[i]
        img = image_bw[int(x-r+1):int(x+r+1),int(y-r+1):int(y+r+1)].reshape(1,D)
        normalized = img / (np.linalg.norm(img))
        fvs[i] = normalized

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs

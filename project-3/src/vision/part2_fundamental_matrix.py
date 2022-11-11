"""Fundamental matrix utilities."""
from typing import Tuple #imported
import numpy as np

                                             #(np.ndarray, np.ndarray)
def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    mean_p = np.sum(points, axis=0) / points.shape[0]
    means = np.identity(3)
    means[0:2,2] = (-1 * mean_p).T

    std = np.array(points, dtype=float)
    std[:,0] = std[:,0] - mean_p[0]
    std[:,1] = std[:,1] - mean_p[1]
    scale_fact = np.diag(np.hstack((np.reciprocal(np.std(std, axis=0)), [1])))

    T = np.dot(scale_fact, means)
    
    homog = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized = np.zeros(points.shape)
    for i, row in enumerate(homog):
        mult = np.dot(T, row)
        points_normalized[i] = mult[0:2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    F_orig=T_b.T.dot(F_norm).dot(T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F = []
    points_norm_a, T_a = normalize_points(points_a)
    points_norm_b, T_b = normalize_points(points_b)
    num = points_a.shape[0]
    ones = np.ones((num, 1))
    A = []
    for a, b in zip(points_norm_a, points_norm_b):
        u1, v1 = a
        u2, v2 = b
        A.append([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1])

    A = np.hstack([A, ones])
    _, _, V = np.linalg.svd(A)
    F_temp = V[-1]
    F_temp = F_temp.reshape(3,3)
    U, S, V = np.linalg.svd(F_temp)
    S[2] = 0
    F_norm = U.dot(np.diag(S)).dot(V)
    F=unnormalize_F(F_norm,T_a,T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F

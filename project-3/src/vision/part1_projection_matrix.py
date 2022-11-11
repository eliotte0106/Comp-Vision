import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    n, c = points_3d.shape
    temp = np.zeros((n*2,c*3 + 2))
    b = np.reshape(points_2d.flatten(),(1,n*2))
    temp[::2,0:c] = points_3d
    temp[::2,c] = np.ones((n))
    temp[1::2,c+1:2*c+1] = points_3d
    temp[1::2,2*c+1] = np.ones((n))
    temp[::2,2*(c+1):] = points_3d * (-1 * np.broadcast_to(b[:,::2].T, points_3d.shape))
    temp[1::2,2*(c+1):] = points_3d * (-1 * np.broadcast_to(b[:,1::2].T, points_3d.shape))

    M = np.ones((temp.shape[1] + 1, 1))
    M[0:temp.shape[1]] = np.linalg.lstsq(temp, b.T, rcond=None)[0]
    M = np.reshape(M, (3, 4))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    projected_points_2d = np.zeros((points_3d.shape[0], 2))
    homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    for i, row in enumerate(homog):
        mul = np.dot(P, row)
        projected_points_2d[i,0] = np.divide(mul[0],mul[2])
        projected_points_2d[i,1] = np.divide(mul[1],mul[2])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    cc = np.asarray([1, 1, 1])
    cc = -np.linalg.inv(M[:, :3]).dot(M[:, 3])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc

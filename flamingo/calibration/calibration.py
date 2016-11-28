#!/usr/bin/env python

__author__ = "Caio Stringari"
__copyright__ = "?"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Caio Stringari"
__email__ = "caio.eadistringari@uon.edu.au"
__status__ = "production"

import numpy as np
import cv2

def calibrate(frames,pattern=(7,9),method="cheesboard",return_error=False):
    '''Find camera intrinsic parameters.


    Parameters
    ----------
    frames : list, tuple
        list of frames to use in the calibration
    pattern : list, tuple, np.ndarray
        pattern for the search. Default is 7x9
    method : str
        Which method to use. Default is cheesboard.

    Returns
    -------
    K : np.ndarray
        3x3 camera matrix
    dist : np.ndarray
        1x5 distortion vector
    error : optional, np.ndarray
        1xNframes error vector

    Notes
    -----
    This function is based on OpenCV calibration
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
    The algorithm is based on [Zhang2000] and [BouguetMCT]
    Zhang. A Flexible New Technique for Camera Calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
    Y.Bouguet. MATLAB calibration tool. http://www.vision.caltech.edu/bouguetj/calib_doc/

    Examples
    --------

    '''
    # Only cheesboard method is implemented at the moment
    if method != "cheesboard":
        raise NotImplementedError

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[1],0:pattern[0]].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.0001)

    # loop over the images trying to identify the chessboard
    for k,frame in enumerate(frames):
        # read the image as 8 bit
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (pattern[0],pattern[1]),None)
        # if found, add object points, image points (after refining them)
        if ret == True:
            print (frame," pattern found")
            # refine corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # append
            objpoints.append(objp)
            imgpoints.append(corners2)
            # store the last processed images
            I = gray
            if k >= 10: break

    # calculate the calibration
    h,  w = I.shape[:2]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # calculate the error
    error = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error.append(cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2))
    error = np.array(error)

    if return_error:
        return K,dist, error
    else:
        return K,dist


if __name__ == '__main__':

    from glob import glob
    frames = sorted(glob("../../data/calibration/*.JPG"))

    K,dist = calibrate(frames,[6,8])

# --
# calibrate camera with checkerboard

import glob
import numpy as np

# image stuff
import cv2
from skimage import io



def cal_cam_checkerboard(file_path, cam_params_file, save_params=True):
  """
  Calibrates the camera with checkerboard images
  code addapted from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  --
  Params:
    file_path - Path to checker board calibration images
  --
  Returns:
    K, dist - intrinsic cam params K and distortion values dist
  """
  print("--Calibration of Camera--")

  # termination criteria
  criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

  # corner selection
  nm_corner = (7, 7)

  n = nm_corner[0]
  m = nm_corner[1]

  # numerate object points
  objp = np.zeros((n * m, 3), np.float32)
  objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)

  # Arrays to store object points and image points from all the images.
  # 3d point in real world space
  objpoints = []

  # 2d points in image plane. 
  imgpoints = []

  # calibration images
  images = glob.glob(file_path + '*.jpg')

  for fname in images:

    print("--image: ", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, nm_corner, None)

    print("found corners: ", ret)

    # If found, add object points, image points (after refining them)
    if ret == True:

      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
      imgpoints.append(corners2)

      # Draw and display the corners
      cv2.drawChessboardCorners(img, nm_corner, corners2, ret)
      
      # show image for debuging
      #io.imshow(img)

  # calibrate cam
  ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

  # save intrinsic camera params
  if save_params:

    print("..save instrinsic cam params.")

    # save to file
    np.save(cam_params_file + '_K', K)
    np.save(cam_params_file + '_dist', dist)

  return K, dist


def undistort_image(img_file, K, dist):
  """
  Test the calibrated camera by undistorting an image
  code addapted from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  saves undistorted image file
  --
  params:
    img_file - file to undistort
    K - instrinsic cam matrix
    dist - lens distortion
  """

  # read img
  img = cv2.imread(img_file)

  # shapes
  h, w = img.shape[:2]

  # get region of interest in checkerboard
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

  # undistort
  dst = cv2.undistort(img, K, dist, None, newcameramtx)

  # crop the image
  x, y, w, h = roi
  dst = dst[y:y+h, x:x+w]
  cv2.imwrite('cal_result.png', dst)


# --
# main
if __name__ == '__main__':

  # --
  # Camera Calibration

  # path to calibration files
  cal_img_file_path = './ignore/checker-board/'
  cam_params_file = 'cam_params'

  # use checkerboard calibration
  cal_cam_checkerboard(cal_img_file_path, cam_params_file)

  # read cam params from file
  K = np.load(cam_params_file + '_K.npy')
  dist = np.load(cam_params_file + '_dist.npy')

  # print intrinsic cam params
  print("--Intrinsic Camera Params--")
  print("K: ", K)
  print("distortion: ", dist)

  # test calibration
  #undistort_image('./ignore/checker-board/check4.jpg', K, dist)

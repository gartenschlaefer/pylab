# --
# Augmented Reality - Assignement 1
# 
# Camera Pose Estimation
# code addapted from https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
#

import glob
import numpy as np

# image stuff
import cv2
from skimage import io

# some other stuff
from cal_cam import cal_cam_checkerboard


def find_keypoints_checker(img, criteria, nm_corner):
  """
  Finds the keypoints in the image
  """
  # convert to grey scale
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Find the checker board corners
  ret, corners = cv2.findChessboardCorners(gray, nm_corner, None)

  # refine corners if found
  if ret == True:
    # refine corners
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

  else:
    corners2 = corners

  return (ret, corners2)


def checker_render_imgs(file_path, K, dist, nm_corner, obj_points, render_obj, criteria, show=True, render='axis'):
  """
  render single images for testing the algorithm
  """

  # run through all cal files and render axis
  for fname in glob.glob(file_path + '*.jpg'):

    print("--image: ", fname)
    img = cv2.imread(fname)

    # find keypoints in image
    ret, key_points = find_keypoints_checker(img, criteria, nm_corner)
    print("found corners: ", ret)

    if ret == True:

      # Find the rotation and translation vectors.
      retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(obj_points, key_points, K, dist)

      # project 3D points to image plane
      img_points, jac = cv2.projectPoints(render_obj, rvecs, tvecs, K, dist)

      # render image with transformedrendering object
      img = render_image(img, key_points, img_points, render)

      if show:
        io.imshow(img)
        io.show()

      cv2.imwrite(path_rendered + render + '_' + fname.split('/')[-1], img)


def get_object_points_checker(nm_corner):
  """
  Points of object
  """

  # checker-board corners
  n = nm_corner[0]
  m = nm_corner[1]

  # numerate object points
  objp = np.zeros((n * m, 3), np.float32)
  objp[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2)

  return objp


def get_3d_axis_points(ax_len=3):
  """
  3D axis
  """
  return np.float32([[ax_len, 0, 0], [0, ax_len, 0], [0, 0, -ax_len]]).reshape(-1, 3)


def get_3d_cube_points():
  """
  cube corners
  """
  return np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3] ])


def get_3d_tree():
  return np.float32([
    # x, y, z
    # botton
    [0, 0, 0], [1, 1, 0], [0.5, 1, 0], [1.5, 2, 0], [0.2, 2, 0], [0.2, 3, 0], [-0.2, 3, 0], [-0.2, 2, 0], [-1.5, 2, 0], [-0.5, 1, 0], [-1, 1, 0],
    
    # top
    [0, 0, -0.3], [1, 1, -0.3], [0.5, 1, -0.3], [1.5, 2, -0.3], [0.2, 2, -0.3], [0.2, 3, -0.3], [-0.2, 3, -0.3], [-0.2, 2, -0.3], [-1.5, 2, -0.3], [-0.5, 1, -0.3], [-1, 1, -0.3],
    
    # christmas balls
    [0.6, 1.5, -0.3], [-0.8, 1.7, -0.3], [-0.2, 0.7, -0.3]
    ])

def draw_3d_axis(img, key_points, img_points):
  """
  draw a 3d axis into the image, placement onto the first key_point
  """

  # pick location of first key point
  corner = tuple(key_points[0].ravel())

  # draw lines in image
  img = cv2.line(img, corner, tuple(img_points[0].ravel()), (255, 0, 0), 5)
  img = cv2.line(img, corner, tuple(img_points[1].ravel()), (0, 255, 0), 5)
  img = cv2.line(img, corner, tuple(img_points[2].ravel()), (0, 0, 255), 5)

  return img


def draw_cube(img, corners, imgpts):
  """
  draw a 3d cube into the image, placement onto the first key_point
  """
  imgpts = np.int32(imgpts).reshape(-1,2)

  # draw ground floor in green
  img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

  # draw pillars in blue color
  for i, j in zip(range(4), range(4, 8)):
    img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

  # draw top layer in red color
  img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

  return img


def draw_tree(img, corners, imgpts):
  """
  draw a 3d cube into the image, placement onto the first key_point
  """

  # colors
  brown = (63, 133, 205)
  dark_brown = (19, 69, 139)
  gold = (32, 165, 218)
  forest_green = (34, 139, 34)
  light_blue = (250, 206, 135)
  crimson = (60, 20, 220)

  imgpts = np.int32(imgpts).reshape(-1,2)

  # ground floor
  img = cv2.drawContours(img, [imgpts[:11]], -1, brown, -3)

  # pillars
  for i, j in zip(range(11), range(11, 22)):
    img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), brown, 2)

  # top
  img = cv2.drawContours(img, [imgpts[11:22]], -1, forest_green, 2)

  # christmas balls
  img = cv2.circle(img, tuple(imgpts[11]), 7, gold, -3)
  img = cv2.circle(img, tuple(imgpts[22]), 5, light_blue, -3)
  img = cv2.circle(img, tuple(imgpts[23]), 5, crimson, -3)
  img = cv2.circle(img, tuple(imgpts[24]), 5, (0, 0, 255), -3)

  return img



def render_image(img, key_points, img_points, render):
  """
  render image with object
  """

  # render object
  if render == 'cube':
    img = draw_cube(img, key_points, img_points)

  elif render == 'tree':
    img = draw_tree(img, key_points, img_points)

  else:
    img = draw_3d_axis(img, key_points, img_points)

  return img



# --
# main
if __name__ == '__main__':


  # --
  # path config

  # path to calibration files
  #cal_img_file_path = './ignore/checker-board/'
  cam_params_file = 'cam_params'

  # path to rendered images
  #path_rendered = './ignore/checker-board/rendered/'

  # video input file
  video_name = 'checker_video_lq.mp4'
  #video_name = 'checker_video_mq.mp4'


  # --
  # Camera Calibration

  # use checkerboard calibration
  #cal_cam_checkerboard(cal_img_file_path, cam_params_file)

  # read cam params from file
  K = np.load(cam_params_file + '_K.npy')
  dist = np.load(cam_params_file + '_dist.npy')

  print("--Intrinsic Camera Params loaded--")


  # --
  # marker settings

  # checker board corners detection
  nm_corner = (7, 7)

  # object points
  obj_points = get_object_points_checker(nm_corner)


  # --
  # render settings
  #render = 'cube'
  #render = 'axis'
  render = 'tree'

  # render selection
  if render == 'cube':
    render_obj = get_3d_cube_points()

  elif render == 'tree':
    render_obj = get_3d_tree()

  else:
    render_obj = get_3d_axis_points()

  # criteria for optimization 
  criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

  # render images for testing the algorithm
  #checker_render_imgs(cal_img_file_path, K, dist, nm_corner, obj_points, render_obj, criteria, show=True, render=render)

  
  # --
  # video params and init

  # video input
  cap = cv2.VideoCapture(video_name)

  # check if video opened
  if (cap.isOpened()== False): 
    print("video cannot open")

  # frame size
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))

  # video codec
  codec = cv2.VideoWriter_fourcc('M','J','P','G')

  # video writer object
  out = cv2.VideoWriter('checker_tracked.avi', fourcc=codec, fps=25, frameSize=(frame_width, frame_height))
  

  # --
  # read video stream

  print("--Read Video stream and render--")

  # read video frame by frame
  while(cap.isOpened()):

    # read frame
    ret, frame = cap.read()

    # end loop condition
    if ret == False:
      break
    
    # find keypoints in image
    ret, key_points = find_keypoints_checker(frame, criteria, nm_corner)

    # found keypoints
    if ret == True:

      # Find the rotation and translation vectors.
      retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(obj_points, key_points, K, dist)

      # project 3D points to image plane
      img_points, jac = cv2.projectPoints(render_obj, rvecs, tvecs, K, dist)

      # render image with transformedrendering object
      frame = render_image(frame, key_points, img_points, render)

    # write frame
    out.write(frame)

   
  # release video capture
  cap.release()
  out.release()

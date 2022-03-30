''' JLL, 2021.11.21 - 2022.3.28
by @Shane https://github.com/ShaneSmiskol
from /home/jinn/openpilot/common/transformations/camera.py
     /home/jinn/OP079C2/selfdrive/ui/paint.cc
read
  https://github.com/JinnAIGroup/B5/blob/main/bz2toh5_plot.py
  https://en.wikipedia.org/wiki/Camera_resectioning (camera calibration)
  https://towardsdatascience.com/camera-calibration-fda5beb373c3
  http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf

device : x->forward, y->right, z->down
road   : x->forward, y->left,  z->up
view   : x->right,   y->down,  z->forward
'''
import numpy as np
import common.transformations.orientation as orient

device_from_view = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]])

FULL_FRAME_SIZE = (1164, 874)
W, H, FOCAL = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1], 910.0

intrinsic_matrix = np.array([
  [FOCAL,   0.,   W/2. + 10],
  [  0.,  FOCAL,  H/2. - 58],
  [  0.,    0.,     1.]])   # W/2 = 582, H/2 = 437

view_from_device = device_from_view.T

roll, pitch, yaw, height = 0., 0., 0., 1.4

device_from_road = orient.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
view_from_road = view_from_device.dot(device_from_road)
extrinsic_matrix = np.hstack((view_from_road, [[0], [height], [0]])).flatten()

extrinsic_matrix_eigen = np.zeros((3, 4))
i = 0
while i < 4*3:
  extrinsic_matrix_eigen[int(i / 4), int(i % 4)] = extrinsic_matrix[i]
  i += 1

StartPt, PATH_DISTANCE = 4, 192

def transform_points(x, y):
  new_x = []
  new_y = []
  i = 0
  while i < PATH_DISTANCE:
    xi = x[i]
    yi = y[i]
    p_car_space = np.array([xi, yi, 0., 1.])
    Ep4 = np.matmul(extrinsic_matrix_eigen, p_car_space)
    Ep = np.array([Ep4[0], Ep4[1], Ep4[2]])
    KEp = np.dot(intrinsic_matrix, Ep)
    p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
      #print('#---  p_full_frame =', p_full_frame)
    new_x.append(p_full_frame[0])
    new_y.append(p_full_frame[1])
    i += 1
  return new_x[StartPt:], new_y[StartPt:]

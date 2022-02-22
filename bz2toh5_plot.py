"""   YPL, JLL, 2022.2.11 - 2.22
for studying
  /home/jinn/dataAll/14/global_pose
  /home/jinn/openpilot/tools/lib/bz2toh5.py
  /home/jinn/openpilot/common/transformations/orientation.py =
  /home/jinn/YPN/yp-Efficient1/lib/orientation.py
read also
  https://github.com/commaai/comma2k19/blob/master/notebooks/processed_readers.ipynb

WWW: Earth-centered, Earth-fixed coordinate system (ECEF)
     Local tangent plane coordinates (NED: North-East-Down)
     Show ECEF Coordinates - dominoc925 (ECEF to map)
     Euler angles; Conversion between quaternions and Euler angles
     Visualizing quaternions (4d numbers) with stereographic projection
     Visualizing quaternions An explorable video series
     Stereographic projection (2d, 3d, 4d)
     A Tutorial on Euler Angles and Quaternions
     Computing Euler angles from a rotation matrix
     Using Rotations to Build Aerospace Coordinate Systems

(YPN) jinn@Liu:~/YPN/yp-Efficient1$ python bz2toh5_plot.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import lib.orientation as orient

path = './14/'
frame_positions = np.load(path+'global_pose/frame_positions')
  #---  frame_positions.shape = (1200, 3) in ECEF
frame_orientations = np.load(path+'global_pose/frame_orientations')
  #---  frame_orientations.shape = (1200, 4) in Quaternions
frame_velocities = np.load(path+'global_pose/frame_velocities')
  #---  frame_velocities.shape = (1200, 3) in ECEF
  #---  frame_velocities[0, :] = [ 11.26643943 -23.23979293 -16.71986762]
s = np.sum(frame_velocities**2, axis=-1)
  #---  s.shape = (1200,) = total # of frames = len(s)
s = s**0.5  # speed (m/sec)
  #---  s = [30.76645261 30.76602006 30.7712556  ... => about 108 km/h
D = s*0.05  # Distance: FPS = 20 frames/second (Hertz (Hz)), dt = 1/20 => about 1.5 m between two frames
velocities = np.linalg.norm(frame_velocities, axis=1)
  #---  velocities = [30.76645261 30.76602006 30.7712556  ... 

for i in range(len(s)):
  ecef_from_local = orient.rot_from_quat(frame_orientations[i])
    #---  ecef_from_local.shape = (3, 3)
  local_from_ecef = ecef_from_local.T
  frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])
    #B = frame_positions[i:] - frame_positions[i]
    #---  i = 0
    #---  frame_positions[i:].shape = (1200, 3)    future trajectory for t > t_i
    #---  frame_positions[i].shape = (3,)          reference position at current time t_i
    #---  frame_positions_local.shape = (1200, 3)  relative displacements to reference position
    #---  B.shape = (1200, 3)
    #---  i = 1
    #---  frame_positions[i:].shape = (1199, 3)
    #---  frame_positions[i].shape = (3,)
    #---  B.shape = (1199, 3)

i = 0
ecef_from_local = orient.rot_from_quat(frame_orientations[i])
local_from_ecef = ecef_from_local.T
frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])

x1 = frame_positions_local[:, 0]  # North = Forward
y1 = frame_positions_local[:, 1]  # East
print('#---  x1.shape =', x1.shape)

x2 = frame_positions_local[:50, 0]
y2 = frame_positions_local[:50, 1]
print('#---  x2.shape =', x2.shape)

i = 1100
ecef_from_local = orient.rot_from_quat(frame_orientations[i])
local_from_ecef = ecef_from_local.T
frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])
x3 = frame_positions_local[:, 0]  # North
y3 = frame_positions_local[:, 1]  # East
print('#---  x3.shape =', x3.shape)

coeff = np.polyfit(x3, y3, 3)  # https://blog.finxter.com/np-polyfit/
  #---  coeff.shape = (4,)  # a = coeff[0], b = coeff[1], etc.
  #print('#---  coeff.shape =', coeff.shape)
cubic_fn = np.poly1d(coeff)  # cubic_fn = ax^3 + bx^2 + cx + d
x4 = np.array([1+i for i in range(200)])  # to North in 1, 2, ..., 10 meters
y4 = cubic_fn(x4)  # East displacements

plt.clf()
plt.subplot(221)
plt.title('All 1200 Frames')
plt.plot(y1, x1)
plt.subplot(222)
plt.title('First 50 Frames')
plt.plot(y2, x2)
plt.subplot(223)
plt.title('Last 100 Frames')
plt.plot(y3, x3)
plt.subplot(224)
plt.title('Polynomial Fit')
plt.plot(y4, x4, color = 'r',alpha = 0.5, label = 'Polynomial fit')
plt.scatter(y3, x3, s = 5, color = 'b', label = 'Data points')
plt.legend()
plt.savefig('bz2toh5_plot_path1.png')
plt.show()
'''
  print('#---  frame_positions[i:].shape =', frame_positions[i:].shape)
  print('#---  frame_positions[i].shape =', frame_positions[i].shape)
  print('#---  B.shape =', B.shape)
  print('#---  frame_positions_local.shape =', frame_positions_local.shape)

    WWW: Einstein summation: A basic introduction to NumPy's einsum
    np.einsum('ij,kj->ki', A, B) = (A*B^T)^T = (kj*ji)^T = ki
    A = local_from_ecef = 3x3 = ij, B = frame_positions[0:] - frame_positions[0] = 1200x3 = kj
    frame_positions_local = B^T*A^T = kj*ji = ki = 3x3

  if i < 2:
    B = frame_positions[i:] - frame_positions[i]
    print('#---  i =', i)
    print('#---  frame_orientations[i] =', frame_orientations[i])
    print('#---  ecef_from_local =', ecef_from_local)
    print('#---  local_from_ecef =', local_from_ecef)
    print('#---  frame_positions[i:] =', frame_positions[i:])
    print('#---  frame_positions[i] =', frame_positions[i])
    print('#---  B =', B)
    print('#---  frame_positions_local =', frame_positions_local)
    #---  i = 0
    #---  frame_orientations[i] = [ 0.44876812 -0.70592979  0.54635776  0.04199406]   a quaternion
    #---  ecef_from_local = [
     [ 3.99459395e-01 -8.09071632e-01  4.31086170e-01]
     [-7.33689247e-01 -2.00745963e-04  6.79485135e-01]
     [-5.49665609e-01 -5.87710008e-01 -5.93687346e-01]]
    #---  local_from_ecef = [
     [ 3.99459395e-01 -7.33689247e-01 -5.49665609e-01]
     [-8.09071632e-01 -2.00745963e-04 -5.87710008e-01]
     [ 4.31086170e-01  6.79485135e-01 -5.93687346e-01]]
    #---  frame_positions[i:] = [
     [-2713579.8706443  -4265885.93515873  3875452.81825503]   in efec from gps
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
     [-2713578.74502414 -4265888.25986234  3875451.14546622]
     ...
     [-2713088.73476472 -4267220.78007648  3874267.04393393]]
    #---  frame_positions[i]  =
     [-2713579.8706443  -4265885.93515873  3875452.81825503]
    #---  B = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 5.62823769e-01 -1.16182939e+00 -8.36047381e-01]
     [ 1.12562016e+00 -2.32470361e+00 -1.67278881e+00]
     ...
     [ 4.91135880e+02 -1.33484492e+03 -1.18577432e+03]]
    #---  frame_positions_local  = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53679346e+00  3.62219001e-02 -5.04695048e-02]
     [ 3.07472407e+00  7.28740617e-02 -1.01248708e-01]
     ...
     [ 1.82732957e+03  2.99795293e+02  8.69381580e+00]]
    #---  i = 1
    #---  frame_positions[i:] = [
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
     [-2713578.74502414 -4265888.25986234  3875451.14546622]
     [-2713578.18298109 -4265889.42201683  3875450.30942687]
     ...
     [-2713088.73476472 -4267220.78007648  3874267.04393393]]
    #---  frame_positions[i]  =
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
    #---  B = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53793802e+00  3.69073859e-02 -5.03681906e-02]
     [ 3.07466117e+00  7.40110251e-02 -1.00989458e-01]
     ...
     [ 1.82573081e+03  3.00129235e+02  8.98236797e+00]]
    #---  frame_positions_local.shape = (1199, 3)
    #---  frame_positions_local  = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53793802e+00  3.69073859e-02 -5.03681906e-02]
     [ 3.07466117e+00  7.40110251e-02 -1.00989458e-01]
     ...
     [ 1.82573081e+03  3.00129235e+02  8.98236797e+00]]

import numpy as np
p = np.array([
[1, 2],
[3, 4],
[5, 6]])
x = [np.sum(p[:i, 0]) for i in range(3)]
y = [np.sum(p[:i, 1]) for i in range(3)]
print(x)
print(y)

##### output
[0, 1, 4]
[0, 2, 6]

# https://ajcr.net/Basic-guide-to-einsum/
import numpy as np
A1 = np.array([1, 2])   # (2,)
A = np.array([[1, 2]])  # (1, 2)
B = np.array([[0,  1],
              [2,  3],
              [4,  5]])  # (3, 2)
print('A1 =', A1)
print('A1.shape =', A1.shape)
print('A =', A)
print('A.shape =', A.shape)
print('B =', B)
print('B.shape =', B.shape)

##### output
A1 = [1 2]
A1.shape = (2,)
A = [[1 2]]
A.shape = (1, 2)
B = [[0 1]
     [2 3]
     [4 5]]
B.shape = (3, 2)
A1 - A1 = [0 0]
A - A1  = [[0 0]]
A - A   = [[0 0]]
B - A1  = [[-1 -1]
 		       [ 1  1]
           [ 3  3]]
A.T = [[1]
       [2]]
A.T.shape = (2, 1)
B.T = [[0 2 4]
       [1 3 5]]
B.T.shape = (2, 3)

A = np.array([[1, 2, 1]])
B = np.array([[0, 1],
              [2, 3],
              [4, 5]])
C = np.einsum('ij,jk', A, B)  # = A*B
C = [[ 8 12]]
D = np.einsum('ij,jk->ki', A, B)  # = (A*B)^T
D = [[ 8]
     [12]]
A = np.array([[1, 2]])
D = np.einsum('ij,kj->ki', A, B)  # = (A*B^T)^T
D = [[ 2]
     [ 8]
     [14]]
'''

"""   YPL, JLL, 2022.2.11 - 2.16
for studying
  /home/jinn/dataAll/14/global_pose
  /home/jinn/openpilot/tools/lib/bz2toh5.py
  /home/jinn/openpilot/common/transformations/orientation.py =
  /home/jinn/YPN/yp-Efficient1/lib/orientation.py

WWW: Earth-centered, Earth-fixed coordinate system
     Local tangent plane coordinates
     Show ECEF Coordinates - dominoc925 (ecef to map)
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
frame_times = np.load(path+'global_pose/frame_times')
  #---  frame_times.shape = (1200,)
frame_positions = np.load(path+'global_pose/frame_positions')
frame_orientations = np.load(path+'global_pose/frame_orientations')
frame_velocities = np.load(path+'global_pose/frame_velocities')
  #---  frame_velocities.shape = (1200, 3)
  #---  frame_velocities[0, :] = [ 11.26643943 -23.23979293 -16.71986762]
s = np.sum(frame_velocities**2, axis=-1)
  #---  s.shape = (1200,)
  #v = np.load(path+'processed_log/CAN/speed/value')
  #---  v.shape = (2994, 1)
s = s**0.5   # speed
D = s*0.05   # FPS = 20 frames/second, dt = 1/20
theta = []
p = [[0, 0]]
for i in range(len(s)):
  ecef_from_local = orient.rot_from_quat(frame_orientations[i])
    # WWW: Conversion between quaternions and Euler angles
  local_from_ecef = ecef_from_local.T
  frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])
   # WWW: Local tangent plane coordinates
  if i < 3:
    B = frame_positions[i:] - frame_positions[i]
    print('#---  i =', i)
    print('#---  frame_orientations[i] =', frame_orientations[i])
    print('#---  ecef_from_local =', ecef_from_local)
    print('#---  local_from_ecef =', local_from_ecef)
    print('#---  frame_positions[i:] =', frame_positions[i:])
    print('#---  frame_positions[i] =', frame_positions[i])
    print('#---  B =', B)
    print('#---  frame_positions_local =', frame_positions_local)
    '''
    #---  i = 0
    #---  frame_orientations.shape = (1200, 4)
    #---  frame_orientations[i] = [ 0.44876812 -0.70592979  0.54635776  0.04199406]   # a quaternion
    #---  ecef_from_local.shape = (3, 3)
    #---  ecef_from_local = [
     [ 3.99459395e-01 -8.09071632e-01  4.31086170e-01]
     [-7.33689247e-01 -2.00745963e-04  6.79485135e-01]
     [-5.49665609e-01 -5.87710008e-01 -5.93687346e-01]]
    #---  local_from_ecef = [
     [ 3.99459395e-01 -7.33689247e-01 -5.49665609e-01]
     [-8.09071632e-01 -2.00745963e-04 -5.87710008e-01]
     [ 4.31086170e-01  6.79485135e-01 -5.93687346e-01]]
    #---  frame_positions.shape = (1200, 3)
    #---  frame_positions[i:].shape = (1200, 3)
    #---  frame_positions[i:] = [
     [-2713579.8706443  -4265885.93515873  3875452.81825503] in efec from gps
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
     [-2713578.74502414 -4265888.25986234  3875451.14546622]
     ...
     [-2713090.26068539 -4267218.50902854  3874268.44888216]
     [-2713089.49918346 -4267219.64418307  3874267.74519594]
     [-2713088.73476472 -4267220.78007648  3874267.04393393]]
    #---  frame_positions[i].shape = (3,)
    #---  frame_positions[i]  =
     [-2713579.8706443  -4265885.93515873  3875452.81825503]
    #---  B.shape = (1200, 3)
    #---  B = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 5.62823769e-01 -1.16182939e+00 -8.36047381e-01]
     [ 1.12562016e+00 -2.32470361e+00 -1.67278881e+00]
     ...
     [ 4.89609959e+02 -1.33257387e+03 -1.18436937e+03]
     [ 4.90371461e+02 -1.33370902e+03 -1.18507306e+03]
     [ 4.91135880e+02 -1.33484492e+03 -1.18577432e+03]]
    #---  frame_positions_local.shape = (1200, 3)
    #---  frame_positions_local  = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53679346e+00  3.62219001e-02 -5.04695048e-02]
     [ 3.07472407e+00  7.28740617e-02 -1.01248708e-01]
     ...
     [ 1.82428153e+03  3.00203714e+02  8.74505583e+00]
     [ 1.82580536e+03  3.00001396e+02  8.71977776e+00]
     [ 1.82732957e+03  2.99795293e+02  8.69381580e+00]]
    #---  i = 1
    #---  frame_positions[i:].shape = (1199, 3)
    #---  frame_positions[i:] = [
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
     [-2713578.74502414 -4265888.25986234  3875451.14546622]
     [-2713578.18298109 -4265889.42201683  3875450.30942687]
     ...
     [-2713090.26068539 -4267218.50902854  3874268.44888216]
     [-2713089.49918346 -4267219.64418307  3874267.74519594]
     [-2713088.73476472 -4267220.78007648  3874267.04393393]]
    #---  frame_positions[i].shape = (3,)
    #---  frame_positions[i]  =
     [-2713579.30782053 -4265887.09698812  3875451.98220765]
    #---  B.shape = (1199, 3)
    #---  B = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53793802e+00  3.69073859e-02 -5.03681906e-02]
     [ 3.07466117e+00  7.40110251e-02 -1.00989458e-01]
     ...
     [ 1.82268267e+03  3.00537101e+02  9.03232443e+00]
     [ 1.82420655e+03  3.00335061e+02  9.00768624e+00]
     [ 1.82573081e+03  3.00129235e+02  8.98236797e+00]]
    #---  frame_positions_local.shape = (1199, 3)
    #---  frame_positions_local  = [
     [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
     [ 1.53793802e+00  3.69073859e-02 -5.03681906e-02]
     [ 3.07466117e+00  7.40110251e-02 -1.00989458e-01]
     ...
     [ 1.82268267e+03  3.00537101e+02  9.03232443e+00]
     [ 1.82420655e+03  3.00335061e+02  9.00768624e+00]
     [ 1.82573081e+03  3.00129235e+02  8.98236797e+00]]
    Einstein summation: A basic introduction to NumPy's einsum
    np.einsum("ij,jk->ki", A, B) = (AB)^T = (ij*jk)^T = B^T*A^T = kj*ji = ki
    A = local_from_ecef = 3x3 = ij, B = frame_positions[0:] - frame_positions[0] = 1200x3 = jk
    frame_positions_local = B^T*A^T = kj*ji = ki = 3x3
    '''
    #print(len(v))
  if i < len(s)-1:
    vector = frame_positions_local[1]
      #---  vector.shape = (3,)
    theta.append(np.arctan(vector[1]/vector[0]))
      #---  theta[0] = 0.023565427402080504
    p.append([D[i]*np.cos(theta[-1]), D[i]*np.sin(theta[-1])])
      #print(D[i])
      #print(vector, theta[i])
p = np.array(p)
  #---  p.shape = (1200, 2)
  #print('#---  p =\n', p)
  #---  p =
  # [[0.         0.        ]
  #  [1.53789551 0.03624788]
  #  [1.53785824 0.03690547]
  #  ...
x = [np.sum(p[:i, 0]) for i in range(len(s))]
y = [np.sum(p[:i, 1]) for i in range(len(s))]
  #print('#---  x =\n', x)
  #print('#---  y =\n', y)
frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[0])
  #---  frame_positions_local.shape = (1200, 3)
  #theta = [np.arctan(i[1]/i[0]) for i in frame_positions_local]
  #print(theta)
  #x=frame_positions_local[:,0]
  #y=frame_positions_local[:,1]
plt.plot(y, x)
plt.savefig('bz2toh5_plot_path.png')
'''
  print('#---  frame_positions[i:].shape =', frame_positions[i:].shape)
  print('#---  frame_positions[i].shape =', frame_positions[i].shape)
  print('#---  B.shape =', B.shape)
  print('#---  frame_positions_local.shape =', frame_positions_local.shape)

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

import numpy as np
# https://ajcr.net/Basic-guide-to-einsum/
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

print('A1 - A1 =', A1 - A1)
print('A - A1 =', A - A1)
print('A - A =', A - A)
print('B - A1 =', B - A1)

print('A.T =', A.T)
print('A.T.shape =', A.T.shape)
print('B.T =', B.T)
print('B.T.shape =', B.T.shape)
C = np.einsum('ij,jk', A, B.T)
print('C =', C)
D = np.einsum('ij,jk->ki', A, B.T)
print('D =', D)

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
B.T = [[0 2 4]
       [1 3 5]]
A.T.shape = (2, 1)
B.T.shape = (2, 3)
C = [[ 2  8 14]]
D = [[ 2]
     [ 8]
     [14]]
'''

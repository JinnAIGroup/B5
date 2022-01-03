'''   HC, JLL, 2021.09.15, 11.19, 12.30
(YPN) jinn@Liu:~/YPN/Leon$ python simulatorB5.py ./fcamera.hevc
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.lanes_image_space import transform_points
from parserB5 import parser

PATH_IDX   = 0      # 192*2+1 = 385
LL_IDX     = 385    # 192*2+2 = 386
RL_IDX     = 771    # 192*2+2 = 386
LEAD_IDX   = 1157   # 11*5+3 = 58
LONG_X_IDX = 1215   # 100*2 = 200
LONG_V_IDX = 1415   # 100*2 = 200
LONG_A_IDX = 1615   # 100*2 = 200
DESIRE_IDX = 1815   # 8
META_IDX   = 1823   # 4
PRED_IDX   = 1827   # 32
POSE_IDX   = 1859   # 12
STATE_IDX  = 1871   # 512
OUTPUT_IDX = 2383

camerafile = sys.argv[1]
supercombo = load_model('models/modelB5.h5', compile = False)   # 1 out = (1, 2383)
'''
supercombo = load_model('models/modelB5.h5', compile = False)   # 1 out = (1, 2383)
  Error:
  /home/jinn/YPN/Leon/common/lanes_image_space.py:89: RuntimeWarning: divide by zero encountered in double_scalars
    p_image = p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
  99 :  new_x_path =  [567.5291180491137, 564.9207628132039, 569.2374128957745] parsed["path"][0] =  [0.01306065 0.0085456  0.03807979]
supercombo = load_model('models/JL11_dlc_model.h5', compile = False)   # 11 outs
  /home/jinn/YPN/Leon/common/lanes_image_space.py:89: RuntimeWarning: divide by zero encountered in double_scalars
    p_image = p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
  99 :  new_x_path =  [571.8503966928305, 571.7481973882532, 575.5411138895821] parsed["path"][0] =  [0.00196582 0.07565063 0.00489863]
supercombo = load_model('models/supercombo079.keras', compile = False)   # 12 outs
  /home/jinn/YPN/Leon/common/lanes_image_space.py:89: RuntimeWarning: divide by zero encountered in double_scalars
    p_image = p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
  99 :  new_x_path =  [699.0010597977995, 666.3408217211254, 646.5449165835355] parsed["path"][0] =  [-0.3373536 -0.3910733 -0.4009965]
'''
#print(supercombo.summary())

def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3
  W = frames.shape[2]
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
desire = np.zeros((1,8))
traffic_convection = np.zeros((1,2))
state = np.zeros((1,512))

cap = cv2.VideoCapture(camerafile)
fps = cap.get(cv2.CAP_PROP_FPS)

x_left = x_right = x_path = np.linspace(0, 192, 192)

(ret, previous_frame) = cap.read()
  #print ("#--- frome no. = ", cap.get(cv2.CAP_PROP_POS_FRAMES))
  #--- frome no. =  1.0
  #cap.release()

if not ret:
   exit()
else:
  frame_no = 1
  img_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[0] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
    #--- imgs_med_model.shape = (2, 384, 512)

fig = plt.figure('OPNet Simulator')
#plt.subplots_adjust( left=0.1, right=1.5, top=1.5, bottom=0.1, wspace=0.2, hspace=0.2)

while True:
  (ret, current_frame) = cap.read()
  if not ret:
       break
  frame_no += 1
  #print ("#--- frame_no =", frame_no)

  frame = current_frame.copy()
  img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[1] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))

  if frame_no > 0:
    plt.clf()
    plt.title("lanes and path")
    plt.xlim(0, 1200)
    plt.ylim(800, 0)

    #frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)
      #--- frame_tensors.shape = (2, 6, 128, 256)
    inputs = [np.vstack(frame_tensors[0:2])[None], desire, traffic_convection, state]

    outputs = supercombo.predict(inputs)
      #---  outputs.shape = (1, 2383)
    if len(outputs) == 1:   # 1 outputs[2383] for modelB5.h5
      o0  = outputs[:, PATH_IDX:   LL_IDX]   #---  o0.shape = (1, 385)
      o1  = outputs[:, LL_IDX:     RL_IDX]
      o2  = outputs[:, RL_IDX:     LEAD_IDX]
      o3  = outputs[:, LEAD_IDX:   LONG_X_IDX]
      o4  = outputs[:, LONG_X_IDX: LONG_V_IDX]
      o5  = outputs[:, LONG_V_IDX: LONG_A_IDX]
      o6  = outputs[:, LONG_A_IDX: DESIRE_IDX]
      o7  = outputs[:, DESIRE_IDX: META_IDX]
      o8  = outputs[:, META_IDX:   PRED_IDX]
      o9  = outputs[:, PRED_IDX:   POSE_IDX]
      o10 = outputs[:, POSE_IDX:   STATE_IDX]
      o11 = outputs[:, STATE_IDX:  OUTPUT_IDX]
      outs = [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]
    else:
      outs = outputs

    parsed = parser(outs)
      # Important to refeed the state
    state = outs[-1]
    pose = outs[-2]   # For 6 DoF Callibration
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.clf()
    ##################
    plt.subplot(221)
    plt.title("Overlay Scene")
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    plt.plot(new_x_right, new_y_right, label='transformed', color='b')

    plt.imshow(frame) # HC: Merge raw image and plot together

    ##################
    plt.subplot(222)
    plt.gca().invert_yaxis()
    plt.title("Camera View")
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])

    plt.plot(new_x_left, new_y_left, label='transformed', color='r')
    plt.plot(new_x_path, new_y_path, label='transformed', color='g')
    plt.plot(new_x_right, new_y_right, label='transformed', color='b')

    plt.legend(['left', 'path', 'right'])

    ##################
    plt.subplot(223) # Resize image
    frame = cv2.resize(frame, (640, 420))
    plt.title("Original Scene")
    plt.imshow(frame)

    ##################
    plt.subplot(224)
    plt.gca().invert_xaxis()
    plt.title("Top-Down View")
      # From main.py
      # lll = left lane line
    plt.plot(parsed["lll"][0], range(0,192), "r-", linewidth=1)
      # path = path cool isn't it ?
    plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
      # rll = right lane line
    plt.plot(parsed["rll"][0], range(0, 192), "b-", linewidth=1)

    #plt.legend(['lll', 'rll', 'path'])
      # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis

    if frame_no < 100:
        print(frame_no,': ','new_x_path = ', new_x_path[:3], 'parsed["path"][0] = ', parsed["path"][0][:3]) # check update. 2021.12.06

    plt.pause(0.001)
    if cv2.waitKey(10) & 0xFF == ord('q'):
          break

  imgs_med_model[0] = imgs_med_model[1]

print ("#--- Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

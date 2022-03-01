'''  JLL, 2021.8.14 - 12.24
from Leon's /home/jinn/YPN/Leon/main2.py
main.py runs h5 or keras models on video data:
8.14 supercomboLeon.keras, 12.6 supercombo079.keras

1. mv the video /dataA/.../fcamera.hevc to /Leon/fcamera.hevc
2. Read output.txt
3. Project: Build our own Net to replace supercombo.h5. Solved on 21.11.25 by Tom.

(YPN) jinn@Liu:~/YPN/Leon$ python main.py ./fcamera.hevc
'''
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser

camerafile = sys.argv[1]
supercombo = load_model('models/I1O12_model.h5', compile=False)
'''
supercombo = load_model('models/modelB4b1.h5', compile=False)
  ValueError: not enough values to unpack (expected 12, got 1)
supercombo = load_model('models/modelB4b12.h5', compile=False)
  # OK
supercombo = load_model('models/modelB4b12_untrained.h5')   # compile=Truth
  # OK
supercombo = load_model('models/modelB4b12.h5')   # compile=Truth
  ValueError: Unknown loss function: custom_loss
supercombo = load_model('models/modelB4b.h5', compile= False)
  ValueError: not enough values to unpack (expected 12, got 1)
supercombo = load_model('models/modelB4a_untrained.h5', compile=False)
supercombo = load_model('models/modelB4a_untrained.h5')
  ValueError: not enough values to unpack (expected 12, got 1)

supercombo = load_model('models/I1O12_model.h5', compile=False)
  # OK
supercombo = load_model('models/JL11_dlc_model.h5', compile=False)
  File "/home/jinn/YPN/Leon/common/tools/lib/parser.py", line 35
  # OK

supercombo = load_model('models/supercomboLeon.keras')
  ValueError: Input 0 of layer rnn_rr is incompatible with the layer: expected axis -1 of input shape to have value 512 but received input with shape [None, 2]

supercombo = load_model('models/supercombo079.keras')
  File "/home/jinn/YPN/Leon/common/tools/lib/parser.py", line 37
  # OK in loading and running
    #---  inputs[ 0 ].shape = (1, 12, 128, 256)
    #---  inputs[ 1 ].shape = (1, 8)
    #---  inputs[ 2 ].shape = (1, 2)
    #---  inputs[ 3 ].shape = (1, 512)
    #---  outs[ 0 ].shape = (1, 385)
    #---  outs[ 1 ].shape = (1, 386)
    #---  outs[ 2 ].shape = (1, 386)
    #---  outs[ 3 ].shape = (1, 58)
    #---  outs[ 4 ].shape = (1, 200)
    #---  outs[ 5 ].shape = (1, 200)
    #---  outs[ 6 ].shape = (1, 200)
    #---  outs[ 7 ].shape = (1, 8)
    #---  outs[ 8 ].shape = (1, 512)
    #---  outs[ 9 ].shape = (1, 4)
    #---  outs[ 10 ].shape = (1, 32)
    #---  outs[ 11 ].shape = (1, 12)
'''
cap = cv2.VideoCapture(camerafile)
'''
RGB (874, 1164, 3) = (H, W, C) => big YUV = (1311, 1164) = (X, Y) =>
small YUV = (384, 512) = (X, Y) => scYUV = (6, 128, 256) = (c, x, y)
'''
bYUVs = []
#for i in tqdm(range(1000)):
for i in tqdm(range(10)):
  ret, frame = cap.read()
    #---  ret =  True
    #---  frame.shape = (874, 1164, 3) = (H, W, C) = (row, col, dep) = (axis0, axis1, axis2)
    # Total float numbers (TFNs) = 874*1164*3
  bYUV = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
  bYUVs.append(bYUV.reshape((874*3//2, 1164))) # 874*3//2 = 1311
    #x = bYUV.reshape((874*3//2, 1164))
    #---  x.shape = (1311, 1164)
    #---  np.shape(bYUVs) = (20, 1311, 1164)

sYUVs = np.zeros((len(bYUVs), 384, 512), dtype=np.uint8) # np.uint8 = 0~255
  #---  sYUVs.shape = (20, 384, 512)

for i, img in tqdm(enumerate(bYUVs)):
  sYUVs[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                           yuv=True, output_size=(512, 256))  # (W, H)
'''
input: YUV img (1311, 1164) => transform_img() => cv2.COLOR_YUV2RGB_I420
  => RGB (874, 1164, 3) => cv2.warpPerspective) => np.clip
  => RGB (256, 512, 3) => cv2.COLOR_RGB2YUV_I420 =>
output: YUV sYUVs[0].shape = (384, 512)  # 256*3//2 = 384
RGB, YUV444: 3 bytes per pixel; YUV420: 6 bytes per 4 pixels [Wiki YUV]
RGB (874, 1164, 3) = 874*1164*3 bytes => YUV (1311, 1164) = 1311*1164 = 874*3//2*1164 bytes
'''

def sYUVs_to_scYUVs(sYUVs):
  #---  np.shape(sYUVs) = (20, 384, 512) = (B, X, Y)  YUV420
  H = (sYUVs.shape[1]*2)//3  # 384x2//3 = 256
  W = sYUVs.shape[2]         # 512
  scYUVs = np.zeros((sYUVs.shape[0], 6, H//2, W//2), dtype=np.uint8)

  scYUVs[:, 0] = sYUVs[:, 0:H:2, 0::2]  # [2::2] get every even starting at 2
  scYUVs[:, 1] = sYUVs[:, 1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
  scYUVs[:, 2] = sYUVs[:, 0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
  scYUVs[:, 3] = sYUVs[:, 1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
  scYUVs[:, 4] = sYUVs[:, H:H+H//4].reshape((-1, H//2, W//2))
  scYUVs[:, 5] = sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))

  return scYUVs
'''
np.shape(scYUVs) = (20, 6, 128, 256) = (B, c, x, y) YUV420 => c = 6 ???
RGB (256, 512, 3) = (H, W, C)     = 256*512*3   bytes =>
YUV (384, 512) = (X, Y) = 384*512 = 256*3/2*512 bytes = 128*512*3
= 128*256*6 = x*y*c => c = 6 QED
2y = Y = W, x*y*6 = x*W*3 = H*W*C/2 = H*W*3/2 => x = H/2, X = H*3/2
[Wiki YUV] => c = 6 = 4Y + 1U + 1V => total Ys = total pixels = 256x512 =>
256x512 (Y) + 256x512/4 (U) + 256x512/4 (V) =  256*512*3/2 bytes
Eg: YUV (6, 6) = (X, Y) => W = Y = 6, H = X*2/3 = 4, x = 2, y = 3 => YUV (c, x, y) = (6, 2, 3)
    RGB (H, W, C) = (4, 6, 3) => H = 4, W = 6; fr[6, 6], scYUVs.shape = (6, 2, 3)
    scYUVs[0] = fr[0:H:2, 0::2] = fr[i, j]; i = 0, 2; j = 0, 2, 4;  shape = (2, 3)  6u
    scYUVs[1] = fr[1:H:2, 0::2] = fr[i, j]; i = 1, 3; j = 0, 2, 4;  shape = (2, 3)  6y1
    scYUVs[2] = fr[0:H:2, 1::2] = fr[i, j]; i = 0, 2; j = 1, 3, 5;  shape = (2, 3)  6y2
    scYUVs[3] = fr[1:H:2, 1::2] = fr[i, j]; i = 1, 3; j = 1, 3, 5;  shape = (2, 3)  6v
    scYUVs[4] = fr[H:H+H//4].reshape((-1, H//2, W//2))      = [1, j].reshape(2, 3)  6y3
    scYUVs[5] = fr[H+H//4:H+H//2].reshape((-1, H//2, W//2)) = [1, j].reshape(2, 3)  6y4
    H:H+H//4 = 4:5, H+H//4:H+H//2 = 5:6
  #---  sYUVs[:, H:H+H//4].shape      = (10, 64, 512)  # H:H+H//4      = 256:256+64     = 256:320
  #---  sYUVs[:, H+H//4:H+H//2].shape = (10, 64, 512)  # H+H//4:H+H//2 = 256+64:256+128 = 320:384
  #---  sYUVs[:, H:H+H//4].reshape((-1, H//2, W//2)).shape      = (10, 128, 256)
  #---  sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2, W//2)).shape = (10, 128, 256)
  #---  np.shape(scYUVs[:, 5]) = (10, 128, 256)
  #---  np.shape(scYUVs[:, 4]) = (10, 128, 256)
[Wiki Bayer filter]
'''

scYUVs = sYUVs_to_scYUVs(np.array(sYUVs)).astype(np.float32)/128.0 - 1.0
  #---  np.shape(np.array(sYUVs)) =  (20, 384, 512)
  #---  np.shape(scYUVs) = (20, 6, 128, 256)
  # RGB: [0, 255], YUV: [0, 255]/128 - 1 = (-1, 1)

desire = np.zeros((1,8))
traffic_convection = np.zeros((1,2))
state = np.zeros((1,512))

print("#---  Input: camerafile = ", camerafile)
print("#---  Input: np.shape(scYUVs) = ", np.shape(scYUVs))

for i in tqdm(range(len(scYUVs) - 1)):
    #inputs = [np.vstack(scYUVs[i:i+2])[None], desire, state]   # for supercomboLeon.keras
  inputs = [np.vstack(scYUVs[i:i+2])[None], desire, traffic_convection, state]   # for supercombo079.keras
  if i==0:
    [print("#---  inputs[", i, "].shape =", np.shape(inputs[i])) for i in range(len(inputs))]
      #---  inputs[ 0 ].shape = (1, 12, 128, 256)
      #---  inputs[ 1 ].shape = (1, 8)
      #---  inputs[ 2 ].shape = (1, 2)
      #---  inputs[ 3 ].shape = (1, 512)
  outs = supercombo.predict(inputs)
  if i==0:
    [print("#---  outs[", i, "].shape =", np.shape(outs[i])) for i in range(len(outs))]
      #---  outs[ 0 ].shape = (1, 385)
      #---  outs[ 1 ].shape = (1, 386)
      #---  outs[ 2 ].shape = (1, 386)
      #---  outs[ 3 ].shape = (1, 58)
      #---  outs[ 4 ].shape = (1, 200)
      #---  outs[ 5 ].shape = (1, 200)
      #---  outs[ 6 ].shape = (1, 200)
      #---  outs[ 7 ].shape = (1, 8)
      #---  outs[ 8 ].shape = (1, 4)
      #---  outs[ 9 ].shape = (1, 32)
      #---  outs[ 10 ].shape = (1, 12)
      #---  outs[ 11 ].shape = (1, 512)
  parsed = parser(outs)
    #---  parsed[ path ].shape = (1, 192)
    #---  parsed[ path_stds ].shape = (1, 192)
    #---  parsed[ lll ].shape = (1, 192)
    #---  parsed[ lll_prob ].shape = (1,)
    #---  parsed[ lll_stds ].shape = (1, 192)
    #---  parsed[ rll ].shape = (1, 192)
    #---  parsed[ rll_prob ].shape = (1,)
    #---  parsed[ rll_stds ].shape = (1, 192)
    #---  parsed[ lead_xyva ].shape = (1, 4)
    #---  parsed[ lead_xyva_std ].shape = (1, 4)
    #---  parsed[ lead_prob ].shape = (1,)
    #---  parsed[ lead_xyva_2s ].shape = (1, 4)
    #---  parsed[ lead_xyva_std_2s ].shape = (1, 4)
    #---  parsed[ lead_prob_2s ].shape = (1,)
    #---  parsed[ lead_all ].shape = (1, 58)
    #---  parsed[ meta ].shape = (1, 32)
    #---  parsed[ desire ].shape = (1, 12)
    #---  parsed[ desire_state ].shape = (1, 4)
    #---  parsed[ long_x ].shape = (1, 200)
    #---  parsed[ long_v ].shape = (1, 200)
    #---  parsed[ long_a ].shape = (1, 200)
    #---  parsed[ trans ].shape = (1, 3)
    #---  parsed[ trans_std ].shape = (1, 3)
    #---  parsed[ rot ].shape = (1, 3)
    #---  parsed[ rot_std ].shape = (1, 3)
    # Important to refeed the state
  state = outs[-1]
    #---  np.shape(state) = (1, 512)
  pose = outs[-2]
    #print(np.array(pose[0,:3]).shape)
    #---  np.shape(pose) = (1, 12)
  ret, frame = cap.read()
    #---  frame.shape = (874, 1164, 3) = (H, W, C)
  frame = cv2.resize(frame, (640, 420))
    #---  frame.shape = (420, 640, 3) = (H, W, C)

    # Show raw camera image
  cv2.imshow("modeld", frame)
    # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")
    # lll = left lane line
  plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
    # rll = right lane line
  plt.plot(parsed["rll"][0], range(0,192), "r-", linewidth=1)
    # path = path cool isn't it ?
  plt.plot(parsed["path"][0], range(0,192), "g-", linewidth=1)
    #plt.scatter(pose[0,:3], range(3), c="y")

    # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

input("Press ENTER twice to close all windows ...")
input("Press ENTER to exit ...")
  # pauses for 1 second
if cv2.waitKey(1000) == 27:   #if ENTER is pressed
  cap.release()
  cv2.destroyAllWindows()
plt.pause(0.5)
plt.close()
  #plt.show()

'''
OPNet/camera.py:
FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

  if i==0:
    #print("#---  inputs = ", inputs)
    print("#---  scYUVs[i:i+2].shape =", np.shape(scYUVs[i:i+2]))
    print("#---  np.vstack(scYUVs[i:i+2]).shape =", np.shape(np.vstack(scYUVs[i:i+2])))
    print("#---  np.vstack(scYUVs[i:i+2])[None].shape =", np.shape(np.vstack(scYUVs[i:i+2])[None]))
    [print("#---  inputs[", i, "].shape =", np.shape(inputs[i])) for i in range(len(inputs))]
    #print("#---  outs =", outs)
    [print("#---  outs[", i, "].shape =", np.shape(outs[i])) for i in range(len(outs))]
    #print("#---  parsed =", parsed)
    [print("#---  parsed[", x, "].shape =", parsed[x].shape) for x in parsed]
    print("#---  np.shape(state) =", np.shape(state))
    print("#---  np.shape(pose) =", np.shape(pose))
    print("#---  frame.shape =", frame.shape)
'''

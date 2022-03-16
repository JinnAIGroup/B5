"""   YPL, JLL, 2021.9.15 - 2022.3.1
Input:
  /home/jinn/dataB/UHD--2018-l08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5
Output:
  Ximgs.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Xin1 = (none, 8)
  Xin2 = (none, 2)
  Xin3 = (none, 512)
  Ytrue0 = outs[ 0 ].shape = (none, 385)
  Ytrue1 = outs[ 1 ].shape = (none, 386)
  Ytrue2 = outs[ 2 ].shape = (none, 386)
  Ytrue3 = outs[ 3 ].shape = (none, 58)
  Ytrue4 = outs[ 4 ].shape = (none, 200)
  Ytrue5 = outs[ 5 ].shape = (none, 200)
  Ytrue6 = outs[ 6 ].shape = (none, 200)
  Ytrue7 = outs[ 7 ].shape = (none, 8)
  Ytrue8 = outs[ 8 ].shape = (none, 4)
  Ytrue9 = outs[ 9 ].shape = (none, 32)
  Ytrue10 = outs[ 10 ].shape = (none, 12)
  Ytrue11 = outs[ 11 ].shape = (none, 512)
"""
import os
import cv2
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

def datagen(batch_size, camera_files):
  Ximgs  = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # Is YUV input img uint8? No. See float8 ysf = convert_float8(ys) in loadyuv.cl
  Xin1   = np.zeros((batch_size, 8), dtype='float32')     # desire shape
  Xin2   = np.zeros((batch_size, 2), dtype='float32')     # traffic convection
  Xin3   = np.zeros((batch_size, 512), dtype='float32')   # rnn state
  Ytrue0 = np.zeros((batch_size, 385), dtype='float32')
  Ytrue1 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
  Ytrue3 = np.zeros((batch_size, 58), dtype='float32')
  Ytrue4 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
  Ytrue7 = np.zeros((batch_size, 8), dtype='float32')
  Ytrue8 = np.zeros((batch_size, 4), dtype='float32')
  Ytrue9 = np.zeros((batch_size, 32), dtype='float32')
  Ytrue10 = np.zeros((batch_size, 12), dtype='float32')
  Ytrue11 = np.zeros((batch_size, 512), dtype='float32')

  Xin2[:, 0] = 1.0   # traffic convection = 1.0 = left hand drive like in Taiwan

  path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
  radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
  for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
    if os.path.isfile(cfile) and os.path.isfile(pfile) and os.path.isfile(rfile):
      print('#---datagenB5  OK: cfile, pfile, rfile exist')
    else:
      print('#---datagenB5  Error: cfile, pfile, or rfile does not exist')
  Nplot = 0

  batchIndx = 0
  while True:
    for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
      print('#---datagenB5  cfile =', cfile)
      with h5py.File(cfile, "r") as cf5:
        pf5 = h5py.File(pfile, 'r')
        rf5 = h5py.File(rfile, 'r')
        cf5X = cf5['X']
        pf5P = pf5['Path']
        rf5L = rf5['LeadOne']
          #---  cf5X.shape = (1150, 6, 128, 256)
          #---  pf5P.shape = (1149, 51)
          #---  rf5L.shape = (1149, 5)

        PWPath    = 192 - pf5P.shape[1] + 1   # PW = pad_width; + 1: ignore "51" = valid_len, i.e., pad it to zero.
          #---  pf5P.shape[1] = 51
        PWLLane   = 386
        PWRLane   = 386
        PWLead    = 29 - rf5L.shape[1]
        PWLongX   = 200
        PWLongV   = 200
        PWLongA   = 200
        PWDesireS = 8
        PWMeta    = 4
        PWDesireP = 32
        PWPose    = 12
        PWState   = 512

        dataN = min(len(cf5X), len(pf5P), len(rf5L))
        ranIdx = list(range(dataN-2-batch_size))   # cannot be the last dataN-1
        random.seed(0)
        random.shuffle(ranIdx)

        for i in range(0, len(ranIdx), batch_size):
          count = 0
          while count < batch_size:
            print("#---  i, count, dataN, count+ranIdx[i] =", i, count, dataN, count+ranIdx[i])
            vsX1 = cf5X[count+ranIdx[i]]
            vsX2 = cf5X[count+ranIdx[i]+1]
              #---  vsX2.shape = (6, 128, 256)
            Ximgs[count] = np.vstack((vsX1, vsX2))   # stack two yuv images i and i+1
              #---  Ximgs[count].shape = (12, 128, 256)

            pf5P1 = pf5P[count+ranIdx[i]]
            pf5P2 = pf5P[count+ranIdx[i]+1]
            rf5L1 = rf5L[count+ranIdx[i]]
            rf5L2 = rf5L[count+ranIdx[i]+1]
              #---  pf5P1.shape = (51,)
              #---  pf5P2.shape = (51,)
              #---  rf5L2.shape = (5,)
            pf5P1 = np.pad(pf5P1[:50], (0, PWPath), 'constant')   # pad PWPath=142 zeros ('constant') to the right of pf5P1[:50]=50; (0, PWPath) = (left, right)
              #---  pf5P1.shape = (192,)
            pf5P2 = np.pad(pf5P2[:50], (0, PWPath+1), 'constant')   # pad PWPath+1=143 zeros to the right of pf5P2[:50]=50
              #---  pf5P2.shape = (193,)
            rf5L1 = np.pad(rf5L1, (0, PWLead), 'constant')
            rf5L2 = np.pad(rf5L2, (0, PWLead), 'constant')
            Y0 = np.hstack((pf5P1, pf5P2))
            Y1 = []
            Y1 = np.pad(Y1, (0, PWLLane), 'constant')
            Y2 = []
            Y2 = np.pad(Y2, (0, PWRLane), 'constant')
            Y3 = np.hstack((rf5L1, rf5L2))
            Y4 = []
            Y4 = np.pad(Y4, (0, PWLongX), 'constant')
            Y5 = []
            Y5 = np.pad(Y5, (0, PWLongV), 'constant')
            Y6 = []
            Y6 = np.pad(Y6, (0, PWLongA), 'constant')
            Y7 = []
            Y7 = np.pad(Y7, (0, PWDesireS), 'constant')
            Y8 = []
            Y8 = np.pad(Y8, (0, PWMeta), 'constant')
            Y9 = []
            Y9 = np.pad(Y9, (0, PWDesireP), 'constant')
            Y10 = []
            Y10 = np.pad(Y10, (0, PWPose), 'constant')
            Y11 = []
            Y11 = np.pad(Y11, (0, PWState), 'constant')
            Ytrue0[count] = Y0
            Ytrue1[count] = Y1
            Ytrue2[count] = Y2
            Ytrue3[count] = Y3
            Ytrue4[count] = Y4
            Ytrue5[count] = Y5
            Ytrue6[count] = Y6
            Ytrue7[count] = Y7
            Ytrue8[count] = Y8
            Ytrue9[count] = Y9
            Ytrue10[count] = Y10
            Ytrue11[count] = Y11
            count += 1
          batchIndx += 1
          print('#---  batchIndx =', batchIndx)
            #---  batchIndx = 181??? (dataN-2-batch_size = 1129, batch_size = 16, 1129÷16 = 70.5, 70×2 epochs = 140)
          '''if Nplot == 0:
            Yb = Ytrue0[0][:]
              #---  Yb.shape = (385,)
              #print('#---  Yb =', Yb)
            print('#---datagenB5  Ytrue0[0][50:51]   =', Ytrue0[0][50:51])    # valid_len = 50
            print('#---datagenB5  Ytrue0[0][242:243] =', Ytrue0[0][242:243])  # valid_len = 50+192 = 242
            print('#---datagenB5  Ytrue3[0][0:1]     =', Ytrue3[0][0:1])      # lcar's d = 385+386+386 = 1157
            print('#---datagenB5  Ytrue3[0][29:30]   =', Ytrue3[0][29:30])    # lcar's d = 1157+29 = 1186
            plt.plot(Yb)
            plt.show()'''
            Nplot += 1
          yield Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11

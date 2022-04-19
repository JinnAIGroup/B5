"""   YJW, YPL, JLL, 2021.9.15 - 2022.4.18
from /home/jinn/YPN/B5/datagenB5.py
Input:
  /models/supercombo079.keras
  /home/jinn/dataB/UHD--2018-l08-02--08-34-47--32/yuv.h5
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
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#physical_gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_gpus[0], True)

def supercombo_y(Ximgs, Xin1, Xin2, Xin3):
   supercombo = load_model('models/supercombo079.keras', compile=False)
   Ximgs = np.expand_dims(Ximgs, axis=0)
     #print(print("#---  np.shape(Ximgs) =", np.shape(Ximgs))
   Xin1 = np.expand_dims(Xin1, axis=0)
   Xin2 = np.expand_dims(Xin2, axis=0)
   Xin3 = np.expand_dims(Xin3, axis=0)
   inputs = [Ximgs, Xin1, Xin2, Xin3]

   outputs = supercombo(inputs)
     #[print("#---  outputs[", i, "].shape =", np.shape(outputs[i])) for i in range(len(outputs))]
   return outputs

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

  Xin1[:, 0] = 1.0   # go straight?
  Xin2[:, 0] = 1.0   # traffic convection = 1.0 = left hand drive like in Taiwan

  for cfile in camera_files:
    if os.path.isfile(cfile):
      print('#---datagenB5  OK: cfile exist')
    else:
      print('#---datagenB5  Error: cfile, pfile, or rfile does not exist')

  batchIndx = 0

  while True:
    for cfile in camera_files:
      print('#---datagenB5  cfile =', cfile)
      with h5py.File(cfile, "r") as cf5:
        cf5X = cf5['X']
          #---  cf5X.shape = (1150, 6, 128, 256)
        dataN = len(cf5X)
        ranIdx = list(range(dataN-2-batch_size))   # cannot be the last dataN-1

        camra_rnn = 0  # for 1st image in vedio using Xin3 = np.zeros((batch_size, 512)
        for i in range(0, len(ranIdx), batch_size):
          count = 0
          while count < batch_size:
            print("#---  i, count, dataN, count+ranIdx[i] =", i, count, dataN, count+ranIdx[i])
            vsX1 = cf5X[count+ranIdx[i]]
            vsX2 = cf5X[count+ranIdx[i]+1]
              #---  vsX2.shape = (6, 128, 256)
            Ximgs[count] = np.vstack((vsX1, vsX2))   # stack two yuv images i and i+1
              #---  Ximgs[count].shape = (12, 128, 256)

              # How to train rnn? See: Understanding Simple Recurrent Neural Networks In Keras
            if camra_rnn != 0:
              Xin3[count] = Xin3_temp
                #---  np.shape(Xin3_temp) = (512,)

            outs = supercombo_y(Ximgs[count], Xin1[count], Xin2[count], Xin3[count])
            Xin3_temp = outs[11][0]
              #print("#---  np.shape(outs[11]) =", np.shape(outs[11]))
              #---  np.shape(outs[11][0]) = (512,)
              #---  np.shape(outs[11]) = (1, 512)
            camra_rnn = 1
              #np.savetxt('rnn_state.txt', outs[11][0], delimiter=',')

            Ytrue0[count] = outs[0]
            Ytrue1[count] = outs[1]
            Ytrue2[count] = outs[2]
            Ytrue3[count] = outs[3]
            Ytrue4[count] = outs[4]
            Ytrue5[count] = outs[5]
            Ytrue6[count] = outs[6]
            Ytrue7[count] = outs[7]
            Ytrue8[count] = outs[8]
            Ytrue9[count] = outs[9]
            Ytrue10[count] = outs[10]
            Ytrue11[count] = outs[11]
            count += 1
          batchIndx += 1
          print('#---  batchIndx =', batchIndx)

          yield Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11

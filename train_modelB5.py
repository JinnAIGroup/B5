"""   YPL, JLL, 2021.9.8 - 2022.2.8
modelB5.dlc = supercombo079.dlc

1. Tasks: temporal state (features) + path planning (PP)
   The temporal state (a 512 array) represents (are features of) all path planning details:
   path prediction, lane detection, lead car xyva, desire etc., i.e., outs[0]...[10] in output.txt.
2. Ground Truth from pathdata.h5, radardata.h5:
   Pose: 56 = Ego path 51 (x, y) + 5 radar lead car's dRel (relative distance), yRel, vRel (velocity), aRel (acceleration), prob
3. Loss: mean squared error (mse)

Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5, pathdata.h5, radardata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5, pathdata.h5, radardata.h5
Output:
  /B5/saved_model/modelB5_lossB.npy

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/B5$ python serverB5.py --port 5557
   (YPN) jinn@Liu:~/YPN/B5$ python serverB5.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/B5$ python train_modelB5.py --port 5557 --port_val 5558

Training History:
  BATCH_SIZE = 16  EPOCHS = 2
  2022.2.3 (B5)   out1: no normalization
    #---  max of y_true, max index = 81.58935 (array([15]), array([242]))
    #---  max of y_pred, max index = 81.73105 (array([10]), array([242]))
    Epoch 00002: val_loss improved from 1.41168 to 1.40772, saving model to ./saved_model/modelB5-BestWeights.hdf5
    71/71 [==============================] - 183s 3s/step -
    loss: 2.5557 - rmse: 1.4936 - mae: 0.1834 - val_loss: 1.4077 - val_rmse: 1.1694 - val_mae: 0.1720
  2022.2.2 (B5)   out2: normalize (/=10) and un-normalize (*=10)
    #---  max of y_true, max index = 138.52 (array([8, 9]), array([1186, 1157]))
    #---  max of y_pred, max index = 76.50992 (array([ 1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    Epoch 00002: val_loss improved from 1.37270 to 1.33507, saving model to ./saved_model/modelB5-BestWeights.hdf5
    71/71 [==============================] - 190s 3s/step -
    loss: 2.5874 - rmse: 1.5068 - mae: 0.1995 - val_loss: 1.3351 - val_rmse: 1.1434 - val_mae: 0.1813
    Training Time: 00:06:20.49
    See ./saved_model/modelB5_outs.png

Road Tests:
  2022.2.8 (modelB5C2C.dlc):
    1. WANRNING: This branch is not tested
    2. Lead car's red chevron always exists (radar error?)
    3. My car quickly turned and stopped.
"""
import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from modelB5 import get_model
from serverB5 import client_generator, BATCH_SIZE

EPOCHS = 2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def get_data(hwm, host, port, model):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11 = tup

    Xins  = [Ximgs, Xin1, Xin2, Xin3]   #  (imgs, traffic_convection, desire, rnn_state)
    Ytrue = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))
      #---  Xins[0].shape = (16, 12, 128, 256)
      #---  Ytrue.shape = (16, 2383)
      # we need the following two lines, otherwise we always get #---  y_true.shape[0] = None
    p = model.predict(x=Xins)
    loss = custom_loss(Ytrue, p)
      #print('#---  loss =', loss)
    yield Xins, Ytrue

def custom_loss(y_true, y_pred):
    #---  y_true.shape = (None, None)
    #---  y_pred.shape = (None, 2383)
    #---  y_true.shape[0] = None
    #---  y_true.shape = (16, 2383)
    #---  y_pred.shape = (16, 2383)
    #---  y_true.shape[0] = 16
  if y_true.shape[0] is not None:
    print('\n#---  max of y_true, max index =', np.amax(y_true), np.where(y_true == np.amax(y_true)))
    print('#---  max of y_pred, max index =', np.amax(y_pred), np.where(y_pred == np.amax(y_pred)))
    y_true[:, 50:51]/=10      # normalize valid_len = 50
    y_true[:, 242:243]/=10    # valid_len = 50+192 = 242
    y_true[:, 1157:1158]/=10  # lcar's d = 385+386+386 = 1157
    y_true[:, 1186:1187]/=10  # lcar's d = 1157+29 = 1186
    y_pred[:, 50:51]/=10
    y_pred[:, 242:243]/=10
    y_pred[:, 1157:1158]/=10
    y_pred[:, 1186:1187]/=10
    ''''''
  loss = tf.keras.losses.mse(y_true, y_pred)

  if y_true.shape[0] is not None:
    y_true[:, 50:51]*=10   # un-normalize
    y_true[:, 242:243]*=10
    y_true[:, 1157:1158]*=10
    y_true[:, 1186:1187]*=10
    y_pred[:, 50:51]*=10
    y_pred[:, 242:243]*=10
    y_pred[:, 1157:1158]*=10
    y_pred[:, 1186:1187]*=10
    ''''''
  return loss

if __name__=="__main__":
  start = time.time()
  parser = argparse.ArgumentParser(description='Training modelB5')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--port_val', type=int, default=5558, help='Port of server for validation dataset.')
  args = parser.parse_args()

    # Build model
  img_shape = (12, 128, 256)
  desire_shape = (8)
  traffic_convection_shape = (2)
  rnn_state_shape = (512)
  num_classes = 6
  model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    #model.summary()

  filepath = "./saved_model/modelB5-BestWeights.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  #model.load_weights('./saved_model/modelB5-BestWeights.hdf5', by_name=True)

  from tensorflow.keras import backend
  def rmse(y_true, y_pred):
  	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
  adam = tf.keras.optimizers.Adam(lr=0.0001)
  model.compile(optimizer=adam, loss=custom_loss, metrics=[rmse, 'mae'])   # metrics=custom_loss???

  history = model.fit(
    get_data(20, args.host, port=args.port, model=model),
    steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
    validation_data=get_data(20, args.host, port=args.port_val, model=model),
    validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
      # steps_per_epoch = total images//BATCH_SIZE

  model.save('./saved_model/modelB5.h5')

  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

  plt.subplot(311)
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  train_loss = np.array(history.history['loss'])
  np.savetxt("./saved_model/train_loss_out2.txt", train_loss, delimiter=",")
  valid_loss = np.array(history.history['val_loss'])
  np.savetxt("./saved_model/valid_loss_out2.txt", valid_loss, delimiter=",")
  plt.ylabel("loss")
  plt.legend(['train', 'validate'], loc='upper right')

  plt.subplot(312)
  plt.plot(history.history["rmse"])
  plt.plot(history.history["val_rmse"])
  train_rmse = np.array(history.history['rmse'])
  np.savetxt("./saved_model/train_rmse_out2.txt", train_rmse, delimiter=",")
  valid_rmse = np.array(history.history['val_rmse'])
  np.savetxt("./saved_model/valid_rmse_out2.txt", valid_rmse, delimiter=",")
  plt.ylabel("rmse")
  plt.legend(['train', 'validate'], loc='upper right')

  plt.subplot(313)
  plt.plot(history.history["mae"])
  plt.plot(history.history["val_mae"])
  train_mae = np.array(history.history['mae'])
  np.savetxt("./saved_model/train_mae_out2.txt", train_mae, delimiter=",")
  valid_mae = np.array(history.history['val_mae'])
  np.savetxt("./saved_model/valid_mae_out2.txt", valid_mae, delimiter=",")
  plt.ylabel("mae")
  plt.xlabel("epoch")
  plt.legend(['train', 'validate'], loc='upper right')
  plt.draw()
  #plt.savefig('./saved_model/modelB5_out2.png')
  plt.pause(0.5)
  input("Press ENTER to close ...")
  plt.close()

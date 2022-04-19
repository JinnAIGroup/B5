"""   YJW, YPL, JLL, 2021.9.8 - 2022.4.19
from /home/jinn/YPN/B5/train_modelB5.py
modelB5.dlc = supercombo079.dlc

1. use the output of supercombo079.keras as ground truth data to train modelB5
2. verify trained modelB5.h5 by simulatorB5.py
3. Tasks: path planning (PP) + temporal state (RNN)
   y_true[2383] = (Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11)
   y_pred[2383] = outs[0] + ... + outs[10] in sim_output.txt.
4. Ground Truth from supercombo079.keras:
   outputs = supercombo(inputs) in datagenB5YJ.py
5. Loss: mean squared error (mse) or Huber loss (better)

Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5
Output:
  /home/jinn/YPN/B5/saved_model/modelB5YJ.h5

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/B5$ python3 serverB5YJ.py --port 5557
   (YPN) jinn@Liu:~/YPN/B5$ python3 serverB5YJ.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/B5$ python3 train_modelB5YJ.py --port 5557 --port_val 5558

Training History:
  BATCH_SIZE = 2  EPOCHS = 2
  1. 4/11: try w/o normalization
     5/5 [==============================] - 39s 8s/step -
     loss: 138.6146 - rmse: 11.6598 - mae: 5.8961 - val_loss: 112.0994 - val_rmse: 10.3778 - val_mae: 5.6667
     Training Time: 00:01:32.09
     $ python simulatorB5.py => NG
  2. 4/11: try w/ normalization
     5/5 [==============================] - 40s 8s/step -
     loss: 0.0160 - rmse: 12.5430 - mae: 6.3640 - val_loss: 0.0127 - val_rmse: 11.0533 - val_mae: 6.0215
     Training Time: 00:01:31.91
     $ python simulatorB5.py => NG
  Conclusion: w/o or w/ normalization makes no difference
  BATCH_SIZE, STEPS, EPOCHS = 1, 1148, 2; w/o normalization
  3. 4/18: try the first 50 few vector
     YTpath = y_true[:, 50]  # try the first 50 few vector
     camra_rnn = 0  # for 1st image in vedio using Xin3 = np.zeros((batch_size, 512)
     for i in range(0, len(ranIdx), batch_size):
     Epoch 1/2
     #---  i, count, dataN, count+ranIdx[i] = 256 0 1150 256   Killed
     256/1148 [=====>........................] - ETA: 48:06 -
     loss: 14.0165 - rmse: 2.8667 - mae: 1.4380
     $ python simulatorB5.py => NG
  4. 4/19: try the whole 2383 vector
     loss = tf.keras.losses.mse(y_true, y_pred)  # for whole 2383 vector
     Epoch 1/2
     #---  i, count, dataN, count+ranIdx[i] = 256 0 1150 256   Killed
     260/1148 [=====>........................] - ETA: 44:56 -
     loss: 13.8437 - rmse: 2.8483 - mae: 1.4290
     $ python simulatorB5.py => NG but better
  Conclusion:
     A: important to move camra_rnn = 0 up
     B: 1, 1148, 2 is needed
     C: 2383 is better than 50
  5. 4/19: try Huber loss with delta=1.0
     #---  i, count, dataN, count+ranIdx[i] = 253 0 1150 253   Killed
     253/1148 [=====>........................] - ETA: 44:39 -
     loss: 0.8905 - rmse: 2.9524 - mae: 1.1404
       delta=2.0
     #---  i, count, dataN, count+ranIdx[i] = 255 0 1150 255   Killed
     255/1148 [=====>........................] - ETA: 49:05 - loss: 1.4773 - rmse: 2.8519 - mae: 1.1421
  Conclusion: Huber loss with delta=1.0 is the best.
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
from serverB5YJ import client_generator, BATCH_SIZE, STEPS, EPOCHS

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    #---  y_true.shape = (16, 2383)
    #---  y_pred.shape = (16, 2383)
    #print("#---  y_true =", np.shape(y_true))

    #loss = tf.keras.losses.mse(y_true, y_pred)  # for whole 2383 vector
  loss = tf.keras.losses.huber(y_true, y_pred, delta=1.0)  # for whole 2383 vector
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

  filepath = "./saved_model/modelB5-BestWeightsYJ.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min')
  callbacks_list = [checkpoint]

  model.load_weights('./saved_model/modelB5-BestWeightsYJ.hdf5')

  from tensorflow.keras import backend
  def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
  adam = tf.keras.optimizers.Adam(lr=0.0001,decay=0.00000001)
  model.compile(optimizer=adam, loss=custom_loss, metrics=[rmse, 'mae'])   # metrics=custom_loss???

  history = model.fit(
    get_data(20, args.host, port=args.port, model=model),
    steps_per_epoch=STEPS, epochs=EPOCHS,
      #steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
      #steps_per_epoch=87998//BATCH_SIZE, epochs=EPOCHS,
    validation_data=get_data(20, args.host, port=args.port_val, model=model),
    validation_steps=STEPS, verbose=1, callbacks=callbacks_list)
      #validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
      #validation_steps=44399//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
      # steps_per_epoch = total images//BATCH_SIZE

  model.save('./saved_model/modelB5YJ.h5')

  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

  plt.subplot(311)
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  train_loss = np.array(history.history['loss'])
  np.savetxt("./saved_model/train_loss_out2YJ.txt", train_loss, delimiter=",")
  valid_loss = np.array(history.history['val_loss'])
  np.savetxt("./saved_model/valid_loss_out2YJ.txt", valid_loss, delimiter=",")
  plt.ylabel("loss")
  plt.legend(['train', 'validate'], loc='upper right')

  plt.subplot(312)
  plt.plot(history.history["rmse"])
  plt.plot(history.history["val_rmse"])
  train_rmse = np.array(history.history['rmse'])
  np.savetxt("./saved_model/train_rmse_out2YJ.txt", train_rmse, delimiter=",")
  valid_rmse = np.array(history.history['val_rmse'])
  np.savetxt("./saved_model/valid_rmse_out2YJ.txt", valid_rmse, delimiter=",")
  plt.ylabel("rmse")
  plt.legend(['train', 'validate'], loc='upper right')

  plt.subplot(313)
  plt.plot(history.history["mae"])
  plt.plot(history.history["val_mae"])
  train_mae = np.array(history.history['mae'])
  np.savetxt("./saved_model/train_mae_out2YJ.txt", train_mae, delimiter=",")
  valid_mae = np.array(history.history['val_mae'])
  np.savetxt("./saved_model/valid_mae_out2YJ.txt", valid_mae, delimiter=",")
  plt.ylabel("mae")
  plt.xlabel("epoch")
  plt.legend(['train', 'validate'], loc='upper right')
  plt.draw()
  #plt.savefig('./saved_model/modelB5_out2.png')
  plt.pause(0.5)
  input("Press ENTER to close ...")
  plt.close()

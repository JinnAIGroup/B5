"""   YJW, YPL, JLL, 2021.9.8 - 2022.4.6
from /home/jinn/YPN/B5/train_modelB5.py
1. use the output of supercombo079.keras as ground truth data to train modelB5
2. verify trained modelB5.h5 by simulatorB5.py
modelB5.dlc = supercombo079.dlc

1. Tasks: temporal state (features) + path planning (PP)
   y_true[2383] = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))
   y_pred[2383] = outs[0] + ... + outs[10] in output.txt.
2. Ground Truth from supercombo079.keras:
   outputs = supercombo(inputs) in datagenB5YJ.py
3. Loss: mean squared error (mse)

Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5
Output:
  /B5/saved_model/modelB5_lossBYJ.npy

Run: on 3 terminals
   (YPN) jinn@Liu:~/YPN/B5$ python3 serverB5YJ.py --port 5557
   (YPN) jinn@Liu:~/YPN/B5$ python3 serverB5YJ.py --port 5558 --validation
   (YPN) jinn@Liu:~/YPN/B5$ python3 train_modelB5YJ.py --port 5557 --port_val 5558

Training History:
  BATCH_SIZE = 4  EPOCHS = 2
  w/ normalization
    5/5 [==============================] - 38s 8s/step -
    loss: 198.4872 - rmse: 13.9900 - mae: 6.7617 - val_loss: 175.9664 - val_rmse: 13.0847 - val_mae: 6.6732
    Training Time: 00:02:49.32
  BATCH_SIZE = 2  EPOCHS = 2
  w/o normalization
    5/5 [==============================] - 40s 8s/step -
    loss: 0.0160 - rmse: 12.5430 - mae: 6.3640 - val_loss: 0.0127 - val_rmse: 11.0533 - val_mae: 6.0215
    Training Time: 00:01:31.91
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
from serverB5YJ import client_generator, BATCH_SIZE, EPOCHS

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
    #print("y_true = ",np.shape(y_true))
    #print(",y_pred = ",np.shape(y_pred))
  if y_true.shape[0] is not None:
    Ymax = np.max(y_true)  # How to use the NumPy max function
    Ymin = np.min(y_true)
    print('\n#---  max of y_true, max index =', Ymax, np.where(y_true == np.max(y_true)))
    print('#---  min of y_true, min index =', Ymin, np.where(y_true == np.min(y_true)))
    print('#---  max of y_pred, max index =', np.max(y_pred), np.where(y_pred == np.max(y_pred)))
    print('#---  min of y_pred, min index =', np.min(y_pred), np.where(y_pred == np.min(y_pred)))
      #---  max of y_true, max index = 266.34155 (array([0]), array([769]))    # at y_true(0, 769); 769-385=384=outs[1][384] in sim_output.txt
      #---  min of y_true, min index = -139.88947 (array([0]), array([1168]))  # MDN_GROUP_SIZE*LEAD_MDN_N+SELECTION=11*5+3=58
      # 1168-385-386-386=1168-1157=11=outs[3][11] (1st of 11, 2nd of 5); -139.889=data[1*11] in driving079.cc; why -???
      #loss = tf.keras.losses.mse((y_true-Ymin)/(Ymax-Ymin), (y_pred-Ymin)/(Ymax-Ymin))  # normalization
    loss = tf.keras.losses.mse(y_true/Ymax, y_pred/Ymax)  # normalization yields better convergence
  else:
    Ymax = 100.
    loss = tf.keras.losses.mse(y_true/Ymax, y_pred/Ymax)
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
      #steps_per_epoch=87998//BATCH_SIZE, epochs=EPOCHS,
    steps_per_epoch=5, epochs=EPOCHS,
    validation_data=get_data(20, args.host, port=args.port_val, model=model),
      #validation_steps=44399//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
    validation_steps=5, verbose=1, callbacks=callbacks_list)
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

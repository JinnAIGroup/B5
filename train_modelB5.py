"""   JLL, SLT, 2021.12.30
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
  71/71 [==============================] - 133s 2s/step -
  loss: 0.0168 - custom_loss: 0.0168 - val_loss: 0.0075 - val_custom_loss: 0.0075
  Training Time: 00:04:32.62

Road Test Result:
  My car quickly turned and stopped. 2021.12.30
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

def gen(hwm, host, port, model):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11 = tup

        Xins  = [Ximgs, Xin1, Xin2, Xin3]   #  (imgs, traffic_convection, desire, rnn_state)
        Ytrue = np.hstack((Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11))

        yield Xins, Ytrue

def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.mse(y_true, y_pred)

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
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=custom_loss, metrics=custom_loss)

    history = model.fit(
        gen(20, args.host, port=args.port, model=model),
        steps_per_epoch=1150//BATCH_SIZE, epochs=EPOCHS,
        validation_data=gen(20, args.host, port=args.port_val, model=model),
        validation_steps=1150//BATCH_SIZE, verbose=1, callbacks=callbacks_list)
          # steps_per_epoch = total images//BATCH_SIZE

    model.save('./saved_model/modelB5.h5')

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    np.save('./saved_model/modelB5_loss', np.array(history.history['loss']))
    lossnpy = np.load('./saved_model/modelB5_loss.npy')
    plt.plot(lossnpy)  #--- lossnpy.shape = (10,)
    plt.draw()
    plt.pause(0.5)
    input("Press ENTER to exit ...")
    plt.close()

'''    YPL, JLL, YJW, 2021.9.17 - 2022.2.8
--- Step 1: from h5 to pb
(YPN) jinn@Liu:~/YPN/B5$ python h5topbB5.py modelB5.pb
----- Frozen pb model inputs needed for converting pb to dlc:
[<tf.Tensor 'Input:0' shape=(None, 12, 128, 256) dtype=float32>,
 <tf.Tensor 'Input_1:0' shape=(None, 8) dtype=float32>,
 <tf.Tensor 'Input_2:0' shape=(None, 2) dtype=float32>,
 <tf.Tensor 'Input_3:0' shape=(None, 512) dtype=float32>]
----- Frozen pb model outputs needed for converting pb to dlc:
[<tf.Tensor 'Identity:0' shape=(None, 2383) dtype=float32>]
----- OK: pb is saved in ./saved_model

--- Step 2: move /home/jinn/YPN/B5/saved_model/modelB5.pb to /home/jinn/snpe/dlc/modelB5.pb

--- Step 3: from pb to dlc
(snpe) jinn@Liu:~/snpe$ export ANDROID_NDK_ROOT=android-ndk-r22b
(snpe) jinn@Liu:~/snpe$ source snpe-1.48.0.2554/bin/envsetup.sh -t snpe-1.48.0.2554
(snpe) jinn@Liu:~/snpe$ snpe-tensorflow-to-dlc --input_network ./dlc/modelB5.pb \
--input_dim Input "1,12,128,256" --input_dim Input_1 "1,8" --input_dim Input_2 "1,2" --input_dim Input_3 "1,512" \
--out_node "Identity" --output_path ./dlc/modelB5.dlc

--- Step 4: from dlc to html
(snpe) jinn@Liu:~/snpe/dlc$ snpe-dlc-viewer -i modelB5.dlc
'''
import os
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from modelB5 import get_model

def h5_to_pb(pb_save_name):
  try:
    img_shape = (12, 128, 256)
    desire_shape = (8)
    traffic_convection_shape = (2)
    rnn_state_shape = (512)
    num_classes = 6
    model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    model.load_weights('./saved_model/modelB5.h5')
    model.summary()
  except ValueError:
    print("----- Error: Can't covert this model to h5")
    return 1

  full_model = tf.function(lambda Input: model(Input))
  full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype)
                                                 for i in range(len(model.inputs))])

  # Get frozen Concrete Function
  frozen_func = convert_variables_to_constants_v2(full_model)
  print('----- Frozen pb model graph: ')
  print(frozen_func.graph)
  layers = [op.name for op in frozen_func.graph.get_operations()]
  print("-" * 50)
  print("----- Frozen pb model inputs needed for converting pb to dlc: ")
  print(frozen_func.inputs)
  print("----- Frozen pb model outputs needed for converting pb to dlc: ")
  print(frozen_func.outputs)

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./saved_model",
                    name=pb_save_name,
                    as_text=False)
  print("----- OK: pb is saved in ./saved_model ")

if __name__=="__main__":
   parser = argparse.ArgumentParser(description='h5 to pb')
   parser.add_argument('pb', type=str, default="transition", help='save pb name')
   args = parser.parse_args()
   h5_to_pb(args.pb)

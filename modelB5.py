'''   JLL, SLT, YJW, 2021.12.30
modelB5 = EfficientNet + RNN + PoseNet => modelB5.dlc = supercombo079.dlc (no Project A)
supercombo: https://drive.google.com/file/d/1L8sWgYKtH77K6Kr3FQMETtAWeQNyyb8R/view
output.txt: https://github.com/JinnAIGroup/OPNet/blob/main/output.txt

Input:
  2 YUV images with 6 channels = (none, 12, 128, 256)
  inputs[ 0 ].shape = (1, 12, 128, 256)   # 12 = 2 frames x 6 channels (YUV_I420: Y=4, U=1, V=1)
  inputs[ 1 ].shape = (1, 8)
  inputs[ 2 ].shape = (1, 2)
  inputs[ 3 ].shape = (1, 512)
Output:
  outs[ 0 ].shape = (1, 385)
  outs[ 1 ].shape = (1, 386)
  outs[ 2 ].shape = (1, 386)
  outs[ 3 ].shape = (1, 58)
  outs[ 4 ].shape = (1, 200)
  outs[ 5 ].shape = (1, 200)
  outs[ 6 ].shape = (1, 200)
  outs[ 7 ].shape = (1, 8)
  outs[ 8 ].shape = (1, 4)
  outs[ 9 ].shape = (1, 32)
  outs[ 10 ].shape = (1, 12)
  outs[ 11 ].shape = (1, 512)
Run:
  (YPN) jinn@Liu:~/YPN/B5$ python modelB5.py
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def stem(x0):
    x = layers.Conv2D(32, 3, strides=2, padding="same", name="stem_conv")(x0)
    x = layers.Activation("elu", name="stem_activation")(x)
    return x

def block(x0):
    count = "1"
    x = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"a_dwconv")(x0)
    print("tmp.shape=", x.shape.as_list())
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(16, 1, padding="same", name="block"+count+"a_project_conv")(x)


    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(16, 1, padding ="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp, x], name = "block"+count+"b_add")

    count = "2"
    x = layers.Conv2D(96, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3, depth_multiplier=1, strides=2, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(24, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(144, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(24, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name="block"+count+"b_add")

    tmp = layers.Conv2D(144, 1, padding="same", name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(24, 1, padding="same", name="block"+count+"c_project_conv")(tmp)
    x = layers.add([tmp,x], name="block"+count+"c_add")

    count="3"
    x = layers.Conv2D(144, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5, depth_multiplier=1,strides=2, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(48, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(288, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(48, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"b_add")

    tmp = layers.Conv2D(288, 1, padding="same", name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(48, 1, padding="same", name="block"+count+"c_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"c_add")

    count="4"
    x = layers.Conv2D(288, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3, depth_multiplier=1,strides=2, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(88, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(528, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(88, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"b_add")

    tmp = layers.Conv2D(528, 1, padding="same", name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(88, 1, padding="same", name="block"+count+"c_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"c_add")

    tmp = layers.Conv2D(528, 1, padding="same", name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(88, 1, padding="same", name="block"+count+"d_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"d_add")

    count = "5"
    x = layers.Conv2D(528, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(120, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(720, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(120, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"b_add")

    tmp = layers.Conv2D(720, 1, padding="same", name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(120, 1, padding="same", name="block"+count+"c_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"c_add")

    tmp = layers.Conv2D(720, 1, padding="same", name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(120, 1, padding="same", name="block"+count+"d_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"d_add")

    count = "6"
    x = layers.Conv2D(720, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(5, depth_multiplier=1,strides = 2, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(208, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(1248, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(208, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"b_add")

    tmp = layers.Conv2D(1248, 1, padding="same", name="block"+count+"c_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"c_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"c_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"c_activation")(tmp)
    tmp = layers.Conv2D(208, 1, padding="same", name="block"+count+"c_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"c_add")

    tmp = layers.Conv2D(1248, 1, padding="same", name="block"+count+"d_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"d_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"d_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"d_activation")(tmp)
    tmp = layers.Conv2D(208, 1, padding="same", name="block"+count+"d_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"d_add")

    tmp = layers.Conv2D(1248, 1, padding="same", name="block"+count+"e_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"e_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(5, depth_multiplier=1, padding="same", name="block"+count+"e_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"e_activation")(tmp)
    tmp = layers.Conv2D(208, 1, padding="same", name="block"+count+"e_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"e_add")

    count = "7"
    x = layers.Conv2D(1248, 1, padding="same", name="block"+count+"a_expand_conv")(x)
    x = layers.Activation("elu", name="block"+count+"a_expand_activation")(x)
    x = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"a_dwconv")(x)
    x = layers.Activation("elu", name="block"+count+"a_activation")(x)
    x = layers.Conv2D(352, 1, padding="same", name="block"+count+"a_project_conv")(x)

    tmp = layers.Conv2D(2112, 1, padding="same", name="block"+count+"b_expand_conv")(x)
    tmp = layers.Activation("elu", name="block"+count+"b_expand_activation")(tmp)
    tmp = layers.DepthwiseConv2D(3, depth_multiplier=1, padding="same", name="block"+count+"b_dwconv")(tmp)
    tmp = layers.Activation("elu", name="block"+count+"b_activation")(tmp)
    tmp = layers.Conv2D(352, 1, padding="same", name="block"+count+"b_project_conv")(tmp)
    x = layers.add([tmp,x], name = "block"+count+"b_add")
    return x

def top(x0):
    x = layers.Conv2D(1408, 1, padding="same", name="top_conv")(x0)
    x = layers.Activation("elu", name="top_activation")(x)
    return x

def RNN(x, desire, traffic_convection, rnn_state):
    desire1 = layers.Dense(use_bias=False, units=8)(desire)
    traffic_convection1 = layers.Dense(use_bias=False, units = 2)(traffic_convection)
    x_concate = layers.Concatenate(axis=-1)([desire1, traffic_convection1, x])
    x_dense = layers.Dense(use_bias=False, units=1024)(x_concate)
    x_1 = layers.Activation("relu")(x_dense)

    rnn_rz = layers.Dense(use_bias=False, units=512)(rnn_state)
    rnn_rr = layers.Dense(use_bias=False, units=512)(rnn_state)
    snpe_pleaser = layers.Dense(use_bias=False, units=512)(rnn_state)
    rnn_rh = layers.Dense(use_bias=False, units = 512)(snpe_pleaser)

    rnn_z = layers.Dense(use_bias=False, units=512)(x_1)
    rnn_h = layers.Dense(use_bias=False, units=512)(x_1)
    rnn_r = layers.Dense(use_bias=False, units=512)(x_1)

    add = layers.add([rnn_rz , rnn_z])
    activation_1 = layers.Activation("sigmoid")(add)
    add_1 = layers.add([rnn_rr , rnn_r])

    activation = layers.Activation("sigmoid")(add_1)
    multiply = rnn_rh*activation
    add_2 = layers.add([rnn_h , multiply])

    activation_2 = layers.Activation("tanh")(add_2)
    one_minus = layers.Dense(use_bias=False, units=512)(activation_1)
    multiply_2 = one_minus*activation_2
    multiply_1 = snpe_pleaser*activation_1
    out11 = layers.add([multiply_1 , multiply_2])
    return out11

def fork1(x):
    xp = layers.Dense(256, activation='relu', name="1_path")(x)
    xp = layers.Dense(256, activation='relu', name="2_path")(xp)
    xp = layers.Dense(256, activation='relu', name="3_path")(xp)
    x0 = layers.Dense(128, activation='relu', name="final_path")(xp)

    xll = layers.Dense(256, activation='relu', name="1_left_lane")(x)
    xll = layers.Dense(256, activation='relu', name="2_left_lane")(xll)
    xll = layers.Dense(256, activation='relu', name="3_left_lane")(xll)
    x1 = layers.Dense(128, activation='relu', name="final_left_lane")(xll)

    xrl = layers.Dense(256, activation='relu', name="1_right_lane")(x)
    xrl = layers.Dense(256, activation='relu', name="2_right_lane")(xrl)
    xrl = layers.Dense(256, activation='relu', name="3_right_lane")(xrl)
    x2 = layers.Dense(128, activation='relu', name="final_right_lane")(xrl)

    xl = layers.Dense(256, activation='relu', name="1_lead")(x)
    xl = layers.Dense(256, activation='relu', name="2_lead")(xl)
    xl = layers.Dense(256, activation='relu', name="3_lead")(xl)
    x3 = layers.Dense(128, activation='relu', name="final_lead")(xl)

    xlx = layers.Dense(256, activation='relu', name="1_long_x")(x)
    xlx = layers.Dense(256, activation='relu', name="2_long_x")(xlx)
    xlx = layers.Dense(256, activation='relu', name="3_long_x")(xlx)
    x4 = layers.Dense(128, activation='relu', name="final_long_x")(xlx)

    xla = layers.Dense(256, activation='relu', name="1_long_a")(x)
    xla = layers.Dense(256, activation='relu', name="2_long_a")(xla)
    xla = layers.Dense(256, activation='relu', name="3_long_a")(xla)
    x5 = layers.Dense(128, activation='relu', name="final_long_a")(xla)

    xlv = layers.Dense(256, activation='relu', name="1_long_v")(x)
    xlv = layers.Dense(256, activation='relu', name="2_long_v")(xlv)
    xlv = layers.Dense(256, activation='relu', name="3_long_v")(xlv)
    x6 = layers.Dense(128, activation='relu', name="final_long_v")(xlv)

    xds = layers.Dense(128, activation='relu', name="1_desire_state")(x)
    x7 = layers.Dense(8, name="final_desire_state")(xds)

    out0 = layers.Dense(385, name="path")(x0)
    out1 = layers.Dense(386, name="left_lane")(x1)
    out2 = layers.Dense(386, name="right_lane")(x2)
    out3 = layers.Dense(58, name="lead")(x3)
    out4 = layers.Dense(200, name="long_x")(x4)
    out5 = layers.Dense(200, name="long_a")(x5)
    out6 = layers.Dense(200, name="long_v")(x6)
    out7 = layers.Softmax(axis=-1, name="desire_state")(x7)
    return out0, out1, out2, out3, out4, out5, out6, out7

def fork2(x):
    x1 = layers.Dense(256, activation='relu', name="meta0")(x)
    out8 = layers.Dense(4, activation='sigmoid', name="meta")(x1)
    dp1 = layers.Dense(32, name="desire_final_dense")(x1)
    dp2 = layers.Reshape((4, 8), name="desire_reshape")(dp1)
    dp3 = layers.Softmax(axis=-1, name="desire_pred0")(dp2)
    out9 = layers.Flatten(name="desire_pred")(dp3)
    return out8, out9

def fork3(x):
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out10 = layers.Dense(12, name="pose")(x)
    return out10

def EffNet(x0):
    x = stem(x0)
    x = block(x)
    x = top(x)
    x = layers.Conv2D(32, 1, padding="same")(x)
    x = layers.Activation("elu")(x)
    x_to_RNNfk2fk3 = layers.Flatten()(x)
    return x_to_RNNfk2fk3

def get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes):
    imgs = keras.Input(shape=img_shape, name="pic")
      #--- imgs.shape = (None, 12, 128, 256)
    in0 = layers.Permute((2, 3, 1))(imgs)
      #--- in0.shape = (None, 128, 256, 12)
    in1 = keras.Input(shape=desire_shape, name="desire")
    in2 = keras.Input(shape=traffic_convection_shape, name="traffic_convection")
    in3 = keras.Input(shape=rnn_state_shape, name="rnn_state")
    in0_in3 = [imgs, in1, in2, in3]

    x_to_RNNfk2fk3 = EffNet(in0)
    out11 = RNN(x_to_RNNfk2fk3, in1, in2, in3)
    out0, out1, out2, out3, out4, out5, out6, out7 = fork1(out11)
    out8, out9 = fork2(x_to_RNNfk2fk3)
    out10      = fork3(x_to_RNNfk2fk3)
    outs = layers.Concatenate(axis=-1)([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11])

      # Define the model
    model = keras.Model(inputs=in0_in3, outputs=outs, name='modelB5')
    return model

if __name__=="__main__":
      # Build model
    img_shape = (12, 128, 256)
    desire_shape = (8)
    traffic_convection_shape = (2)
    rnn_state_shape = (512)
    num_classes = 6

    #model = get_model(img_shape, desire_shape, traffic_convection_shape, rnn_state_shape, num_classes)
    #model.summary()

    #model.save('./saved_model/modelB5.h5')
      # for converting to onnx
    #model = keras.models.load_model("./saved_model/modelB5.h5")
    #tf.saved_model.save(model, "./saved_model/modelB5_onnx")
      # $python3 -m tf2onnx.convert --saved-model ./saved_model/modelB5_onnx --output "./saved_model/modelB5.onnx"

    #(YPN) jinn@Liu:~/YPN/B5$ sudo pip install onnx
    import onnx
    import keras2onnx
    onnx_model_name = 'modelB5.onnx'
    model = load_model('./saved_model/modelB5.h5')
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, onnx_model_name)

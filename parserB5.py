'''  JLL, 2021.8.16 - 2022.3.2
/home/jinn/YPN/Leon/parserB5.py
from /home/jinn/YPN/Leon/common/tools/lib/parser.py
'''
import numpy as np

PATH_DISTANCE = 192   # MODEL_PATH_DISTANCE in modeldata.h, data.h
LANE_OFFSET = 1.8
LEAD_X_SCALE = 10   # x_scale in driving.cc
LEAD_Y_SCALE = 10   # y_scale in driving.cc
LEAD_V_SCALE = 1

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softplus(x):
  # fix numerical stability
  #return np.log1p(np.exp(x))
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

def softmax(x):
  x = np.copy(x)
  axis = 1 if len(x.shape) > 1 else 0
  x -= np.max(x, axis=axis, keepdims=True)
  if x.dtype == np.float32 or x.dtype == np.float64:
    np.exp(x, out=x)
  else:
    x = np.exp(x)
  x /= np.sum(x, axis=axis, keepdims=True)
  return x

def parser(outs):
  if len(outs) == 11:   # 11 for supercomboLeon.keras and JL11_dlc_model.h5
    path, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose = outs
      #---  path.shape = (1, 385)
  if len(outs) == 12:   # 12 for supercombo079.keras
    path, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose, state = outs

  out_dict = {}
  if path is not None:
    if path.shape[1] == PATH_DISTANCE*2 + 1:
      out_dict['path'] = path[:, :PATH_DISTANCE]
      out_dict['path_stds'] = softplus(path[:, PATH_DISTANCE:2*PATH_DISTANCE])
      out_dict['path_stds'][int(path[0,-1]):] = 1e3
    elif path.shape[1] == PATH_DISTANCE*2:  # indent error in old file
      out_dict['path'] = path[:, :PATH_DISTANCE]
      out_dict['path_stds'] = softplus(path[:, PATH_DISTANCE:2*PATH_DISTANCE])
  else:
    path_reshaped = path[:,:-1].reshape((path.shape[0], -1, PATH_DISTANCE*2 + 1))
    out_dict['paths'] = path_reshaped[:, :, :PATH_DISTANCE]
    out_dict['paths_stds'] = softplus(path_reshaped[:, :, PATH_DISTANCE:PATH_DISTANCE*2])
    out_dict['path_weights'] = softmax(path_reshaped[:,:,-1])
    lidx = np.argmax(out_dict['path_weights'])
    out_dict['path'] = path_reshaped[:, lidx, :PATH_DISTANCE]
    out_dict['path_stds'] = softplus(path_reshaped[:, lidx, PATH_DISTANCE:PATH_DISTANCE*2])

  if ll is not None:
    out_dict['lll'] = ll[:, :PATH_DISTANCE] + LANE_OFFSET
    out_dict['lll_prob'] = sigmoid(ll[:, -1])
    out_dict['lll_stds'] = softplus(ll[:, PATH_DISTANCE:-2])
    out_dict['lll_stds'][int(ll[0,-2]):] = 1e3
  if rl is not None:
    out_dict['rll'] = rl[:, :PATH_DISTANCE] - LANE_OFFSET
    out_dict['rll_prob'] = sigmoid(rl[:, -1])
    out_dict['rll_stds'] = softplus(rl[:, PATH_DISTANCE:-2])
    out_dict['rll_stds'][int(rl[0,-2]):] = 1e3

    # Find the distribution that corresponds to the current lead
  lead_reshaped = lead[:,:-3].reshape((-1,5,11))
  lead_weights = softmax(lead_reshaped[:,:,8])
  lidx = np.argmax(lead_weights[0])
  out_dict['lead_xyva'] = np.column_stack([lead_reshaped[:,lidx, 0] * LEAD_X_SCALE,
                                           lead_reshaped[:,lidx, 1] * LEAD_Y_SCALE,
                                           lead_reshaped[:,lidx, 2] * LEAD_V_SCALE,
                                           lead_reshaped[:,lidx, 3]])
  out_dict['lead_xyva_std'] = np.column_stack([softplus(lead_reshaped[:,lidx, 4]) * LEAD_X_SCALE,
                                               softplus(lead_reshaped[:,lidx, 5]) * LEAD_Y_SCALE,
                                               softplus(lead_reshaped[:,lidx, 6]) * LEAD_V_SCALE,
                                               softplus(lead_reshaped[:,lidx, 7])])
    # Find the distribution that corresponds to the lead in 2s
  out_dict['lead_prob'] = sigmoid(lead[:, -3])
  lead_weights_2s = softmax(lead_reshaped[:,:,9])
  lidx = np.argmax(lead_weights_2s[0])
  out_dict['lead_xyva_2s'] = np.column_stack([lead_reshaped[:,lidx, 0] * LEAD_X_SCALE,
                                              lead_reshaped[:,lidx, 1] * LEAD_Y_SCALE,
                                              lead_reshaped[:,lidx, 2] * LEAD_V_SCALE,
                                              lead_reshaped[:,lidx, 3]])
  out_dict['lead_xyva_std_2s'] = np.column_stack([softplus(lead_reshaped[:,lidx, 4]) * LEAD_X_SCALE,
                                                  softplus(lead_reshaped[:,lidx, 5]) * LEAD_Y_SCALE,
                                                  softplus(lead_reshaped[:,lidx, 6]) * LEAD_V_SCALE,
                                                  softplus(lead_reshaped[:,lidx, 7])])
  out_dict['lead_prob_2s'] = sigmoid(lead[:, -2])
  out_dict['lead_all'] = lead

  if meta is not None:
    out_dict['meta'] = meta
  if desire_pred is not None:
    out_dict['desire'] = desire_pred
  if desire_state is not None:
    out_dict['desire_state'] = desire_state

  if long_x is not None:
    out_dict['long_x'] = long_x
  if long_v is not None:
    out_dict['long_v'] = long_v
  if long_a is not None:
    out_dict['long_a'] = long_a
  if pose is not None:
    out_dict['trans'] = pose[:,:3]
    out_dict['trans_std'] = softplus(pose[:,6:9]) + 1e-6
    out_dict['rot'] = pose[:,3:6] * np.pi / 180.0
    out_dict['rot_std'] = (softplus(pose[:,9:12]) + 1e-6) * np.pi / 180.0
  return out_dict

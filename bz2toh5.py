"""   YPL, JLL, 2021.3.22 - 2022.2.14
bz2toh5.py generates training data from rlog.bz2 for train_model.py (e.g. modelB5)
from /home/jinn/YPN/ypdesk/remove.py
read /home/jinn/YPN/yp-Efficient1/bz2toh5_plot.py

(OP082) jinn@Liu:~/openpilot/tools/lib$ python bz2toh5.py
Input:
  /home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/rlog.bz2
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/global_pose
Output:
  /home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/radar.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
"""
import os
import h5py   # https://docs.h5py.org/en/stable/
import numpy as np
import matplotlib.pyplot as plt
import common.transformations.orientation as orient
from tools.lib.logreader import LogReader

def mkpath(all_dirs):
  for dir in all_dirs:
    frame_times = np.load(dir + 'global_pose/frame_times')
    frame_positions = np.load(dir + 'global_pose/frame_positions')
    frame_orientations = np.load(dir + 'global_pose/frame_orientations')
    frame_velocities = np.load(dir + 'global_pose/frame_velocities')
    velocities = np.linalg.norm(frame_velocities, axis=1)
      #---  frame_times.shape = (1200,)
      #---  frame_positions.shape = (1200, 3)
      #---  frame_orientations.shape = (1200, 4)
      #---  frame_velocities.shape = (1200, 3)
      #---  len(velocities) = 1200
    num_f = len(frame_times)
      #---  num_f = 1200
    if not os.path.isfile(dir + 'pathdata.h5'):
      with h5py.File(dir + 'pathdata.h5', 'w') as f:
        f.create_dataset('Path', (num_f-50, 51))
        f.create_dataset('V', (num_f-50, 1))
        for ori in range(num_f-50):
          ecef_from_local = orient.rot_from_quat(frame_orientations[ori])
            #---  ecef_from_local.shape = (3, 3)
          local_from_ecef = ecef_from_local.T
            #---  local_from_ecef.shape = (3, 3)
          frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[ori:] - frame_positions[ori])
            #---  frame_positions_local.shape = (1200, 3)
            #---  frame_positions_local.shape = (1199, 3)
          print('#---  len(frame_positions_local) =', len(frame_positions_local))
          x_f = frame_positions_local[:50, 0]
          y_f = frame_positions_local[:50, 1]
          linear_model = np.polyfit(x_f, y_f, 3)
          linear_model_fn = np.poly1d(linear_model)
          vaild_len = x_f[-1]
          x_e = np.array([1+i for i in range(50)])
          y_e = linear_model_fn(x_e)
          out = np.hstack([y_e, vaild_len])
          f['Path'][ori] = out
          f['V'][ori] = velocities[ori]
          print(ori, out)
    fh5 = h5py.File(dir + 'pathdata.h5', 'r')   # read pathdata.h5
    Path = fh5['Path']
    Velo = fh5['V']
    print('#---  shape.Path, shape.Velo =', shape.Path, shape.Velo)

def mkradar(files):
  for f in files:
    if 'bz2' not in f:
      continue
    lr = LogReader(f)
      #---  lr = <tools.lib.logreader.LogReader object at 0x7f8bf0406c10>
    logs = list(lr)
      #---  len(logs) = 69061
    new_list = [l.which() for l in logs]
    new_list = list(set(new_list))   # get unique items (to set and back to list)
      #---  len(new_list)  = 37
    plt.plot([l.carState.vEgo for l in logs if l.which() == 'carState'], linewidth=3)
    plt.title('Car speed from raw logs (m/s)', fontsize=25)
    plt.xlabel('boot time (s)', fontsize=18)
    plt.ylabel('speed (m/s)', fontsize=18)
    plt.show()

    r_t = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='radarState'])
    f_t = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='frame'])
    radarr = [l.radarState.leadOne for l in logs if l.which()=='radarState']
    frame = [l for l in logs if l.which()=='frame']
    num_r = len(radarr)
    num_f = len(frame)
      #---  num_r =, num_f = 1201 0
      #---  TData.rlog.bz2 does not have 'frame' but USAData.raw_log.bz2 does

    rcount = num_r - 1191  # -1191 produces very small .h5 for debugging
    radar_file = f.replace('rlog.bz2', 'radar.h5')
    radar = []
    if not os.path.isfile(radar_file):   # if radar.h5 does not exist
        #for i in range(fcount):
      for i in range(rcount):
        radar.append(radarr[i])
      with h5py.File(radar_file, 'w') as f1:
        f1.create_dataset('LeadOne', (rcount, 5))   # dRel, yRel, vRel, aRel, prob
        for i in range(rcount):
          d = radar[i].dRel
          y = radar[i].yRel
          v = radar[i].vRel
          a = radar[i].aRel
          if a==0 and y==0 and v==0 and d==0:
            prob = 0
          else:
            prob = 1
          f1['LeadOne'][i] = [d, y, v, a, prob]
    fh5 = h5py.File(radar_file, 'r')   # read radar.h5
      #--- list(fh5.keys()) = ['LeadOne']
    dataset = fh5['LeadOne']
      #--- dataset.shape = (10, 5)  # 10 = 1201 - 1191 (10 frames)
      #--- dataset.dtype = float32
      #print("    #--- dataset.dtype =", dataset.dtype)

if __name__ == '__main__':
  '''dirs = os.listdir('/home/jinn/dataA')
  dirs = ['/home/jinn/dataA/'+ i +'/' for i in dirs]
    #---  dirs = ['/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/']
  all_files = []
  for di1 in dirs:
    dir1 = os.listdir(di1)
    files = [di1 + f for f in dir1]
    for f in files:
      all_files.append(f)
  mkradar(all_files)   # use /dataA'''

  dirs1 = os.listdir('/home/jinn/dataB')
  all_dirs = ['/home/jinn/dataB/'+ i +'/' for i in dirs1]
  mkpath(all_dirs)   # use /dataB

'''
#---  len(new_list)  = 37
print(new_list)
['androidLog', 'cameraOdometry', 'can', 'carControl', 'carEvents', 'carParams', 'carState',
'clocks', 'controlsState', 'deviceState', 'driverCameraState', 'driverMonitoringState', 'driverState',
'gpsLocation', 'gpsNMEA', 'gpsLocationExternal', 'initData',
'lateralPlan', 'liveCalibration', 'liveLocationKalman', 'liveParameters', 'liveTracks',
'logMessage', 'longitudinalPlan', 'model', 'pandaState', 'procLog', 'qcomGnssDEPRECATD',
'radarState', 'roadCameraState', 'roadEncodeIdx',
'sendcan', 'sensorEvents', 'sentinel', 'thumbnail', 'ubloxGnss', 'ubloxRaw']

print([l.which() for l in logs[:10]])
['initData', 'sentinel', 'roadCameraState', 'roadCameraState', 'roadCameraState', 'roadCameraState', 'roadCameraState', 'roadCameraState', 'roadCameraState', 'roadCameraState']

  print("#---  all_files =", all_files)
#---  all_files = [
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/camera.h5',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/radar.h5',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/rlog.bz2',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/fcamera.hevc']

  print("#---  all_dirs =", all_dirs)
#---  all_dirs = [
'/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/',
'/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/']
'''

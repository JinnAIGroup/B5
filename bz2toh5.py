"""   YPL, JLL, 2021.3.22 - 2021.10.5
bz2toh5.py generates training data from rlog.bz2 for train_modelB5.py etc.

(OP082) jinn@Liu:~/openpilot/tools/lib$ python bz2toh5.py
Input: /home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/rlog.bz2
Output: /home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/radar.h5
"""
import os
import h5py   # https://docs.h5py.org/en/stable/
import numpy as np
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader

dirs = os.listdir('/home/jinn/dataA')
dirs = ['/home/jinn/dataA/'+ i +'/' for i in dirs]
#---  dirs = ['/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/']

all_files = []
for di1 in dirs:
  dir1 = os.listdir(di1)
  path = [di1 + d for d in dir1]
  for f in path:
    all_files.append(f)

def mkradar(files):
  for f in files:
    if 'bz2' not in f:
      continue
    lr = LogReader(f)
    #---  lr = <tools.lib.logreader.LogReader object at 0x7f8bf0406c10>
    logs = list(lr)
    #---  len(logs) = 69061
    new_list = [l.which() for l in logs]
    new_list = list(set(new_list)) # get unique items (to set and back to list)
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
    #---  num_r =, num_f  = 1201 0
    #---  TData.rlog.bz2 does not have 'frame' but USAData.raw_log.bz2 does

    rcount = num_r - 1191  # -1191 produces very small .h5 for debugging
    radar_file = f.replace('rlog.bz2', 'radar.h5')
    radar = []
    if not os.path.isfile(radar_file): # if radar.h5 does not exist
      #for i in range(fcount):
      for i in range(rcount):
        radar.append(radarr[i])
      with h5py.File(radar_file, 'w') as f1:
        f1.create_dataset('LeadOne', (rcount, 5)) # dRel, yRel, vRel, aRel, prob
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
    fh5 = h5py.File(radar_file, 'r')
    #--- list(fh5.keys()) = ['LeadOne']
    dataset = fh5['LeadOne']
    #--- dataset.shape = (10, 5)  # 10 = 1201 - 1191 (10 frames)
    #--- dataset.dtype = float32
    #print("    #--- dataset.dtype =", dataset.dtype)

if __name__ == '__main__':
  mkradar(all_files)

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

#print("#---  all_files =", all_files)
#---  all_files = [
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/camera.h5',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/radar.h5',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/rlog.bz2',
'/home/jinn/dataA/8bfda98c9c9e4291|2020-05-11--03-00-57--61/fcamera.hevc']
'''

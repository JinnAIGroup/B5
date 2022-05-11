"""   YPL, JLL, 2021.3.22 - 2022.5.5
bz2toh5.py generates net_outputs.lead (58-vector in outputs[3]) from rlog.bz2 for train_modelB5YJ.py
from /home/jinn/openpilot/tools/lib/bz2toh5B5.py
read /home/jinn/YPN/yp-Efficient1/bz2toh5_plot.py

(OP082) jinn@Liu:~/openpilot/tools/lib$ python bz2toh5.py
Input:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/raw_log.bz2 (/dataB USA)
Output:
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/lead_data.h5
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tools.lib.logreader import LogReader

def mkradar(files):
  for f in files:
    if 'bz2' not in f:
      continue
    print('#---  f =', f)
    lr = LogReader(f)  # <= ents <= event_read_multiple_bytes() <= capnp_log <= log = capnp.load()
      #---  lr = <tools.lib.logreader.LogReader object at 0x7f8bf0406c10>
    logs = list(lr)
    print('#---  logs[0] =', logs[0])
    print('#---  logs[100] =', logs[100])
    new_list = [l.which() for l in logs]
    new_list = list(set(new_list))   # set() creates a set of unique items in Python
    print('#---  len(logs), len(new_list) =', len(logs), len(new_list))
      #---  len(logs), len(new_list) = 76267 28

    CS_t = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='carState'])
    RS_t = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='radarState'])
    print('#---  len(CS_t), len(RS_t) =', len(CS_t), len(RS_t))
      #---  len(CS_t), len(RS_t) = 6000 1200  (100 Hz, 20 Hz or FPS)

    plt.plot([l.carState.vEgo for l in logs if l.which() == 'carState'], linewidth=3)  # carState, vEgo defined in log.capnp
    plt.title('Car speed from raw logs', fontsize=25)
    plt.xlabel('video time (10 ms)', fontsize=18)
    plt.ylabel('speed (m/s)', fontsize=18)
    plt.show()

    plt.plot([l.radarState.leadOne.dRel for l in logs if l.which() == 'radarState'], linewidth=3)
    plt.title('leadOne.dRel', fontsize=25)
    plt.xlabel('frame no. (20 FPS)', fontsize=18)
    plt.ylabel('dRel (m)', fontsize=18)
    plt.show()

    plt.plot([l.radarState.leadOne.yRel for l in logs if l.which() == 'radarState'], linewidth=3)
    plt.title('leadOne.yRel', fontsize=25)
    plt.xlabel('frame no. (20 FPS)', fontsize=18)
    plt.ylabel('yRel (m)', fontsize=18)
    plt.show()

    plt.plot([l.radarState.leadOne.modelProb for l in logs if l.which() == 'radarState'], linewidth=3)
    plt.title('leadOne.modelProb', fontsize=25)
    plt.xlabel('frame no. (20 FPS)', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.show()

    plt.plot([l.radarState.leadTwo.dRel for l in logs if l.which() == 'radarState'], linewidth=3)
    plt.title('leadTwo.dRel', fontsize=25)
    plt.xlabel('frame no.', fontsize=18)
    plt.ylabel('dRel (m)', fontsize=18)
    plt.show()

    leadOne = [l.radarState.leadOne for l in logs if l.which()=='radarState']
    leadTwo = [l.radarState.leadTwo for l in logs if l.which()=='radarState']
    print('#---  len(leadOne), len(leadTwo) =', len(leadOne), len(leadTwo))

      #lead_file = f.replace('rlog.bz2', 'lead_data.h5')   # use /dataA (Taiwan)
    lead_file = f.replace('raw_rlog.bz2', 'lead_data.h5')   # use /dataB (USA)
    frames = len(leadOne)  # total frames
    lead = []
    if not os.path.isfile(lead_file):   # if lead_data.h5 does not exist
      for i in range(frames):
        lead.append(leadOne[i])
      with h5py.File(lead_file, 'w') as f1:
        f1.create_dataset('LeadOne', (frames, 5))   # dRel, yRel, vRel, aRel, prob
        for i in range(frames):
          d = lead[i].dRel
          y = lead[i].yRel
          v = lead[i].vRel
          a = lead[i].aRel
          if a==0 and y==0 and v==0 and d==0:
            prob = 0
          else:
            prob = 1
          f1['LeadOne'][i] = [d, y, v, a, prob]
    #fh5 = h5py.File(lead_file, 'r')   # read lead_data.h5
      #--- list(fh5.keys()) = ['LeadOne']
    #dataset = fh5['LeadOne']
      #--- dataset.shape = (10, 5)  # 10 = 1201 - 1191 (10 frames)
      #--- dataset.dtype = float32''''''

if __name__ == '__main__':
  dirs = os.listdir('/home/jinn/dataB')
  dirs = ['/home/jinn/dataB/'+ i +'/' for i in dirs]
  all_files = []
  for di1 in dirs:
    dir1 = os.listdir(di1)
    files = [di1 + f for f in dir1]
    for f in files:
      all_files.append(f)
  mkradar(all_files)

'''
#---  len(logs) = 69061
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
'''

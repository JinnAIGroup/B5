#!/usr/bin/env python3
import importlib
import math
from collections import defaultdict, deque

import cereal.messaging as messaging
from cereal import car
from common.numpy_fast import interp
from common.params import Params
from common.realtime import Ratekeeper, Priority, config_realtime_process
from selfdrive.config import RADAR_TO_CAMERA
from selfdrive.controls.lib.cluster.fastcluster_py import cluster_points_centroid
from selfdrive.controls.lib.radar_helpers import Cluster, Track
from selfdrive.swaglog import cloudlog


class KalmanParams():
  def __init__(self, dt):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.1s and 1.0s
    assert dt > .01 and dt < .1, "Radar time step must be between .01s and 0.1s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [dt * 0.01 for dt in range(1, 11)]
    K0 = [0.12288, 0.14557, 0.16523, 0.18282, 0.19887, 0.21372, 0.22761, 0.24069, 0.2531, 0.26491]
    K1 = [0.29666, 0.29331, 0.29043, 0.28787, 0.28555, 0.28342, 0.28144, 0.27958, 0.27783, 0.27617]
    self.K = [[interp(dt, dts, K0)], [interp(dt, dts, K1)]]


def laplacian_cdf(x, mu, b):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_cluster(v_ego, lead, clusters):
  # match vision point to best statistical cluster match
  offset_vision_dist = lead.xyva[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_cdf(c.dRel, offset_vision_dist, lead.xyvaStd[0])
    prob_y = laplacian_cdf(c.yRel, -lead.xyva[1], lead.xyvaStd[1])
    prob_v = laplacian_cdf(c.vRel, lead.xyva[2], lead.xyvaStd[2])

    # This is isn't exactly right, but good heuristic
    return prob_d * prob_y * prob_v

  cluster = max(clusters, key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  dist_sane = abs(cluster.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(cluster.vRel - lead.xyva[2]) < 10) or (v_ego + cluster.vRel > 3)
  if dist_sane and vel_sane:
    return cluster
  else:
    return None


def get_lead(v_ego, ready, clusters, lead_msg, low_speed_override=True):
  # Determine leads, this is where the essential logic happens
  if len(clusters) > 0 and ready and lead_msg.prob > .5:
    cluster = match_vision_to_cluster(v_ego, lead_msg, clusters)
  else:
    cluster = None

  lead_dict = {'status': False}
  if cluster is not None:
    lead_dict = cluster.get_RadarState(lead_msg.prob)
  elif (cluster is None) and ready and (lead_msg.prob > .5):
    lead_dict = Cluster().get_RadarState_from_vision(lead_msg, v_ego)

  if low_speed_override:
    low_speed_clusters = [c for c in clusters if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_clusters) > 0:
      closest_cluster = min(low_speed_clusters, key=lambda c: c.dRel)

      # Only choose new cluster if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_cluster.dRel < lead_dict['dRel']):
        lead_dict = closest_cluster.get_RadarState()

  return lead_dict


class RadarD():
  def __init__(self, radar_ts, delay=0):
    self.current_time = 0

    self.tracks = defaultdict(dict)
    self.kalman_params = KalmanParams(radar_ts)

    # v_ego
    self.v_ego = 0.
    self.v_ego_hist = deque([0], maxlen=delay+1)

    self.ready = False

  def update(self, sm, rr, enable_lead):
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.updated['carState']:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
    if sm.updated['modelV2']:
      self.ready = True

    ar_pts = {}
    for pt in rr.points:
      ar_pts[pt.trackId] = [pt.dRel, pt.yRel, pt.vRel, pt.measured]

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    idens = list(sorted(self.tracks.keys()))
    track_pts = list([self.tracks[iden].get_key_for_cluster() for iden in idens])

    # If we have multiple points, cluster them
    if len(track_pts) > 1:
      cluster_idxs = cluster_points_centroid(track_pts, 2.5)
      clusters = [None] * (max(cluster_idxs) + 1)

      for idx in range(len(track_pts)):
        cluster_i = cluster_idxs[idx]
        if clusters[cluster_i] is None:
          clusters[cluster_i] = Cluster()
        clusters[cluster_i].add(self.tracks[idens[idx]])
    elif len(track_pts) == 1:
      # FIXME: cluster_point_centroid hangs forever if len(track_pts) == 1
      cluster_idxs = [0]
      clusters = [Cluster()]
      clusters[0].add(self.tracks[idens[0]])
    else:
      clusters = []

    # if a new point, reset accel to the rest of the cluster
    for idx in range(len(track_pts)):
      if self.tracks[idens[idx]].cnt <= 1:
        aLeadK = clusters[cluster_idxs[idx]].aLeadK
        aLeadTau = clusters[cluster_idxs[idx]].aLeadTau
        self.tracks[idens[idx]].reset_a_lead(aLeadK, aLeadTau)

    # *** publish radarState ***
    dat = messaging.new_message('radarState')
    dat.valid = sm.all_alive_and_valid() and len(rr.errors) == 0
    radarState = dat.radarState
    radarState.mdMonoTime = sm.logMonoTime['modelV2']
    radarState.canMonoTimes = list(rr.canMonoTimes)
    radarState.radarErrors = list(rr.errors)
    radarState.carStateMonoTime = sm.logMonoTime['carState']

    if enable_lead:
      if len(sm['modelV2'].leads) > 1:
        radarState.leadOne = get_lead(self.v_ego, self.ready, clusters, sm['modelV2'].leads[0], low_speed_override=True)
        radarState.leadTwo = get_lead(self.v_ego, self.ready, clusters, sm['modelV2'].leads[1], low_speed_override=False)
    return dat


# fuses camera and radar data for best lead detection
def radard_thread(sm=None, pm=None, can_sock=None):
  config_realtime_process(2, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("radard got CarParams")

  # import the radar from the fingerprint
  cloudlog.info("radard is importing %s", CP.carName)
  RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % CP.carName).RadarInterface

  # *** setup messaging
  if can_sock is None:
    can_sock = messaging.sub_sock('can')
  if sm is None:
    sm = messaging.SubMaster(['modelV2', 'carState'])
  if pm is None:
    pm = messaging.PubMaster(['radarState', 'liveTracks'])

  RI = RadarInterface(CP)

  rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
  RD = RadarD(CP.radarTimeStep, RI.delay)

  # TODO: always log leads once we can hide them conditionally
  enable_lead = CP.openpilotLongitudinalControl or not CP.radarOffCan

  while 1:
    can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    rr = RI.update(can_strings)

    if rr is None:
      continue

    sm.update(0)

    dat = RD.update(sm, rr, enable_lead)
    dat.radarState.cumLagMs = -rk.remaining*1000.

    pm.send('radarState', dat)

    # *** publish tracks for UI debugging (keep last) ***
    tracks = RD.tracks
    dat = messaging.new_message('liveTracks', len(tracks))

    for cnt, ids in enumerate(sorted(tracks.keys())):
      dat.liveTracks[cnt] = {
        "trackId": ids,
        "dRel": float(tracks[ids].dRel),
        "yRel": float(tracks[ids].yRel),
        "vRel": float(tracks[ids].vRel),
      }
    pm.send('liveTracks', dat)

    rk.monitor_time()


def main(sm=None, pm=None, can_sock=None):
  radard_thread(sm, pm, can_sock)


if __name__ == "__main__":
  main()

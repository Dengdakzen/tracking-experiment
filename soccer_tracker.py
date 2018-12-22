from numba import jit
import os.path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2
import json
import utils
from sort import convert_bbox_to_z, convert_x_to_bbox

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.feet_pos = np.array([np.round(0.5*(bbox[0] + bbox[2])),bbox[3]])

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    x_bbox = convert_x_to_bbox(self.kf.x)
    self.feet_pos = np.array([np.round(0.5*(x_bbox[0] + x_bbox[2])),x_bbox[3]])
    self.history.append(x_bbox)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

class Soccer_Players(object):
    """
    The goal is to obtain 23 trackers and make sure the tracking can resist to the end of the video
    """
    def __init__(self,tracklets_data,init_frame_num = 200):
        self.active = []
        self.sleep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        self.init_candidate = []
        for i in tracklets_data:
            if i['start_frame'] < init_frame_num and utils.in_court(utils.convert_bbox_to_feet_pos(i['boxes'][0])):
                self.init_candidate.append(i)
            

if __name__ == "__main__":
    with open('2min_tracklets.json','r') as w:
        data = json.load(w)
    candidates = []
    for i in data:
        if i['start_frame'] < 200:
            box = i['boxes'][0]
            feet_pos = np.array([np.round(box[0] + 0.5*box[2]),box[1] + box[3]])
            if utils.in_court(feet_pos):
                candidates.append(i)
        else:
            break
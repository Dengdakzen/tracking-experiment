import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

def compute_dot_product(p1,p2,pos):
    a = p1.reshape([2,1])
    b = p2.reshape([2,1])
    c = pos.reshape([2,1])
    d = np.concatenate((b-c,a-c),axis = 1)
    return np.linalg.det(d)

def upper_bound(pos):
    p1 = np.array([1088,232])
    p2 = np.array([2743,227])
    if compute_dot_product(p2,p1,pos) >= 0:
        return 1
    else:
        return 0

def left_bound(pos):
    p1 = np.array([1088,232])
    p2 = np.array([80,726])
    if compute_dot_product(p1,p2,pos) >= 0:
        return 1
    else:
        return 0

def right_bound(pos):
    p1 = np.array([3769,704])
    p2 = np.array([2743,227])
    if compute_dot_product(p1,p2,pos) >= 0:
        return 1
    else:
        return 0

def down_bound_1(pos):
    p1 = np.array([1966,778])
    p2 = np.array([80,726])
    if compute_dot_product(p2,p1,pos) >= 0:
        return 1
    else:
        return 0

def down_bound_2(pos):
    p1 = np.array([3769,704])
    p2 = np.array([1966,778])
    if compute_dot_product(p1,p2,pos) <= 0:
        return 1
    else:
        return 0

def in_court(pos):
    return upper_bound(pos) and left_bound(pos) and right_bound(pos) and down_bound_1(pos) and down_bound_2(pos)

def convert_bbox_to_feet_pos(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  return np.array([np.round(bbox[0] + 0.5*bbox[2]),bbox[1]])




if __name__ == "__main__":
    with open('2min_tracklets.json','r') as w:
        data = json.load(w)
    count = 0
    for i in data:
        if i['start_frame'] < 200:
            count += 1
        else:
            break
    print(count)
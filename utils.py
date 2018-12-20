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




if __name__ == "__main__":
    print(in_court(np.array([1500,300])))
    print(in_court(np.array([3800,800])))
    print(in_court(np.array([3800,0])))

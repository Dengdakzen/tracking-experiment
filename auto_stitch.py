import json
import numpy as np

inf_num = 100000

def dis_box2box(b1,b2):
    return np.sqrt((b1[0]+0.5*b1[2] - b2[0]-0.5*b2[2])**2 + (b1[1]+0.5*b1[3] - b2[1]-0.5*b2[3])**2)

class player:
    def __init__(self,tracklet):
        self.boxes = tracklet['boxes']
        self.start_frame = tracklet["start_frame"]
        self.end_frame = tracklet["end_frame"]
        self.time_threshold = 10


    def distance(self,tracklet):
        if tracklet["start_frame"] <= self.end_frame:
            return inf_num
        elif tracklet["start_frame"] < self.end_frame + self.time_threshold:
            return dis_box2box(self.boxes[-1],tracklet["boxes"][0])

    






if __name__ == "__main__":
    file_path = "2min_tracklets.json"

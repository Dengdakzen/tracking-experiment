import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

class point:
    def __init__(self,x,y):
        self.x = x
        self.y = y


class rectangle:
    def __init__(self,i):
        self.LT = point(i['x'],i['y'])
        self.RD = point(i['x']+i['width'],i['y'] + i['height'])

def compute_dot_product(p1,p2,p):
    return (p1.y - p.y)*(p2.x - p.x) - (p1.x - p.x)*(p2.y - p.y)

def upper_bound(i):
    p1 = point(1088,232)
    p2 = point(2743,227)
    p = point(i['x']+0.5*i['width'],i['y'] + i['height'])
    if compute_dot_product(p2,p1,p) >= 0:
        return 1
    else:
        return 0

def left_bound(i):
    p1 = point(1088,232)
    p2 = point(80,726)
    p = point(i['x']+0.5*i['width'],i['y'] + i['height'])
    if compute_dot_product(p1,p2,p) >= 0:
        return 1
    else:
        return 0

def right_bound(i):
    p1 = point(3769,704)
    p2 = point(2743,227)
    p = point(i['x']+0.5*i['width'],i['y'] + i['height'])
    if compute_dot_product(p1,p2,p) >= 0:
        return 1
    else:
        return 0

def down_bound_1(i):
    p1 = point(1966,778)
    p2 = point(80,726)
    p = point(i['x']+0.5*i['width'],i['y'] + i['height'])
    if compute_dot_product(p2,p1,p) >= 0:
        return 1
    else:
        return 0

def down_bound_2(i):
    p1 = point(3769,704)
    p2 = point(1966,778)
    p = point(i['x']+0.5*i['width'],i['y'] + i['height'])
    if compute_dot_product(p1,p2,p) <= 0:
        return 1
    else:
        return 0

def in_court(i):
    return upper_bound(i) and left_bound(i) and right_bound(i) and down_bound_1(i) and down_bound_2(i)

def IOU(i, j):
    A = rectangle(i)
    B = rectangle(j)
    W = min(A.RD.x, B.RD.x) - max(A.LT.x, B.LT.x)
    H = min(A.RD.y, B.RD.y) - max(A.LT.y, B.LT.y)
    if W <= 0 or H <= 0:
        return 0
    SA = (A.RD.x - A.LT.x) * (A.RD.y - A.LT.y)
    SB = (B.RD.x - B.LT.x) * (B.RD.y - B.LT.y)
    cross = W * H
    print(cross/(SA + SB - cross))
    return cross/(SA + SB - cross)


cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
largedata = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/dazhen/桌面/clipped_detection.avi',fourcc, 25.0, (3840,800))
savefilepath = "/home/dazhen/darknet2/clipped_3/players.json"
for i in range(3034):
    thisframe = {}
    filepath1 = "/home/dazhen/darknet2/clipped_3/jsons/" + str(i) + ".json"
    filepath2 = "/home/dazhen/darknet2/clipped_3/jsons_mid/" + str(i) + ".json"
    filepath3 = "/home/dazhen/darknet2/clipped_3/original_frames/" + str(i) + ".jpg"
    # print(filepath1)
    frame = cv2.imread(filepath3)
    
    
    with open(filepath1,'r') as a:
        data_ori = json.load(a)
        # print(data_ori)
    with open(filepath2,'r') as b:
        data_mid = json.load(b)
        # print(data_mid)
    thisframe['Frame'] = data_ori['Frame']
    player_ori = data_ori['Players']
    player_mid = data_mid['Players']
    players = []
    for j in player_ori:
        if (j['x']  < 1230 or j['x'] + j['width'] > 1330 or j['x']  < 2510 or j['x'] + j['width'] > 2610) and in_court(j):
            players.append(j)
            # cv2.rectangle(frame,(j['x'],j['y']),(j['x'] + j['width'],j['y']+j['height']),(255,0,0),2)

    players2 = []
    for k in player_mid:
        if (k['x'] >= 1230 and k['x'] + k['width'] <= 1330) or (k['x'] >= 2510 and k['x'] + k['width'] <= 2610) and upper_bound(k):
            players2.append(k)
            # cv2.rectangle(frame,(k['x'],k['y']),(k['x'] + k['width'],k['y']+k['height']),(0,0,255),2)
            # for j in players:
            #     if IOU(j,k) > 0.3:
                    # cv2.rectangle(frame,(j['x'],j['y']),(j['x'] + j['width'],j['y']+j['height']),(0,255,0),-1)

    total_players = players + players2
    maps = np.ones(len(total_players))
    for j_index in range(len(total_players)):
        for k_index in range(j_index + 1,len(total_players)):
            if maps[k_index] == 0:
                continue
            if(IOU(total_players[j_index],total_players[k_index]) > 0.1):
                if total_players[j_index]['probability'] > total_players[k_index]['probability']:
                    maps[k_index] = 0
                else:
                    maps[j_index] = 0
                    break

    final_players = []
    for j in range(maps.shape[0]):
        if maps[j] == 1:
            final_players.append(total_players[j])

    for j in final_players:
            cv2.rectangle(frame,(j['x'],j['y']),(j['x'] + j['width'],j['y']+j['height']),(255,0,0),2)

    thisframe['Players'] = final_players
    # out.write(frame)
    cv2.imshow("frame",frame)
    cv2.waitKey(1)
    largedata.append(thisframe)

# out.release()
with open(savefilepath,'w') as fi:
    json.dump(largedata,fi)
cv2.destroyAllWindows()


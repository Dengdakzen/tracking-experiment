import json
import os
import csv
import numpy as np

class tracklet:
    def __init__(self,input_vector):
        self.id = int(input_vector[1])
        self.start_frame = int(input_vector[0])
        self.end_frame = input_vector[0]
        self.boxes = [[int(float(input_vector[2])),int(float(input_vector[3])),int(float(input_vector[4])),int(float(input_vector[5]))]]

    def add(self,input_vector):
        self.end_frame = input_vector[0]
        self.boxes.append([int(float(input_vector[2])),int(float(input_vector[3])),int(float(input_vector[4])),int(float(input_vector[5]))])

class tracklets:
    def __init__(self):
        self.index = set()
        self.index2index = {}
        self.data = []
        self.count = 0

    def add(self,input_vector):
        if input_vector[1] in self.index:
            data_index = self.index2index[input_vector[1]]
            self.data[data_index].add(input_vector)
        else:
            self.index.add(input_vector[1])
            self.index2index[input_vector[1]] = len(self.data)
            new_tracklet = tracklet(input_vector)
            self.data.append(new_tracklet)
            self.count += 1




if __name__ == "__main__":
    file_path = './output/2min.txt'
    with open(file_path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        k = tracklets()

        for idx,row in enumerate(spamreader):
            new_row = np.array(row)
            k.add(new_row)

            # if idx > 100:
            #     break
    
    print(k.count)
    data = []
    for i in k.data:
        this = {"id":i.id,"start_frame":int(i.start_frame) - 1,"end_frame":int(i.end_frame) - 1,"boxes":i.boxes}
        data.append(this)

    # save_path = "2min_tracklets.json"
    # with open(save_path,'w+') as w:
    #     json.dump(data,w,indent=4)
    print(len(data))
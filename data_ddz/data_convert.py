import json
import numpy as np
import os
import csv

DetName = "data_ddz/det.txt"
with open(DetName,"w") as f:
    output = csv.writer(f)
    for i in range(3022):
        FileName = "{:s}{:0>5d}{:s}".format("data_ddz/result_2/",i,".json")
        with open(FileName,'r') as r:
            data = json.load(r)
            for index , element in enumerate(data):
                a = (i + 1 , index + 1,element['box'][0],element['box'][1],element['box'][2],element['box'][3],element['score'],-1,-1,-1)
                output.writerows([a])

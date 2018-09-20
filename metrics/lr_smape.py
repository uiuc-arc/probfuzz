#!/usr/bin/env python
import os
import json
import sys
import numpy as np
try:
    smape_thres=sys.argv[1]
    weight=sys.argv[2]
    bias=sys.argv[3]
    data_dir=sys.argv[4]
except:
    print('True')
    exit(1)

with open(os.path.join(data_dir, 'data.json')) as json_data:
    d = json.load(json_data)
    datas = [x for x in d.keys() if x=='x' or x=='y']
    data_x = sorted(datas)[0]
    data_y = sorted(datas)[1]
    predict = [x * float(weight) + float(bias) for x in d[data_x]]
    observe = d[data_y]
    ape = np.zeros(len(observe))
    for i,p in enumerate(predict):
        o = observe[i]
        ape[i] = abs(o - p) / ((abs(o) + abs(p)))
        smape = np.mean(ape)
        if abs(smape) <= smape_thres:
            print('True: ' + str(smape))
        else:
            print('False: ' + str(smape))

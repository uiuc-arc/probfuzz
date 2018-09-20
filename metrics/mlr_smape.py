#!/usr/bin/env python
import os
import json
import sys
import numpy as np
smape_thres=sys.argv[1]
weight=sys.argv[2]
bias=sys.argv[3]
data_dir=sys.argv[4]

w = np.fromstring(weight, dtype=np.float32, sep=',')
b = float(bias)
with open(os.path.join(data_dir, 'data.json')) as json_data:
    d = json.load(json_data)
    datas = [x for x in d.keys() if x == 'x' or x == 'y']
    data_x = np.array(d[sorted(datas)[0]])
    observe = np.array(d[sorted(datas)[1]])

    predict = np.matmul(data_x, w) + b

    ape = np.zeros(len(observe))
    for i, p in enumerate(predict):
        o = observe[i]
        ape[i] = abs(o - p) / ((abs(o) + abs(p)))
    smape = np.mean(ape)
    if abs(smape) <= smape_thres:
        print('True: ' + str(smape))
    else:
        print('False: ' + str(smape))
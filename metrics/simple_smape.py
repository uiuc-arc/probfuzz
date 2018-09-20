#!/usr/bin/env python
import os
import json
import sys
import numpy as np
try:
    smape_thres=sys.argv[1]
    weight=float(sys.argv[2])
    bias=float(sys.argv[3])
    weight2=float(sys.argv[4])
    bias2=float(sys.argv[5])
    data_dir=sys.argv[6]
except:
    print('True')
    exit(1)

diff_weight = abs(weight - weight2)/(abs(weight)+abs(weight2))
diff_bias = abs(bias - bias2)/(abs(bias)+abs(bias2))
mean = (diff_bias + diff_weight) / 2
if mean <= smape_thres:
    print('True: '+str(mean))
else:
    print('False: '+str(mean))

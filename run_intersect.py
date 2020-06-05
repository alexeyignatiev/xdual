#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## xdual.py
##

from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset
import numpy as np


if __name__ == '__main__':


    model = 'temp/digits_3_5_20000_16_data/digits_3_5_20000_16_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl'
    file_data  = 'bench/partition/digits_3_5_20000_16_data.csv'
    ofile = "digit_id_partition"
    nb = 50
    if (True):
        with open(file_data) as f:
            lines=f.readlines()
            for i, line in enumerate(lines[1:nb+1]):
                data = np.fromstring(line, dtype=float, sep=',')
                data_str = ",".join(map(str, data))
                ex = "python3.6 ./xdual.py --outputimages \"{}_first_{}\" -N 100 -e ortools -M -x '{}' --xtype contrastive -v  {}".format(ofile, i, data_str, model)
                print(ex)
                os.system(ex)
            for i, line in enumerate(lines[-nb:]):
                data = np.fromstring(line, dtype=float, sep=',')
                data_str = ",".join(map(str, data))
                ex = "python3.6 ./xdual.py --outputimages \"{}_last_{}\" -N 100 -e ortools -M -x '{}' --xtype contrastive -v  {}".format(ofile, i, data_str, model)
                print(ex)
                os.system(ex)
            
    model = 'temp/real_fake_20000_16_data/real_fake_20000_16_data_nbestim_100_maxdepth_6_testsplit_0.2.mod.pkl'
    file_data  = 'bench/gan/real_fake_20000_16_data.csv'
    ofile = "digit_id_gan"
    with open(file_data) as f:
        lines=f.readlines()
        for i, line in enumerate(lines[1:nb+1]):
            data = np.fromstring(line, dtype=float, sep=',')
            data_str = ",".join(map(str, data))
            ex = "python3.6 ./xdual.py --outputimages \"{}_first_{}\" -N 100 -e ortools -M -x '{}' --xtype contrastive -v  {}".format(ofile, i, data_str, model)
            print(ex)
            os.system(ex)
        for i, line in enumerate(lines[-nb:]):
            data = np.fromstring(line, dtype=float, sep=',')
            data_str = ",".join(map(str, data))
            ex = "python3.6 ./xdual.py --outputimages \"{}_last_{}\" -N 100 -e ortools -M -x '{}' --xtype contrastive -v  {}".format(ofile, i, data_str, model)
            print(ex)
            os.system(ex)

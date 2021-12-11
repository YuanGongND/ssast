# -*- coding: utf-8 -*-
# @Time    : 4/13/21 12:31 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_sc_summary.py

import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

requirement_list = {'test': 0.0}

# second pass
for requirement in requirement_list.keys():
    threshold = requirement_list[requirement]
    global_max_mAP = 0
    result = []
    result_cs = []
    result_mAP = []
    root_path = './exp/'
    exp_list = get_immediate_subdirectories(root_path)
    exp_list.sort()
    for exp in exp_list:
        if requirement in exp and os.path.isfile(root_path + exp + '/eval_result.csv'):
            try:
                print(exp)
                cur_res = np.loadtxt(root_path + exp + '/eval_result.csv', delimiter=',')

                result.append([exp, cur_res[0], cur_res[1], cur_res[2], cur_res[3]])
            except:
                pass

    np.savetxt('./result/' + requirement + '.csv', result, delimiter=',', fmt='%s')

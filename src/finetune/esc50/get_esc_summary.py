# -*- coding: utf-8 -*-
# @Time    : 11/15/20 5:19 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_esc_summary.py

# get summary of all esc exp results.

import os
import numpy as np

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

result = []
root_path = '/data/sls/scratch/yuangong/ssast/src/finetune/esc50/exp/'
exp_list = get_immediate_subdirectories(root_path)
exp_list.sort()
for exp in exp_list:
    if os.path.isfile(root_path + exp + '/result.csv'):
        try:
            print(exp)
            cur_res = np.loadtxt(root_path + exp + '/result.csv', delimiter=',')
            cur_acc_fold = list(np.loadtxt(root_path + exp + '/acc_fold.csv', delimiter=','))

            best_epoch = np.argmax(cur_res[:,1])

            result.append([exp]+cur_acc_fold)
        except:
            pass
np.savetxt('/data/sls/scratch/yuangong/ssast/src/finetune/esc50/result/esc_results.csv', result, delimiter=',', fmt='%s')
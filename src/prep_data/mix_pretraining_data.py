# -*- coding: utf-8 -*-
# @Time    : 8/13/21 5:31 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : mix_mask_tr_data.py

# combine audioset and librispeech, count how many samples of each of them.

import json
import random

def combine_json(file_list, name='audioset_librispeech960'):
    wav_list = []
    for file in file_list:
        with open(file, 'r') as f:
            cur_json = json.load(f)
        cur_data = cur_json['data']
        print(len(cur_data))
        random.shuffle(cur_data)
        for entry in cur_data:
            entry['labels'] = '/m/09x0r'

        wav_list = wav_list + cur_data
    with open(name + '.json', 'w') as f:
        print(len(wav_list))
        json.dump({'data': wav_list}, f, indent=1)


if __name__ == '__main__':
    audioset_data = '/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json'
    librispeech_data = '/data/sls/scratch/yuangong/sslast2/src/prep_data/librispeech_tr960_cut.json'
    combine_json([audioset_data, librispeech_data], name='audioset_librispeech')
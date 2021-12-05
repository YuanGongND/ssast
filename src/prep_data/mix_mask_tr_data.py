# -*- coding: utf-8 -*-
# @Time    : 8/13/21 5:31 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : mix_mask_tr_data.py

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
            # add base path
            if 'howto100m.json' in file:
                #print('add link')
                entry['wav'] = '/data/sls/d/howto/parsed_videos/' + entry['wav']

        wav_list = wav_list + cur_data[0: 1953082*2]
    with open(name + '.json', 'w') as f:
        print(len(wav_list))
        json.dump({'data': wav_list}, f, indent=1)


if __name__ == '__main__':
    fsd50k_data = '/data/sls/scratch/yuangong/aed-pc/datafiles/fsd50k_tr_full.json'
    audioset_data = '/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json'
    howto100m_data = '/data/sls/scratch/yuangong/sslast2/src/how100m/howto100m.json'
    librispeech_data = '/data/sls/scratch/yuangong/sslast2/src/how100m/librispeech_tr960.json'

    #combine_json([audioset_data, librispeech_data, fsd50k_data], name='as_li_fsd')
    combine_json([audioset_data, librispeech_data, fsd50k_data, howto100m_data], name='all2')

# mix speech and non-speech data.

# mix howto100m with full audioset

# mix librispeech960 with full audioset.
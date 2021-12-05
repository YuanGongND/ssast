# -*- coding: utf-8 -*-
# @Time    : 7/11/21 6:55 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_howto100m.py

# go through the VGGSound and the HowTo100M data and get the statistics

import os,torchaudio,pickle,json,time

def walk(path, name):
    sample_cnt = 0
    pathdata = os.walk(path)
    wav_list = []
    begin_time = time.time()
    for root, dirs, files in pathdata:
        for file in files:
            if file.endswith('.wav'):
                sample_cnt += 1
                waveform, sr = torchaudio.load(root + os.sep + file)
                # print('audio length = {:.3f}'.format(waveform.shape[1]/sr))
                # split into 10s clips, abandon the rest.
                clip_num = int(waveform.shape[1]/sr/10)

                for i in range(clip_num):
                    # 	/m/09x0r is dummy label, label is not needed for SSL
                    cur_path = root + os.sep + file
                    cur_dict = {"wav": cur_path[32:], "labels": '/m/09x0r', 'segment': str(i)}
                    wav_list.append(cur_dict)

                #print('audio clips = {:d}'.format(clip_num))
                if sr != 16000:
                    print(root + os.sep + file + ' sampling rate is not 16k!')
                if sample_cnt % 1000 == 0:
                    end_time = time.time()
                    print('find {:d}k .wav files, time eclipse: {:.1f} seconds.'.format(int(sample_cnt/1000), end_time-begin_time))
                    begin_time = end_time
                if sample_cnt % 1e4 == 0:
                    with open(name + '.json', 'w') as f:
                        json.dump({'data': wav_list}, f, indent=1)
                    print('file saved.')
    print(sample_cnt)
    # for subdir, dirs, files in os.walk(path):
    #     for filename in files:
    #         filepath = subdir + os.sep + filename
    #
    #         if filepath.endswith(".jpg") or filepath.endswith(".png"):
    #             print(filepath)

vggsound_path = '/data/sls/placesaudio/datasets/VGGSound'
howto100m_path = '/data/sls/d/howto/parsed_videos/'
walk(howto100m_path, 'howto100m')
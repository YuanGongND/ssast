# -*- coding: utf-8 -*-
# @Time    : 12/6/21 12:35 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : check_load_model.py

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_base_400/audio_model.158.pth', map_location=device)
#sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/frame_base_400/audio_model.197.pth', map_location=device)
sd = torch.load('./test_mdl.pth', map_location=device)

conv = sd['module.v.patch_embed.proj.weight']
print(conv.shape)

pos_embed = sd['module.v.pos_embed']
print(pos_embed.shape)

# keys = list(sd.keys())
# for key in keys:
#     if 'fshape' in key:
#         print(key)
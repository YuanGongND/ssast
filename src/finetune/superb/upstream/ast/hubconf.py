# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_frame_base_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Frame-400.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_frame_base_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Frame-400.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_frame_base_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Frame-400.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

# Patch-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_patch_base_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Patch-400.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_patch_base_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Patch-400.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_patch_base_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/ssast/pretrained_model/SSAST-Base-Patch-400.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

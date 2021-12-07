# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

# models used in final version, comparing patch vs frame base models. Patch models has already been tested in original paper, so main focus is base frame models.
# also test models scaling with tiny and small models
# name starts with 224 is a mistake, actually is a base 384 models.

def ast224_frame_base_1s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_1s_ssl_250(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.166.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_1s_ssl_400(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.197.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

# 6s for emotion
def ast224_frame_base_6s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_6s_ssl_250(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.166.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_6s_ssl_400(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.197.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

# 10s for sid
def ast224_frame_base_10s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_10s_ssl_250(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m250-contmask-1-mix-framenocluster-asli/models/audio_model.166.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_base_10s_ssl_400(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.197.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

# tiny and small models, for final version, tiny models has been tested in rebuttal, but do it again for insurance
# tiny models
def ast_tiny224_patch_1s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_tiny224_patch_1s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.185.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_tiny224_patch_6s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_tiny224_patch_6s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.185.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_tiny224_patch_10s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_tiny224_patch_10s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.185.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

# small models
def ast_small224_patch_1s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_small224_patch_1s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.148.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_small224_patch_6s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_small224_patch_6s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.148.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_small224_patch_10s_final_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast_small224_patch_10s_final_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'small224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36f-full-small224-pFalse-b64-lr2.67e-4-m400-contmask3-mix-16-16/models/audio_model.148.pth'
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

# -------------------------------------#

# models used in the rebuttal, basically tiny models.
def ast224_frame_1s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_1s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_1s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.120.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_1s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.120.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

# 10s for sid
# models used in the rebuttal, basically tiny models.
def ast224_frame_10s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 1024
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_10s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 1024
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_10s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.120.pth'
    kwargs["target_length"] = 1024
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_10s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.120.pth'
    kwargs["target_length"] = 1024
    return _UpstreamExpert(ckpt, *args, **kwargs)


# 6s for emotion

# models used in the rebuttal, basically tiny models.
def ast224_frame_6s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.1.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_6s_scratch(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.1.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_frame_6s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_f'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask36r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask-1-mix-framenocluster-asli/models/audio_model.120.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast224_patch_6s_ssl(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'tiny224_p'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask35r-full-tiny224-pFalse-b120-lr5e-4-m400-contmask3-mix-16-16/models/audio_model.120.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)



# models used in the first submission

# def ast224(ckpt, *args, **kwargs):
#     kwargs['model_size'] = 'tiny224'
#     kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask27-full-tiny224-pFalse-b128-lr5e-4-m400-contmask3-mix-randcont-predpatch-10x/models/audio_model.94.pth'
#     return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_5s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 500
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_5s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 500
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_5s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 500
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_7s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 700
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_7s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 700
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_7s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 700
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_8s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 800
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_8s_cmvn(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 800
    kwargs['cmvn'] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_8s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 800
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_8s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 800
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_9s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 900
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_9s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 900
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_9s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 900
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_scratch_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = None
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_im_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_10s_lo(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

# ablation study models 10s

def ast384_ab2_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mix-asli/models/audio_model.134.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab3_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpc-asli/models/audio_model.110.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab4_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpg-asli/models/audio_model.39.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab5_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mix-asli/models/audio_model.119.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab6_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpc-asli/models/audio_model.145.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab7_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpg-asli/models/audio_model.95.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab8_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-asli/models/audio_model.133.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab9_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-4-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab10_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-as/models/audio_model.155.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab11_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab13_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-bal/models/audio_model.20.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab14_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-li/models/audio_model.135.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab15_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpc-asli/models/audio_model.148.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab16_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpg-asli/models/audio_model.172.pth'
    kwargs["target_length"] = 1024
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_imnokd_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_nokd'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_assup_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/aed-trans/exp/formal2/table2/testk27-trans-whole-lr0.00001-bal-DeiT48-fs10-ts10-singlechannel-lrcut10-bs12-pretrainFalse/models/audio_model_wa.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe5_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.5.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe10_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.10.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe20_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.20.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe50_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.50.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe100_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.100.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe150_10s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.150.pth'
    kwargs["target_length"] = 1000
    kwargs["lo"] = True
    return _UpstreamExpert(ckpt, *args, **kwargs)

# abalation study models 6s

def ast384_ab2_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mix-asli/models/audio_model.134.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab3_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpc-asli/models/audio_model.110.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab4_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpg-asli/models/audio_model.39.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab5_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mix-asli/models/audio_model.119.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab6_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpc-asli/models/audio_model.145.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab7_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpg-asli/models/audio_model.95.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab8_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-asli/models/audio_model.133.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab9_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-4-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab10_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-as/models/audio_model.155.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab11_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab13_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-bal/models/audio_model.20.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab14_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-li/models/audio_model.135.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab15_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpc-asli/models/audio_model.148.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab16_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpg-asli/models/audio_model.172.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_imnokd_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_nokd'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_assup_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/aed-trans/exp/formal2/table2/testk27-trans-whole-lr0.00001-bal-DeiT48-fs10-ts10-singlechannel-lrcut10-bs12-pretrainFalse/models/audio_model_wa.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe5_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.5.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe10_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.10.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe20_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.20.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe50_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.50.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe100_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.100.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe150_6s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.150.pth'
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

# ablation study models, 1s

def ast384_ab2_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mix-asli/models/audio_model.134.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab3_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpc-asli/models/audio_model.110.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab4_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m100-contmask3-mpg-asli/models/audio_model.39.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab5_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mix-asli/models/audio_model.119.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab6_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpc-asli/models/audio_model.145.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab7_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m250-contmask3-mpg-asli/models/audio_model.95.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab8_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-1-mix-asli/models/audio_model.133.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab9_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask-4-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab10_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-as/models/audio_model.155.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab11_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.160.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab13_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-bal/models/audio_model.20.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab14_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-li/models/audio_model.135.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab15_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpc-asli/models/audio_model.148.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_ab16_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpg-asli/models/audio_model.172.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_imnokd_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384_nokd'
    kwargs['pretrain_path'] = 'imagenet'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_assup_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/aed-trans/exp/formal2/table2/testk27-trans-whole-lr0.00001-bal-DeiT48-fs10-ts10-singlechannel-lrcut10-bs12-pretrainFalse/models/audio_model_wa.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe5_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.5.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe10_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.10.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe20_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.20.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe50_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.50.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe100_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.100.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ast384_abe150_1s(ckpt, *args, **kwargs):
    kwargs['model_size'] = 'base384'
    kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-asli/models/audio_model.150.pth'
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)


# def ast384li(ckpt, *args, **kwargs):
#     kwargs['model_size'] = 'base384'
#     kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mix-li/models/audio_model.68.pth'
#     return _UpstreamExpert(ckpt, *args, **kwargs)
#
# def ast384mpc(ckpt, *args, **kwargs):
#     kwargs['model_size'] = 'base384'
#     kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpc-asli/models/audio_model.122.pth'
#     return _UpstreamExpert(ckpt, *args, **kwargs)
#
# def ast384mpg(ckpt, *args, **kwargs):
#     kwargs['model_size'] = 'base384'
#     kwargs['pretrain_path'] = '/data/sls/scratch/yuangong/sslast2/egs/audioset/exp/mask32-full-base384-pFalse-b24-lr1e-4-m400-contmask3-mpg-asli/models/audio_model.125.pth'
#     return _UpstreamExpert(ckpt, *args, **kwargs)
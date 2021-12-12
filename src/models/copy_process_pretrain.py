# -*- coding: utf-8 -*-
# @Time    : 12/6/21 6:48 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : copy_process_pretrain.py

import ast_models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # copy the pretrained models
# ast_mdl = ast_models.ASTModel(
#     fshape=16, tshape=16, fstride=16, tstride=16,
#     input_fdim=128, input_tdim=1024, model_size='base384',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_base_400/audio_model.158.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=True)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/patch_base_400/audio_model.999.pth')

# ast_mdl = ast_models.ASTModel(
#     fshape=16, tshape=16, fstride=16, tstride=16,
#     input_fdim=128, input_tdim=1024, model_size='base384',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_base_250/audio_model.117.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=False)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/patch_base_250/audio_model.999.pth')

# ast_mdl = ast_models.ASTModel(
#     fshape=16, tshape=16, fstride=16, tstride=16,
#     input_fdim=128, input_tdim=1024, model_size='tiny224',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_tiny_400/audio_model.185.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=False)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/patch_tiny_400/audio_model.999.pth')

# ast_mdl = ast_models.ASTModel(
#     fshape=16, tshape=16, fstride=16, tstride=16,
#     input_fdim=128, input_tdim=1024, model_size='small',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_small_400/audio_model.148.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=False)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/patch_small_400/audio_model.999.pth')

# ast_mdl = ast_models.ASTModel(
#     fshape=128, tshape=2, fstride=128, tstride=2,
#     input_fdim=128, input_tdim=1024, model_size='base384',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/frame_base_400/audio_model.197.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=False)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/frame_base_400/audio_model.999.pth')

# ast_mdl = ast_models.ASTModel(
#     fshape=128, tshape=2, fstride=128, tstride=2,
#     input_fdim=128, input_tdim=1024, model_size='base',
#     pretrain_stage=True)
# ast_mdl = torch.nn.DataParallel(ast_mdl)
# sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/frame_base_250/audio_model.166.pth', map_location=device)
# ast_mdl.load_state_dict(sd, strict=False)
# torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/frame_base_250/audio_model.999.pth')

ast_mdl = ast_models.ASTModel(
    fshape=128, tshape=2, fstride=128, tstride=2,
    input_fdim=128, input_tdim=1024, model_size='tiny',
    pretrain_stage=True)
ast_mdl = torch.nn.DataParallel(ast_mdl)
sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/frame_tiny_400/best_audio_model.pth', map_location=device)
ast_mdl.load_state_dict(sd, strict=False)
torch.save(ast_mdl.state_dict(), '/Users/yuan/Documents/ssast/pretrained_model/frame_tiny_400/audio_model.999.pth')

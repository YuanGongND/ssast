# -*- coding: utf-8 -*-
# @Time    : 12/6/21 6:48 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : copy_process_pretrain.py

import ast_model_uni_frame2
import torch

# copy the pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ast_mdl = ast_model_uni_frame2.ASTModel(
    fshape=16, tshape=16, fstride=16, tstride=16,
    input_fdim=128, input_tdim=1024, model_size='base384',
    pretrain=True)
ast_mdl = torch.nn.DataParallel(ast_mdl)
sd = torch.load('/Users/yuan/Documents/ssast/pretrained_model/patch_base_400/audio_model.158.pth', map_location=device)
ast_mdl.load_state_dict(sd, strict=False)
torch.save(ast_mdl.state_dict(), './test_processed_mdl.pth')
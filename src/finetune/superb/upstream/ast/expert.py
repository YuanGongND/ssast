# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

import torch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from ..interfaces import UpstreamBase
from .ast_models import ASTModel
from .audio import create_transform, FeatureExtractor

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        model_size = kwargs["model_size"]
        pretrain_path = kwargs["pretrain_path"]
        target_length = kwargs["target_length"]
        model_size, model_type = model_size.split('_')[0], model_size.split('_')[1]
        self.preprocessor = FeatureExtractor(target_length=target_length, apply_cmvn=False)
        if model_type == 'p':
            print('now train a patch models')
            self.model = ASTModel(fshape=16, tshape=16, fstride=10, tstride=10, input_tdim=target_length, input_fdim=128,
                                  model_size=model_size, pretrain_stage=False, load_pretrained_mdl_path=pretrain_path)
        else:
            print('now train a frame models')
            self.model = ASTModel(fshape=128, tshape=2, fstride=128, tstride=1, input_tdim=target_length, input_fdim=128,
                                  model_size=model_size, pretrain_stage=False, load_pretrained_mdl_path=pretrain_path)

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        features = torch.stack(features, dim=0)
        hidden_states, features = self.model(features)
        return {"last_hidden_state": features, "hidden_states": hidden_states}

if __name__ == '__main__':
    input_tdim = 1024
    #ast_mdl = TransModelMask()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ast_mdl = ASTModel(fshape=16, tshape=16, fstride=10, tstride=10, input_tdim=100, input_fdim=128,
                        model_size='base', pretrain_stage=False, load_pretrained_mdl_path='')
    test_input = torch.zeros([2, 100, 128]).to(device)
    o = ast_mdl(test_input)
    print(o.shape)
# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

import torch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from ..interfaces import UpstreamBase
from .ast_patch import ASTPatch
from .ast_frame import ASTModelUniFrame2
from .audio import create_transform, FeatureExtractor

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        #ckpt = torch.load(ckpt, map_location="cpu")
        model_size = kwargs["model_size"]
        pretrain_path = kwargs["pretrain_path"]
        target_length = kwargs["target_length"]

        # if it is just a model size
        if len(model_size.split('_')) == 1:
            # by default, model is patch based
            model_type = 'p'
        else:
            model_size, model_type = model_size.split('_')[0], model_size.split('_')[1]

        if 'lo' in kwargs:
            if kwargs['lo'] == True:
                self.stride = 16
            else:
                self.stride = 10
        else:
            self.stride = 10

        if 'cmvn' in kwargs:
            if kwargs['cmvn'] == True:
                self.apply_cmvn = True
            else:
                self.apply_cmvn = False
        else:
            self.apply_cmvn = False

        self.preprocessor = FeatureExtractor(target_length=target_length, apply_cmvn=self.apply_cmvn)

        if model_type == 'p':
            print('now train a patch model')
            # if pretrain_path == None:
            #     self.model = AST(imagenet_pretrain=False, audioset_pretrain=False, input_tdim=target_length, tstride=self.stride, fstride=self.stride, model_size=model_size)
            # elif pretrain_path == 'imagenet':
            #     self.model = AST(imagenet_pretrain=True, audioset_pretrain=False, input_tdim=target_length, tstride=self.stride, fstride=self.stride)
            # else:
            self.model = ASTPatch(audioset_pretrain=True, input_tdim=target_length, model_size=model_size, pretrain_path=pretrain_path, tstride=self.stride, fstride=self.stride)
        else:
            print('now train a frame model')
            self.model = ASTModelUniFrame2(audioset_pretrain=True, input_tdim=target_length, model_size=model_size, pretrain_path=pretrain_path, fshape=128, tshape=2, fstride=128, tstride=1)

    def forward(self, wavs):
        #print(len(wavs))
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        features = torch.stack(features, dim=0)

        # print(torch.mean(features))
        # print(torch.max(features))
        # print(torch.min(features))
        # print(torch.std(features[0]))
        # print(torch.mean(torch.abs(features)))

        hidden_states, features = self.model(features)

        #features = features / 4

        # print(features.shape)
        # print(torch.mean(features))
        # print(torch.max(features))
        # print(torch.min(features))
        # print(torch.std(features[0]))
        # print(torch.mean(torch.abs(features)))

        return {"last_hidden_state": features, "hidden_states": hidden_states}
        #return {"default": features}

if __name__ == '__main__':
    input_tdim = 1024
    #ast_mdl = TransModelMask()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ast_mdl = AST(imagenet_pretrain=True, input_tdim=100, model_size='tiny224')
    test_input = torch.zeros([2, 100, 128]).to(device)
    o = ast_mdl(test_input)
    print(o.shape)
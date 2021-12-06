# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_model_uni.py

# the unified ast model for all pretraining/fine-tuning tasks.

import torch.nn as nn
import torch
import sys
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/models/")
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/")
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2,
                 fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024,
                 audioset_pretrain=False,
                 model_size='base384', pretrain_path=None):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # initialize the model from timm package
        if audioset_pretrain == False:
            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base384_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384, base384_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base model) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = f_dim * t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('patch shape: frequency: {:d}, time: {:d}'.format(fshape, tshape))
            print('patch array dimension: frequency: {:d}, time: {:d}'.format(f_dim, t_dim))
            print('number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        elif audioset_pretrain == True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sd = torch.load(pretrain_path, map_location=device)

            print('now load a SSL pretrained model from ' + pretrain_path)
            audio_model = ASTModel(label_dim=527, fstride=128, tstride=2, fshape=128, tshape=2, input_fdim=128, input_tdim=1024,
                                      imagenet_pretrain=False, audioset_pretrain=False, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

            # if the stride of the pretrained model is different with that for fine-tuning.
            # generally they should be different as patch overlapping is not used in pretraining.
            if fstride != 128 or tstride != 2:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained model
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, 512, self.original_embedding_dim).transpose(
                1, 2).reshape(1, self.original_embedding_dim, 1, 512)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 512:
                new_pos_embed = new_pos_embed[:, :, :, 256 - int(t_dim / 2): 256 - int(t_dim / 2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(1, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def gen_maskid(self, sequence_len=512, mask_size=100, max_cont=10):
        mask_id = []
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cont = randrange(min(sequence_len - start_id + 1, max_cont))
            cur_mask = list(range(start_id, start_id + cont))
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))
        return torch.tensor(mask_id[:mask_size])

    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cont=3):
        mask_id = []
        if cont > 0:
            cont=randrange(cont) + 3
        else:
            # fix cont
            cont = -cont
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            # fix the cont number
            cont = cont
            cur_mask = []
            for i in range(0, cont):
                for j in range(0, cont):
                    mask_cand = start_id + 64 * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # if model has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    # mask patch embedding is learnable, random mask size (but same for a specific batch)
    def mpclearnpatchrand(self, x, show_mask=False, eval_mask=-1):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # # rand range mask from 1 to self.mask_patch
        # if eval_mask == -1:
        #     cur_mask_patch = randrange(300, self.mask_patch)
        # else:
        cur_mask_patch = eval_mask
        #print(cur_mask_patch)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), save the true embedding of masked samples
        encode_samples = torch.empty((B, cur_mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, cur_mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 576(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            # print('random mask with uniform distribution')
            mask_index[i] = self.gen_maskid_frame(512, cur_mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the x, add small number to avoid underflow
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # original deit code
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # directly use the AST output of the masked positions as the prediction of the masked embedding
        pred = torch.empty((B, cur_mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.cpredlayer(x[i, mask_index[i] + 2, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch, at the same time step
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, cur_mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * cur_mask_patch)
        nce = nce / (-1. * B * cur_mask_patch)

        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(self.mask_patch, 1).expand(self.mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            print(total)
            print(self.softmax(total))
            print(torch.argmax(self.softmax(total), dim=0))
            print(self.mask_correct)
            print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([128, 1024]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # masked patch generation task with random input
    def mpgrand(self, input, eval_mask=-1):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        # now input in shape [B, 512, 256], where 256*256 is the patch
        input = self.unfold(input).transpose(1, 2)

        # # rand range mask from 1 to self.mask_patch
        # if eval_mask == -1:
        #     cur_mask_patch = randrange(300, self.mask_patch)
        # else:
        cur_mask_patch = eval_mask
        #print(cur_mask_patch)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, cur_mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 576(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            mask_index[i] = self.gen_maskid_frame(512, cur_mask_patch)
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the x, add small number to avoid underflow
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # original deit code
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # directly use the AST output of the masked positions as the prediction of the masked embedding
        pred = torch.empty((B, cur_mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, cur_mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + 2, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse, mse

    def forward(self, x, task, mask_patch=-1):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        # for finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # for finetuning (ft), use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # for pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpclearnpatchrand(x, show_mask=False, eval_mask=mask_patch)
        # for pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpgrand(x, eval_mask=mask_patch)
        else:
            raise Exception('Task unrecognized.')

if __name__ == '__main__':
    input_tdim = 1024
    #ast_mdl = TransModelMask()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ast_mdl_u = ASTModelUniFrame2(imagenet_pretrain=False, mask_patch=400, contnum=5, model_size='tiny224')
    #ast_mdl_p = ASTModelUniFrame2(imagenet_pretrain=True, mask_patch=200, contnum=5, model_size='tiny224')
    #ast_mdl = torch.nn.DataParallel(ast_mdl)
    # ast_mdl = ast_mdl.to(device)
    test_input = torch.zeros([1, input_tdim, 128]).to(device)
    acc, nce = ast_mdl_u(test_input, task='mpc', mask_patch=400)

    plt.imshow(nce[0, 0, :])
    plt.show()

    print(torch.sum(nce) / 99 / 256)

    #pred, masked = ast_mdl(test_input, task='mpc')
    # mse, mse = ast_mdl(test_input, task='mpg')
    # pred = ast_mdl(test_input, task='ft')
    # print(pred.shape)
    # predm, mask = ast_mdl(test_input, task='reconstruct')

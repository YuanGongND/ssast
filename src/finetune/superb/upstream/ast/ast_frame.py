import torch.nn as nn
import torch
import sys
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/models/")
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/")
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from torch.cuda.amp import autocast
from timm.models.layers import to_2tuple
from linformer import LinformerSelfAttention
from random import randrange
from matplotlib import pyplot as plt
import random
import os

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

# linear attention block
class LinearBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, seq_len=1024):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinformerSelfAttention(dim, heads=num_heads, seq_len=seq_len)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = timm.layers.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = timm.models.vision_transformer.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ASTModelUniFrame2(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2,
                 fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=False, audioset_pretrain=False,
                 linformer=False, sinpos=False,
                 model_size='base384',
                 mask_patch=0, contnum=0, pretrain_path=None):

        super(ASTModelUniFrame2, self).__init__()
        print('---------------SSL AST Model Summary---------------')
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
        print('now use continuous mask')

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain == False:
            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base384_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=imagenet_pretrain)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384, base384_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            # TODO: remove this layer after fusion is tested
            self.mlp_head2 = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            # TODO: concatenate cls and avgtoken for finetuning
            self.mlp_head3 = nn.Sequential(nn.LayerNorm(self.original_embedding_dim * 2), nn.Linear(self.original_embedding_dim * 2, label_dim))

            self.embedding_weight = nn.Parameter(torch.tensor([1 / self.depth] * self.depth))
            # the weight to balance [cls+dist], and all sequence tokens
            self.output_weight = nn.Parameter(torch.tensor([0.5] * 2))

            # SSL Pretraining Stuff
            # only initialize these layer for pretraining, this avoids the model loading mismatch between up/down stream tasks
            if mask_patch != 0:
                print('currently in pretraining mode with {:d} masked patchs and clustering factor of {:d}'. format(mask_patch, contnum))
                self.mask_patch = mask_patch
                self.softmax = nn.Softmax(dim=-1)
                self.lsoftmax = nn.LogSoftmax(dim=-1)
                self.mask_correct = torch.nn.Parameter(torch.arange(0, self.mask_patch), requires_grad=False)
                self.contnum = contnum
                self.fshape, self.tshape = fshape, tshape
                self.fstride, self.tstride = fstride, tstride

                # classification prediction layer
                # print('now use single layer prediction layer')
                # self.cpredlayer = nn.Linear(self.original_embedding_dim, 256)
                # TODO: fix it after test
                print('now use two layer prediction layer')
                self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
                # generation prediction layer
                #self.gpredlayer = nn.Linear(self.original_embedding_dim, fshape*tshape)
                print('now use two layer prediction layer')
                self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
                # unfold used for generation
                self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
                #print('now use two-layer prediction layer')
                # self.predlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, self.original_embedding_dim))

                # TODO: fix it after test
                # the learnable mask embedding
                self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
                self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = f_dim * t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('patch shape: frequency: {:d}, time: {:d}'.format(fshape, tshape))
            print('patch array dimension: frequency: {:d}, time: {:d}'.format(f_dim, t_dim))
            print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            # TODO can use sinusoidal positional embedding instead
            if sinpos == True:
                print('sinusoidal positional embedding is used.')
                new_pos_embed = nn.Parameter(
                    get_sinusoid_encoding(self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim),
                    requires_grad=False)
            else:
                print('trainable positional embedding is used.')
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        elif audioset_pretrain == True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrain_base_path = ''
            sd = torch.load(pretrain_base_path + '/' + pretrain_path, map_location=device)

            # check if it is a supervised pretrained model, or self supervissed model
            # SSL with no split overlap has 514 pos embedding, supervised with 6 split has an overlap of 1214 pos embedding
            # can be used to judge if it is an SSL or supervised pretrained model

            save_mdl_pos = sd['module.v.pos_embed'].shape[1]
            # if SSL

            print('now loading a SSL pretrained model')
            print('now load model from ' + pretrain_path)
            audio_model = ASTModelUniFrame2(label_dim=527, fstride=128, tstride=2, fshape=128, tshape=2, input_fdim=128, input_tdim=1024,
                                      imagenet_pretrain=False, audioset_pretrain=False, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            # everything is based on self, not audio_model
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # reinitialize all mlp heads
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))
            self.mlp_head2 = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                           nn.Linear(self.original_embedding_dim, label_dim))
            self.mlp_head3 = nn.Sequential(nn.LayerNorm(self.original_embedding_dim * 2),
                                           nn.Linear(self.original_embedding_dim * 2, label_dim))

            self.output_weight = nn.Parameter(torch.tensor([0.5] * 2))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

            # if the stride of the pretrained model is different with that for fine-tuning.
            if fstride != 128 or tstride != 2:
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
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

    def gen_maskid_square(self, sequence_len=512, mask_size=100, max_cont=3):
        mask_id = []
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cont = randrange(max_cont)
            # cont = 2
            cur_mask = []
            for i in range(-cont + 1, cont):
                for j in range(-cont + 1, cont):
                    mask_cand = start_id + 64 * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_square2(self, sequence_len=512, mask_size=100, cont=3):
        mask_id = []
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

    def gen_maskid_square3(self, sequence_len=512, mask_size=100, cont=3):
        mask_id = []
        cont=randrange(cont) + 1
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

    # start mask from 2
    def gen_maskid_square4(self, sequence_len=512, mask_size=100, cont=3):
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

    # mask the entire frame rather then
    def gen_maskid_frame2(self, sequence_len=512, mask_size=100):
        mask_id = []
        cont = randrange(28) + 9
        # cluster from 9 (3**2) to 36 (6**2), keep it consistent with the patch based methods
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            # fix the cont number
            cur_mask = []
            # mask the entire frame
            for i in range(cont):
                mask_cand = start_id + i
                if mask_cand > 0 and mask_cand < sequence_len:
                    cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame3(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    # mask based on enery
    def gen_maskid_energy(self, sequence_len=512, mask_size=100, max_cont=3, input=None):
        mask_id = []
        weight = torch.mean(input, dim=1)
        #weight = weight - torch.min(weight) + 1e-6
        #print(torch.min(weight))
        weight = torch.relu(weight)
        while len(list(set(mask_id))) <= mask_size:
            #start_id = np.random.choice(sequence_len, p=weight)
            start_id = torch.multinomial(weight, 1, replacement=True)[0]
            cont = randrange(max_cont)
            # cont = 2
            cur_mask = []
            for i in range(-cont + 1, cont):
                for j in range(-cont + 1, cont):
                    mask_cand = start_id + 64 * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
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

        # TODO: change it back after test
        # x1 = (x[:, 0] + x[:, 1]) / 2
        # x2 = torch.mean(x[:, 2:, :], dim=1)
        # x = self.output_weight[0] * x1 + self.output_weight[1] * x2
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)

        x = self.mlp_head(x)
        return x

    #@autocast
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

        # TODO: change it back after test
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    #@autocast
    def finetuningtest1(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # TODO: fix this after test
        x_wa = torch.zeros_like(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
            #print(self.embedding_weight[blk_id])
            x_wa = x_wa + x * self.embedding_weight[blk_id]
        x = self.v.norm(x_wa)

        # for blk_id, blk in enumerate(self.v.blocks):
        #     x = blk(x)
        #     if blk_id > 999:
        #         break

        # x = self.v.norm(x)

        #x = (x[:, 0] + x[:, 1]) / 2
        x = torch.mean(x[:, 2:, :], dim=1)

        x = self.mlp_head(x)
        return x

    #@autocast
    # use decision-level fusion for cls token and embedding token.
    def finetuningfusion1(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        #x = self.v.norm(x)

        # TODO: change it back after test
        # x1 = (x[:, 0] + x[:, 1]) / 2
        # x2 = torch.mean(x[:, 2:, :], dim=1)
        # x = self.output_weight[0] * x1 + self.output_weight[1] * x2

        x_1 = torch.mean(x[:, 2:, :], dim=1)
        x_1 = self.mlp_head(x_1)

        x_2 = (x[:, 0, :] + x[:, 1, :]) / 2
        x_2 = self.mlp_head(x_2)

        x = (x_1 + x_2) / 2

        return x


    #@autocast
    # use decision-level fusion for cls token and embedding token.
    def finetuningfusion2(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)

        # TODO: change it back after test
        #x = self.v.norm(x)

        # TODO: change it back after test
        # x1 = (x[:, 0] + x[:, 1]) / 2
        # x2 = torch.mean(x[:, 2:, :], dim=1)
        # x = self.output_weight[0] * x1 + self.output_weight[1] * x2

        x_1 = torch.mean(x[:, 2:, :], dim=1)
        x_1 = self.mlp_head(x_1)

        x_2 = (x[:, 0, :] + x[:, 1, :]) / 2
        x_2 = self.mlp_head(x_2)

        x = x_1 * self.output_weight[0] + x_1 * self.output_weight[1]

        return x

    #@autocast
    # concatenate cls and avg_token
    def finetuningcat(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)

        x_1 = torch.mean(x[:, 2:, :], dim=1)
        x_2 = (x[:, 0, :] + x[:, 1, :]) / 2

        x = torch.cat([x_1, x_2], dim=-1)
        x = self.mlp_head3(x)

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
            mask_index[i] = self.gen_maskid_frame3(512, cur_mask_patch)
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
                result = [float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
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
            mask_index[i] = self.gen_maskid_frame3(512, cur_mask_patch)
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

    def reconstruct(self, input):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        # now input in shape [B, 512, 256], where 256=16*16 is the patch
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, self.mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 576(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            # mask_index[i] = torch.randperm(self.num_patches)[:self.mask_patch]
            mask_index[i] = self.gen_maskid_square(512, self.mask_patch, self.contnum)
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # mask the x, add small number to avoid underflow
        x = x * mask_dense

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
        pred = input.clone()  # [B, 512, 256]
        masked = input.clone()

        for i in range(B):
            pred[i, mask_index[i], :] = self.gpredlayer(x[i, mask_index[i] + 2, :])
            masked[i, mask_index[i], :] = -1

        # print(pred.shape)
        # print(masked.shape)
        fold = torch.nn.Fold(output_size=([128, 1024]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
        pred = fold(pred.transpose(1, 2))
        masked = fold(masked.transpose(1, 2))

        return pred, masked

    def getrep2(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # TODO: fix this after test
        reps = []
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
            reps.append(x)
        x = self.v.norm(x)

        return reps, x

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        return self.getrep2(x)
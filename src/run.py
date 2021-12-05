# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
from traintest_mask_improve import trainmask, validatemask

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used for training", choices=["librispeech", "howto100m", "audioset", "esc50", "speechcommands"])
parser.add_argument("--te_dataset", type=str, default=None, help="the dataset used for test, if None, use same with dataset", choices=["librispeech", "howto100m", "audioset", "esc50", "speechcommands"])

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, default=16, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, default=16, help="shape of patch on the time dimension")
parser.add_argument("--sinpos", help="if True, use sinusoidal positional embedding (untrainable)", type=ast.literal_eval, default='False')
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument('--model_size', help='the size of AST model', type=str, default='base384')
parser.add_argument('--linformer', help='use linear transformer or not ', type=ast.literal_eval, default='False')
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=0)
parser.add_argument('--edgemask', help='how many edges to mask to avoid shortcut', type=int, default=0)
parser.add_argument("--pretrain_path", type=str, default='none', help="the ssl pretrained model path")
parser.add_argument("--contnum", type=int, default=0, help="contimask")
parser.add_argument("--task", type=str, default='ft', help="task, in ['ft','mpg','mpc','reconstruct']")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for masking experiments, how many iterations to check lr and verification")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlphead_lr/lr, used in ft only")
parser.add_argument('--warmup', help='if use warmup', type=ast.literal_eval, default='True')
parser.add_argument('--freeze', help='if freeze ast except the mlp head', type=ast.literal_eval, default='False')
parser.add_argument('--freezetype', help='if freeze ast except the mlp head', type=int, default=0)

args = parser.parse_args()

# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')
    # dataset spectrogram mean and std, used to normalize the input
    norm_stats = {'librispeech':[-4.2677393, 4.5689974], 'howto100m':[-4.2677393, 4.5689974], 'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    target_length = {'librispeech': 1024, 'howto100m':1024, 'audioset':1024, 'esc50':512, 'speechcommands':128}
    # if add noise for data augmentation, only use for speech commands
    noise = {'librispeech': False, 'howto100m': False, 'audioset': False, 'esc50': False, 'speechcommands':True}

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                  'noise':noise[args.dataset]}

    if args.dataset == 'howto100m':
        print('not shuffle the data.')
        shuffle = False
    else:
        shuffle = True

    # evaluate howto100m on audioset
    if args.te_dataset != None:
        print('evaluate on another dataset: ' + args.te_dataset)
        args.dataset = args.te_dataset

    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

    # TODO: remove this after test
    if 'ft' not in args.task:
        # for param in audio_model.v.patch_embed.proj.parameters():
        #     param.requires_grad = False
        # audio_model.v.cls_token.requires_grad = False
        # audio_model.v.dist_token.requires_grad = False
        # audio_model.v.pos_embed.requires_grad = False

        # for param in audio_model.v.blocks.parameters():
        #     param.requires_grad = False

        # print('freeze ln layer all')
        # for module in audio_model.modules():
        #     # print(module)
        #     if isinstance(module, torch.nn.modules.normalization.LayerNorm):
        #         print(module)
        #         # if hasattr(module, 'weight'):
        #         #     module.weight.requires_grad_(False)
        #         # if hasattr(module, 'bias'):
        #         #     module.bias.requires_grad_(False)
        #         module.eval()
        pass
    else:
        pass
        # print('freeze ln layer all')
        # for module in audio_model.modules():
        #     # print(module)
        #     if isinstance(module, torch.nn.modules.normalization.LayerNorm):
        #         print(module)
        #         if hasattr(module, 'weight'):
        #             module.weight.requires_grad_(False)
        #         if hasattr(module, 'bias'):
        #             module.bias.requires_grad_(False)
        #         module.eval()

        # print('now freeze for downstream task')
        # for param in audio_model.parameters():
        #     param.requires_grad = False
        # for param in audio_model.mlp_head.parameters():
        #     param.requires_grad = True
        # for param in audio_model.mlp_head2.parameters():
        #     param.requires_grad = True
        # audio_model.v.cls_token.requires_grad = True
        # audio_model.v.dist_token.requires_grad = True
        # audio_model.v.pos_embed.requires_grad = True
        # for param in audio_model.v.patch_embed.proj.parameters():
        #     param.requires_grad = True

    audio_model = models.ASTModelUni(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride,
                                     linformer=args.linformer, sinpos=args.sinpos,
                                     fshape=args.fshape, tshape=args.tshape,
                                     input_fdim=128, input_tdim=target_length[args.dataset],
                                     imagenet_pretrain=args.imagenet_pretrain,
                                     audioset_pretrain=args.audioset_pretrain,
                                     model_size=args.model_size,
                                     mask_patch=args.mask_patch, contnum=args.contnum, pretrain_path=args.pretrain_path)
    audio_model = torch.nn.DataParallel(audio_model)
    # print('reinitialize the mlp layer')
    # audio_model.module.mlp_head = torch.nn.Sequential(torch.nn.LayerNorm(audio_model.module.original_embedding_dim), torch.nn.Linear(audio_model.module.original_embedding_dim, args.n_class))
    # try:
    #     print('also reinitialize second mlp layer')
    #     audio_model.module.mlp_head2 = torch.nn.Sequential(torch.nn.LayerNorm(audio_model.module.original_embedding_dim), torch.nn.Linear(audio_model.module.original_embedding_dim, args.n_class))
    #     audio_model.module.mlp_head3 = torch.nn.Sequential(torch.nn.LayerNorm(audio_model.module.original_embedding_dim*2), torch.nn.Linear(audio_model.module.original_embedding_dim*2, args.n_class))
    # except:
    #     print('no second or no third mlp layer')

    if args.freeze == True:
        print('now freeze for downstream task')
        for param in audio_model.module.parameters():
            param.requires_grad = False

        # mlp heads
        #if args.freezetype != 1:
        print('unfreeze mlp heads')
        for param in audio_model.module.mlp_head.parameters():
            param.requires_grad = True
        for param in audio_model.module.mlp_head2.parameters():
            param.requires_grad = True
        for param in audio_model.module.mlp_head3.parameters():
            param.requires_grad = True

        # if args.freezetype != 2:
        #     # cls token and positional_embedding
        #     print('unfreeze cls tokens and positional embedding')
        #     audio_model.module.v.cls_token.requires_grad = True
        #     audio_model.module.v.dist_token.requires_grad = True
        #     audio_model.module.v.pos_embed.requires_grad = True
        #
        # if args.freezetype != 3:
        #     # unfreeze transformer
        #     # print('unfreeze transformer')
        #     # blk_num = len(audio_model.module.v.blocks)
        #     # for blk_id, blk in enumerate(audio_model.module.v.blocks):
        #     #     # only unfreeze the last layer
        #     #     if blk_id == blk_num - 1:
        #     #         print('unfreeze {:d} transformer layer')
        #     #         for param in blk.parameters():
        #     #             param.requires_grad = True
        #
        #     for param in audio_model.module.v.blocks.parameters():
        #         param.requires_grad = True
        #
        # if args.freezetype != 4:
        #     # unfreeze transformer
        #     print('unfreeze patch projection layer')
        #     for param in audio_model.module.v.patch_embed.proj.parameters():
        #         param.requires_grad = True

        # if args.freezetype == 5:
        #     # unfreeze transformer
        #     print('unfreeze transformer')
        #     blk_num = len(audio_model.module.v.blocks)
        #     for blk_id, blk in enumerate(audio_model.module.v.blocks):
        #         # only unfreeze the last layer
        #         if blk_id == blk_num - 1:
        #             print('unfreeze {:d} transformer layer'.format(blk_id))
        #             for param in blk.parameters():
        #                 param.requires_grad = True
        #
        # if args.freezetype == 6:
        #     # unfreeze transformer
        #     print('unfreeze transformer')
        #     blk_num = len(audio_model.module.v.blocks)
        #     for blk_id, blk in enumerate(audio_model.module.v.blocks):
        #         # only unfreeze the last layer
        #         if blk_id == 0:
        #             print('unfreeze {:d} transformer layer'.format(blk_id))
        #             for param in blk.parameters():
        #                 param.requires_grad = True
        #
        # if args.freezetype == 7:
        #     # unfreeze transformer
        #     print('unfreeze transformer')
        #     blk_num = len(audio_model.module.v.blocks)
        #     for blk_id, blk in enumerate(audio_model.module.v.blocks):
        #         # only unfreeze the last layer
        #         print('unfreeze {:d} transformer layer'.format(blk_id))
        #         for param in blk.parameters():
        #             param.requires_grad = True
        #
        # if args.freezetype == 8:
        #     # unfreeze transformer
        #     print('unfreeze transformer')
        #     blk_num = len(audio_model.module.v.blocks)
        #     for blk_id, blk in enumerate(audio_model.module.v.blocks):
        #         # only unfreeze the last layer
        #         if blk_id != blk_num-1:
        #             print('unfreeze {:d} transformer layer'.format(blk_id))
        #             for param in blk.parameters():
        #                 param.requires_grad = True

        if args.freezetype == 9:
            # unfreeze transformer
            print('unfreeze patch projection layer')
            for param in audio_model.module.v.patch_embed.proj.parameters():
                param.requires_grad = True

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

if args.mask_patch == 0:
    print('Now starting training for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)
else:
    print('Now starting SSL pretraining for {:d} epochs'.format(args.n_epochs))
    trainmask(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best model on the validation set
    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])


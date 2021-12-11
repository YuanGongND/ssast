import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
import numpy as np
import pickle

def trainmask(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Now running on : ' + str(device))

    # initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    train_acc_meter = AverageMeter()
    train_nce_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
    trainables = audio_trainables
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    result = []
    audio_model.train()

    # training until break
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        # save from-scratch models before the first epoch
        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, global_step+1))

        for i, (audio_input, _) in enumerate(train_loader):
            # measure data loading time
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape)
            # if pretrain with discriminative objective
            if args.task == 'pretrain_mpc':
                acc, loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                # this is for multi-gpu support, in our code, loss is calculated in the model
                # pytorch concatenates the output of each gpu, we thus get mean of the losses of each gpu
                acc, loss = acc.mean(), loss.mean()
            # if pretrain with generative objective
            elif args.task == 'pretrain_mpg':
                loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                loss = loss.mean()
                # dirty code to make the code report mse loss for generative objective
                acc = loss
            # if pretrain with joint discriminative and generative objective
            elif args.task == 'pretrain_joint':
                acc, loss1 = audio_model(audio_input, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                acc, loss1 = acc.mean(), loss1.mean()
                loss2 = audio_model(audio_input, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                loss2 = loss2.mean()
                loss = loss1 + 10 * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            train_acc_meter.update(acc.detach().cpu().item())
            train_nce_meter.update(loss.detach().cpu().item())
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

            # pretraining data is usually very large, save model every epoch is too sparse.
            # save the model every args.epoch_iter steps.
            epoch_iteration = args.epoch_iter
            if global_step % epoch_iteration == 0:
                print('---------------- step '+ str(global_step) +' evaluation ----------------')
                equ_epoch = int(global_step/epoch_iteration) + 1
                acc_eval, nce_eval = validatemask(audio_model, test_loader, args, equ_epoch)

                print("masked acc train: {:.6f}".format(acc))
                print("nce loss train: {:.6f}".format(loss))
                print("masked acc eval: {:.6f}".format(acc_eval))
                print("nce loss eval: {:.6f}".format(nce_eval))
                result.append([train_acc_meter.avg, train_nce_meter.avg, acc_eval, nce_eval, optimizer.param_groups[0]['lr']])
                np.savetxt(exp_dir + '/result.csv', result, delimiter=',')

                if acc > best_acc:
                    best_acc = acc
                    torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, equ_epoch))
                if len(train_loader.dataset) > 2e5:
                    torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

                # if the task is generation, stop after eval mse loss stop improve
                if args.task == 'pretrain_mpg':
                    # acc_eval is in fact the mse loss, it is dirty code
                    scheduler.step(-acc_eval)
                else:
                    scheduler.step(acc_eval)

                print('# {:d}, step {:d}-{:d}, lr: {:e}'.format(equ_epoch, global_step-epoch_iteration, global_step, optimizer.param_groups[0]['lr']))

                _save_progress()

                finish_time = time.time()
                print('# {:d}, step {:d}-{:d}, training time: {:.3f}'.format(equ_epoch, global_step-epoch_iteration, global_step, finish_time-begin_time))
                begin_time = time.time()

                train_acc_meter.reset()
                train_nce_meter.reset()
                batch_time.reset()
                per_sample_time.reset()
                data_time.reset()
                per_sample_data_time.reset()
                loss_meter.reset()
                per_sample_dnn_time.reset()

                # change the models back to train mode
                audio_model.train()
                print('---------------- evaluation finished ----------------')
        epoch += 1

def validatemask(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    A_acc = []
    A_nce = []
    with torch.no_grad():
        for i, (audio_input, _) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape)
            # always use mask_patch=400 for evaluation, even the training mask patch number differs.
            if args.task == 'pretrain_mpc':
                acc, nce = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                A_acc.append(torch.mean(acc).cpu())
                A_nce.append(torch.mean(nce).cpu())
            elif args.task == 'pretrain_mpg':
                mse = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                # this is dirty code to track mse loss, A_acc and A_nce now track mse, not the name suggests
                A_acc.append(torch.mean(mse).cpu())
                A_nce.append(torch.mean(mse).cpu())
            elif args.task == 'pretrain_joint':
                acc, _ = audio_model(audio_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                mse = audio_model(audio_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)

                A_acc.append(torch.mean(acc).cpu())
                # A_nce then tracks the mse loss
                A_nce.append(torch.mean(mse).cpu())

        acc = np.mean(A_acc)
        nce = np.mean(A_nce)

    return acc, nce

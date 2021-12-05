import shutil
import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler

def trainmask(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Now running on : ' + str(device))

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    train_acc_meter = AverageMeter()
    train_nce_meter = AverageMeter()
    progress = []
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1000000))
    print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1000000))
    trainables = audio_trainables
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # LR scheduler
    print('auto lr scheduler')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    # print('fix lr scheduler')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 35, 40, 45, 50], gamma=0.5, last_epoch=-1)

    epoch += 1

    # amp part
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    result = []
    audio_model.train()

    # training until break
    while True:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        # save model before the first epoch
        if len(train_loader.dataset) > 2e5:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, global_step+1))

        for i, (audio_input, labels) in enumerate(train_loader):
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

            # single task
            # #with autocast():

            if args.task != 'mix':
                #cur_mask_patch = min(int(global_step/args.epoch_iter) * 5 + 100, 400)
                cur_mask_patch = args.mask_patch
                #print(cur_mask_patch)
                acc, loss = audio_model(audio_input, args.task, mask_patch=cur_mask_patch)
                acc, loss = acc.mean(), loss.mean()

            else:
                #cur_mask_patch = min(int(global_step/args.epoch_iter) * 5 + 100, 400)
                cur_mask_patch = args.mask_patch
                #print(cur_mask_patch)
                acc1, loss1 = audio_model(audio_input, 'mpc', mask_patch=cur_mask_patch)
                acc1, loss1 = acc1.mean(), loss1.mean()

                #print('only mask 250 for mpg')
                acc2, loss2 = audio_model(audio_input, 'mpg', mask_patch=cur_mask_patch)
                acc2, loss2 = acc2.mean(), loss2.mean()

                acc = acc1
                #print(loss1, loss2)
                loss = loss1 + 10 * loss2

            # original optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # amp optimiztion
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

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
                    best_acc_epoch = equ_epoch
                    torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, equ_epoch))
                if len(train_loader.dataset) > 2e5:
                    torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

                # if the task is generation, stop after eval mse loss stop improve
                if args.task == 'mpg':
                    scheduler.step(-acc_eval)
                else:
                    scheduler.step(acc_eval)
                #scheduler.step()

                #print('number of params groups:' + str(len(optimizer.param_groups)))
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

                # change the model back to train mode
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

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    A_acc = []
    A_nce = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            if args.task != 'mix':
                acc, nce = audio_model(audio_input, args.task, mask_patch=400)

                A_acc.append(torch.mean(acc).cpu())
                A_nce.append(torch.mean(nce).cpu())
            else:
                acc, _ = audio_model(audio_input, 'mpc', mask_patch=400)
                nce, _ = audio_model(audio_input, 'mpg', mask_patch=400)

                A_acc.append(torch.mean(acc).cpu())
                A_nce.append(torch.mean(nce).cpu())

        acc = np.mean(A_acc)
        nce = np.mean(A_nce)

    return acc, nce

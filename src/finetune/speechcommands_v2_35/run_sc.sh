#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp
mkdir -p ./slurm_log

pretrain_exp=patch_base_250
pretrain_epoch=999

dataset=speechcommands
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=128
noise=True
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json

bal=none
lr=2.5e-4
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=128
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
model_size=base384
head_lr=1

pretrain_path=/data/sls/scratch/yuangong/ssast/pretrained_model/${pretrain_exp}/audio_model.${pretrain_epoch}.pth
exp_dir=./exp/test05-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_epoch}-${head_lr}x-noise${noise}-6

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss BCE --metrics acc

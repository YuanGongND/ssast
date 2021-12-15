#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,12]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast_as"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp

if [ -e SSAST-Base-Patch-400.pth ]
then
    echo "pretrained model already downloaded."
else
    wget https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1 -O SSAST-Base-Patch-400.pth
fi

pretrain_exp=
pretrain_model=SSAST-Base-Patch-400
pretrain_path=./${pretrain_exp}/${pretrain_model}.pth

dataset=audioset
set=balanced
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
noise=False

task=ft_avgtok
model_size=base
head_lr=1
warmup=True

if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
elif [ $set == full ]
then
  bal=bal
  lr=1e-5
  epoch=5
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
fi

te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
freqm=48
timem=192
mixup=0.5
fstride=10
tstride=10
fshape=16
tshape=16
batch_size=12
exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-3

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 10 --lrscheduler_step 5 --lrscheduler_decay 0.5 --wa True --wa_start 6 --wa_end 25 \
--loss BCE --metrics mAP
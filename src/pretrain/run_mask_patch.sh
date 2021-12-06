#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[3],sls-1080-3,sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast_mask"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../sslast2/bin/activate
export TORCH_HOME=../../pretrained_models

echo $#
if [ $# -ne 4 ]
then
  echo 'test mode'
  ballr=1e-4
  mask_patch=100
  contnum=-1
  imp=False
  task=mix
else
  ballr=$1
  mask_patch=$2
  imp=$3
  contnum=4
  task=$4
fi

imagenetpretrain=$imp
model=ast
dataset=audioset
set=full
model_size='base384'
fshape=16
tshape=16
linformer=False
adaptschedule=True
if [ $set == balanced ]
then
  bal=none
  lr=$ballr
  if [ $imagenetpretrain == True ]
  then
    epoch=20
  else
    epoch=20
  fi
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
else
  bal=none
  lr=$ballr
  if [ $imagenetpretrain == True ]
  then
    epoch=20
  else
    epoch=20
  fi
  tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
  #tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
  #tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/librispeech_tr960_cut.json
  #tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/balanced_train_data_type1_2_mean.json
fi
te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
freqm=0
timem=0
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=24
lr_patience=2
exp_dir=./exp/mask32-${set}-${model_size}-p$imagenetpretrain-b$batch_size-lr${lr}-m${mask_patch}-contmask${contnum}-${task}-asli-100-1
#if [ -d $exp_dir ]; then
#  echo 'exp exist'
#  exit
#fi
#mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--model_size ${model_size} --linformer ${linformer} \
--adaptschedule ${adaptschedule} --mask_patch ${mask_patch} \
--fshape ${fshape} --tshape ${tshape} --edgemask 0 --n-print-steps 100 \
--contnum ${contnum} --task ${task} --lr_patience ${lr_patience} --epoch_iter 4000
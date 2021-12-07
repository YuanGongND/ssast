#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
export TORCH_HOME=../../pretrained_models

pretrain_exp=patch_base_400
pretrain_epoch=999

dataset=esc50
dataset_mean=-6.6268077
dataset_std=5.358466
target_length=512
noise=True

bal=none
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=50
batch_size=48
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
model_size=base384
head_lr=10

pretrain_path=/data/sls/scratch/yuangong/ssast/pretrained_model/${pretrain_exp}/audio_model.${pretrain_epoch}.pth
base_exp_dir=./exp/test03-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_epoch}-${head_lr}x-noise${noise}

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --pretrain False --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc
done

python ./get_esc_result.py --exp_path ${base_exp_dir}
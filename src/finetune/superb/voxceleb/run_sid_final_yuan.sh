#!/bin/bash
#SBATCH -p sm,1080
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-[5,12]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="s3p-sid"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
export TORCH_HOME=.

mdl=ast_small224_patch_10s_final_ssl
lr=1e-4

expname=$mdl$f${lr}final-afteraccept
expdir=./exp/sid_final/$expname
mkdir -p $expdir; mkdir -p ./log/sid

python3 run_downstream.py --expdir $expdir -m train -u $mdl -d voxceleb1 -c config.yaml -s hidden_states -o config.optimizer.lr=${lr} -f > ./log/sid/$expname.log

#!/bin/bash
#SBATCH -p sm,1080
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-[5,12]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="s3p-ks"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
export TORCH_HOME=.
mkdir exp

mdl=mdl=ssast_frame_base_6s
# search the learning rate
lr=5e-5

expname=ks_${mdl}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

python3 run_downstream.py -m train --expdir ${expdir} -n speech_commands -u $mdl -f -d speech_commands -c config.yaml -s hidden_states -o config.optimizer.lr=$lr > ./log/ks/$expname.log

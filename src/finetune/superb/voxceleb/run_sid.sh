#!/bin/bash
##SBATCH -p sm,1080
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-[5,12]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="s3p-sid"
#SBATCH --output=./log_%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/ssast/venvssast/bin/activate
export TORCH_HOME=.
mkdir exp

# frame based SSAST
mdl=ssast_frame_base_10s
lr=1e-4
## patch based SSAST
#mdl=ssast_patch_base_10s
#lr=1e-4

expname=sid_${mdl}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

python3 run_downstream.py --expdir $expdir -m train -u $mdl -d voxceleb1 -c config_sid.yaml -s hidden_states -o config.optimizer.lr=${lr} -f

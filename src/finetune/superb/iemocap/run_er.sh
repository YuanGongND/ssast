#!/bin/bash
##SBATCH -p sm,1080
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-[5,12]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="s3p-emo"
#SBATCH --output=./log_%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/ssast/venvssast/bin/activate
export TORCH_HOME=.
mkdir exp

# frame based SSAST
mdl=ssast_frame_base_6s
## patch based SSAST
#mdl=ssast_patch_base_6s

expname=emotion_${mdl}
expdir=./exp/$expname
mkdir -p $expdir

for test_fold in fold1 fold2 fold3 fold4 fold5;
do
  echo "running cross-validation on $test_fold"
  mkdir -p $expdir/unfreeze_cross-valid-on-${test_fold}; mkdir -p ./log/emotion/unfreeze_cross-valid-on-${test_fold}
  python3 run_downstream.py --expdir $expdir/unfreeze_cross-valid-on-${test_fold} -m train -u $mdl -d emotion -c config_er.yaml -s hidden_states -o "config.downstream_expert.datarc.test_fold='$test_fold'" -f
done

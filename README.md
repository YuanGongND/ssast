# SSAST: Self-Supervised Audio Spectrogram Transformer
 - [News](#News)
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Getting Started](#Getting-Started)
 - [SSAST Model](#SSAST-Model) 
 - [Data Preparation](#Data-Preparation)
 - [Self-Supervised Pretraining](#Self-Supervised-Pretraining)  
 - [Fine-tuning On Downstream Tasks](#Fine-tuning-On-Downstream-Tasks)
 - [Pretrained Models](#Pretrained-Models)
 - [Contact](#Contact)

## News
March, 2022: We released a new preprint [*CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification*](https://arxiv.org/abs/2203.06760), where we proposed a knowledge distillation based method to further improve the AST model performance without changing its architecture. This method can be applied in the fine-tuning stage of SSAST.

Feb 2022: I will present SSAST at [AAAI 2022](https://aaai.org/Conferences/AAAI-22/) at 12:00 PM - 1:45 PM (EST) on February 25th and then 7:45 PM - 9:30 PM (EST) on February 27th. 

## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/ssast/blob/main/figure/10854_ssast.png?raw=true" alt="Illustration of AST." width="800"/></p>

This repository contains the official implementation (in PyTorch) of the **Self-Supervised Audio Spectrogram Transformer (SSAST)** proposed in the AAAI 2022 paper [SSAST: Self-Supervised Audio Spectrogram Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/21315) ([Yuan Gong](https://yuangongnd.github.io/), [Cheng-I Jeff Lai](http://people.csail.mit.edu/clai24/), [Yu-An Chung](http://people.csail.mit.edu/andyyuan/), [James Glass](https://www.csail.mit.edu/person/jim-glass); MIT CSAIL). [[Slides](https://drive.google.com/file/d/1X4d21qJUSTSBpbVjB6p3IGaDH1ulxb-U/view?usp=sharing)]

SSAST is the first **patch-based** joint discriminative and generative self-supervised learning framework, and also the first self-supervised learning framework for AST. SSAST significantly boosts AST performance on all downstream tasks we evaluated with an average improvement of 60.9%, leading to similar or even better results than a supervised pretrained AST. SSAST can be used as a drop-in replacement of previous ImageNet (supervised) pretrained AST, and has the advantage of 1) no labeled data is used; 2) flexible patch size and shape, ImagenNet pretraining only supports square patches; and 3) better performance on many tasks, in particular speech tasks.

## Citing  
Please cite our paper if you find this repository useful. 
```  
@inproceedings{gong2022ssast,
  title={SSAST: Self-Supervised Audio Spectrogram Transformer},
  author={Gong, Yuan and Lai, Cheng-I and Chung, Yu-An and Glass, James},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={10699--10709},
  year={2022}
}
```  
```  
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```  

  
## Getting Started  

### Prepare the Environment
Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd ast/ 
python3 -m venv venvssast
source venvssast/bin/activate
pip install -r requirements.txt 
```

### Where is the code?
The SSAST model and pretraining function code is in `src/models/ast_model.py`.\
The self-supervised pretraining script is `src/pretrain/{run_mask_{frame,patch}, run_mask_{frame,patch}_tiny}`, which calls `src/run.py`, which then calls `src/traintest_mask.py`, which then calls `src/models/ast_model.py`.\
The fine-tuning scripts are in `src/finetune/`, for PSLA experiments, these scripts call `src/run.py`, which then calls `src/traintest.py`, which then calls `src/traintest_mask.py`, which then calls `src/models/ast_model.py`.\
The data preparation samples are in `src/prep_data`.

## SSAST Model

The SSAST model script is in ``src/models/ast_models.py``. 

```python
ASTModel(label_dim=527,
         fshape=16, tshape=16 fstride=10, tstride=10,
         input_fdim=128, input_tdim=1024, model_size='base',
         pretrain_stage=True, load_pretrained_mdl_path=None)
```  

### Parameters:
`label_dim` : The number of classes, only need to specify in the fine-tuning stage.\
`fshape`: The side length of the patch on the frequency dimension. \
`tshape`: The side length of the patch on the time dimension. \
`fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6. \
`tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6. \
`input_fdim`: The number of frequency bins of the input spectrogram.\
`input_tdim`: The number of time frames of the input spectrogram. \
`model_size`: The model size of AST, should be in `[tiny, small, base]` (default: `base`). \
`pretrain_stage`: Set as ``True`` in the self-supervised pretraining stage and ``False`` in the fine-tuning stage. \
`load_pretrained_mdl_path`: The pretrained model used for fine-tuning. Only needed when `pretrain_stage=False` as it is for fine-tuning. 

### Methods:
`forward(x, task, cluster=True, mask_patch=400)` \
The entry method of the class that calls fine-tuning and pretraining methods. Parameters:
* `x`: the input spectrogram in shape `[batch_size, time_frame_num, frequency_bin_num]. `Note: the input spectrogram should be normalized with dataset mean and std, see [here](https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191).
* `task`: the pretraining or fine-tuning task, should in `[ft_avgtok, ft_cls, pretrain_mpc, pretrain_mpg]`, see below for details.
* `cluster`: set `True` if using cluster patch masking strategy.
* `mask_patch`: the number of patch masked, only needed in the pretraining stage.

`finetuningavgtok(x)`: fine-tune the model by using the average of the outputs of all tokens as the clip represention. Return in shape `[batch_size, label_dim]`.

`finetuningcls(x)`: fine-tune the model by using the output of the `cls` token as clip represention. Return in shape `[batch_size, label_dim]`.

`mpc(x, mask_patch=mask_patch, cluster=cluster)`: pretrain the model with `mask_patch` number of masked patches with the discriminative objective. Return the accuracy and NCE loss of the pretext task.

`mpg(x, mask_patch=mask_patch, cluster=cluster)`: pretrain the model with `mask_patch` number of masked patches with the generative objective. Return the mean square error of the pretext task.

### Example:
``` python
# pretraining stage
# suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
input_tdim = 1024
# create a 16*16 patch based AST model for pretraining.
# note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
ast_mdl = ASTModel(
             fshape=16, tshape=16, fstride=16, tstride=16,
             input_fdim=128, input_tdim=input_tdim, model_size='base',
             pretrain_stage=True)
# # alternatively, create a frame based AST model
# ast_mdl = ASTModel(
#              fshape=128, tshape=2, fstride=128, tstride=2,
#              input_fdim=128, input_tdim=input_tdim, model_size='base',
#              pretrain=True)

# do pretraining, see src/traintest_mask.py for our full pretraining code
# input in shape [batch_size, input_tdim, input_fdim]
test_input = torch.zeros([10, input_tdim, 128])
# mask 100 patches for both discriminative and generative loss
acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
loss = nce_loss + 10 * mse_loss
# do back propagate and update the model, etc

# after pretraining, save the pretrained model.
# the code is designed for Dataparallel model
ast_mdl = torch.nn.DataParallel(ast_mdl)
torch.save(ast_mdl.state_dict(), './test_mdl.pth')

# fine-tuning stage
# now you have a labeled dataset you want to finetune AST on
# suppose the avg length is 100 frames (1s) and there are 35 classes
# the fshape and tshape must be same in pretraining and finetuning
# but fstride and tstride can be different in pretraining and finetuning
# using smaller strides improves the performance but also increase the computational overhead
# set pretrain_stage as False since now is in the finetuning stage
# provide the path of the pretrained model you want to load
input_tdim = 100  # fine-tuning data length can be different with pretraining data length
ast_mdl = ASTModel(label_dim=35,
             fshape=16, tshape=16, fstride=10, tstride=10,
             input_fdim=128, input_tdim=input_tdim, model_size='base',
             pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
# # alternatively, use a frame based AST model
# ast_mdl = ASTModel(label_dim=35,
#              fshape=128, tshape=2, fstride=128, tstride=1,
#              input_fdim=128, input_tdim=input_tdim, model_size='base',
#              pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')

# do finetuning, see src/traintest.py for our finetuning code
test_input = torch.zeros([10, input_tdim, 128])
prediction = ast_mdl(test_input, task='ft_avgtok')
# output should in shape [batch_size, label_dim]
print(prediction.shape)
# calculate the loss, do back propagate, etc
```

## Data Preparation

For both pretraining and fine-tuning, our dataloader requires two files:
* A json file containing path of the audio and corresponding label.
  * Self-supervised pretraining does not  need any label, but our current version of `dataloader.py` needs label information to run, you need to use a dummy label for pretraining data. Below is an example json file.

```json
 {
    "data": [
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/_/--4gqARaEJE_0.000.flac",
            "labels": "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
        },
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/_/--BfvyPmVMo_20.000.flac",
            "labels": "/m/03l9g"
        },
      // ... many audio files
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/0/-0BIyqJj9ZU_30.000.flac",
            "labels": "/m/07rgt08,/m/07sq110,/t/dd00001"
        }
    ]
}
```
* A csv file containing label information. The labels should be consistent with those in the json file.
  * Again, even for self-supervised pretraining, a dummy csv file is needed.
```csv
index,mid,display_name
0,/m/07rwj00,"dog"
1,/m/07rwj01,"rooster"
2,/m/07rwj02,"pig"
...
```

Examples: we provide our script to prepare data for a set of datasets.
* Librispeech: We have librispeech preparation code in `src/prep_data/librispeech/prep_librispeech.py`.
* AudioSet: You will need to download and process AudioSet data by yourself as AudioSet are YouTube videos, please see [here](https://research.google.com/audioset/download.html).
* FSD50K: FSD50K is not used in the paper, but FSD50K is AudioSet-like, 
* ESC-50: in `src/prep_data/esc50/prep_esc.py`
* Speechcommands V2-35: in `src/prep_data/speechcommands/prep_sc.py`
* Combining multiple datasets: see `src/prep_data/mix_pretraining_data` for our code to combine librispeech and AudioSet (used in the paper).

## Self-Supervised Pretraining

### Reproduce our experiments
The pretraining scripts are in `src/pretrain/`, we provide scripts to pretrain tiny/base and patch-based/frame-based AST model. The one we use for our main model in the paper is ``src/pretrain/run_mask_patch.sh``.
The scripts were tested on 4 GTX TITAN GPUs with 12GB memory. Please prepare the data as mentioned in [Data Preparation](#Data-Preparation).

### Pretrain on custom dataset
First, prepare the data files (the json and csv file) as described in [Data Preparation](#Data-Preparation). \
Second, modify our pretraining scripts are in `src/pretrain/`. Unless you have a resource constraint, it is always better to pretrain a base model than a tiny/small model. If your downstream task is speech, we suggest to train a frame-based SSAST (i.e., follow `run_mask_frame.sh`), if the downstream task is audio, we suggest to train a patch-based SSAST (i.e., follow `run_mask_patch.sh`). It is good to train and compare both. 

From `src/pretrain/run_mask_{frame,patch}.sh`, basically, the only things need to be changed are the data part.
```python
# your data json files
tr_data=/data/sls/scratch/yuangong/sslast2/src/prep_data/audioset_librispeech.json
te_data=/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json
# normalization stats, the mean and std of the entire dataset.
# if the custom dataset is also speech/audio, it is fine to use the same norm stats with us.
# check https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py
dataset_mean=-4.2677393
dataset_std=4.5689974
# audio length in frames, dataloader cut/pad all audios to this length
target_length=1024
# the number of frequency bins of your spectrogram. 
# if you want to train a frame-based SSAST, you need to change fshape with num_mel_bins
num_mel_bins=128
```

## Fine-tuning on Downstream Tasks

### PSLA training pipeline experiments 
* **ESC-50:** We suggest to start from ESC-50 experiments as our recipe is almost one click (i.e., the script handles data downloading, data processing, pre-trained model downloading, training and evaluation). Check `src/finetune/esc50/{run_esc_patch, run_esc_frame}.sh` for fine-tune patch-based and frame-based SSAST, respectively. To run, just `cd src/finetune/esc50` and then ` sbatch run_esc_{patch,frame}.sh` (slurm user) or `./run_esc_{patch,frame}.sh` (local user).
* **Speech Commands V2-35:** Check `src/finetune/speechcommands_v2_35/run_sc.sh`. It is also one-click and handles everything. Just `cd src/finetune/speechcommands_v2_35`, and then `sbatch run_sc.sh` (slurm user) or `./run_sc.sh` (local user).
* **AudioSet:** Check `src/finetune/audioset/run_as.sh`. Since AudioSet are YouTube videos, you will need to prepare the data by yourself. Note our experiment uses label enhanced balanced AudioSet, see [psla training pipeline](https://github.com/YuanGongND/psla) for how we enhance the label.

### SUPERB training pipeline experiments

**Note:** The SUPERB package function has changed after our experiments. In the lastest version, the new [`get_downsample_rate`](https://github.com/s3prl/s3prl/issues/96) is not friendly to patch-based AST as patch-based AST does not process spectrogram in frame-by-frame manner. If you want to reproduce our experiments on patch-based AST, please download [the old version SUPERB](https://github.com/s3prl/s3prl/tree/099ce807a6ffa6bf2482ceecfcaf83dea23da355), or, if you are only interested in frame-based AST (which performs better on speech tasks), you can use the latest version of SUPERB without problem.

We provide everything needed to reproduce our SUPERB experiments. The scripts are in `src/finetune/superb/`.

First, download and install SUPERB package [[old, work for both patch and frame AST]](https://github.com/s3prl/s3prl/tree/099ce807a6ffa6bf2482ceecfcaf83dea23da355) [[latest, only work for frame SSAST]](https://github.com/s3prl/s3prl).
```
cd yoursuperbpath/s3prl-master/
pip install -e ./
```
Second, modify the paths in `src/finetune/unstream/ast/hubconf.py` to your pretrained SSAST model absolute path, you can use our pretrained model.

Then, copy our `src/finetune/unstream/ast` to the SUPERB upstream directory:
```
cp -r src/finetune/unstream/ast yoursuperbpath/s3prl-master/s3prl/upstream/
```

Third, change the dataset path in `src/finetune/superb/{speechcommands_v1, iemocap, voxceleb}/{config_ks.yaml, config_er.yaml, config_sid.yaml}`.

Then, copy the downstream code and configuration to the SUPERB directory, e.g., for the speech commands task:
```
cp src/finetune/superb/speechcommands_v1/{run_ks.sh,config.yaml} yoursuperbpath/s3prl-master/s3prl/
```
Finally, run the training script:
```
cd yoursuperbpath/s3prl-master/s3prl/
# for local user
./run_ks.sh
# or, for slurm user
sbatch run_ks.sh
```
You can find the result logs in `yoursuperbpath/s3prl-master/s3prl/exp/expname/log.log`

### Fine-tune on custom dataset
It is easy to fine-tune on a new dataset. In general, PSLA training pipeline is stronger. You can start from any of the AudioSet, ESC-50, and SpeechCommands recipes (`run_sc.sh`, `run_esc_{frame,patch}.sh`, `run_as.sh`) and search the hyper-parameters. The only thing you need to modify is the shell script. For speech task, we suggest to fine-tune a frame-based SSAST, for audio task, we suggest to fine-tune a patch-based SSAST.

## Pretrained-Models

We provide the following self-supervised pretrained models. All models are trained with full AudioSet + Librispeech unless otherwise indicated. Click the model name to download. Tiny model should be able to pretrain and fine-tune on an 8GB GPU with a reasonable batch size.

For best performance, you should use either of the following models, patch-based AST is better for audio tasks, frame-based AST is better for speech tasks.

| Model Name                                                                                        | Data  | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size  | Avg Audio  Performance | Avg Speech  Performance |
|---------------------------------------------------------------------------------------------------|-------|-----------------|-----------------|-------------------|-------------|------------------------|-------------------------|
| [SSAST-Base-Patch-400](https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 400               | Base (89M)  | 59.9                   | 79.5                    |
| [SSAST-Base-Frame-400](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 400               | Base (89M)  | 57.6                   | 84.0                    |

Following models does not have best performance, we release them for analysis purpose and low-resource devices.

| Model Name                                                                                        | Data  | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size  | Avg Audio  Performance | Avg Speech  Performance |
|---------------------------------------------------------------------------------------------------|-------|-----------------|-----------------|-------------------|-------------|------------------------|-------------------------|
| [SSAST-Base-Patch-250](https://www.dropbox.com/s/mxrm9qog6aj8hif/SSAST-Base-Patch-250.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 250               | Base (89M)  | 58.6                   | 79.5                    |
| [SSAST-Base-Frame-250](https://www.dropbox.com/s/4e6l7ulhwrfoana/SSAST-Base-Frame-250.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 250               | Base (89M)  | 55.6                   | 81.6                    |
| [SSAST-Small-Patch-400](https://www.dropbox.com/s/i24w446rl9pkf05/SSAST-Small-Patch-400.pth?dl=1) | AudioSet + Librispeech | 16              | 16              | 400               | Small (23M) | 58.1                   | 78.2                    |
| [SSAST-Tiny-Patch-400](https://www.dropbox.com/s/fkbtf78y94113wz/SSAST-Tiny-Patch-400.pth?dl=1)   | AudioSet + Librispeech | 16              | 16              | 400               | Tiny (6M)   | 53.3                   | 75.7                    |
| [SSAST-Tiny-Frame-400](https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 400               | Tiny (6M)   | 47.8                   | untested                |

Following models are used in our ablation study in Table 2 of the paper, they does not have best performance, we release them for analysis purpose only.

<p align="left"><img src="https://github.com/YuanGongND/ssast/raw/main/figure/ablation.jpg?raw=true" alt="Ablation Study Table" width="400"/></p>

We set the 16x16 patch based AST pretrained with 400 masked patches, joint discriminative and generative objectives, on both AudioSet-2M and Librispeech as the base model. We then **change one factor at a time**. 

| ID in Ablation Study |             Model             | Download |
|:--------------------:|:-----------------------------:|:--------:|
|           1          |       100 Masked Patches      |   [Link](https://www.dropbox.com/s/0oyrtfbjzkwho2p/audio_model_100m.pth?dl=1)   |
|           2          | Only Discriminative Objective |   [Link](https://www.dropbox.com/s/znuzgwf2zvrpjkr/audio_model_dis.pth?dl=1)   |
|           3          |   Only Generative Objective   |   [Link](https://www.dropbox.com/s/u6ws5fjrid10x4p/audio_model_gen.pth?dl=1)   |
|           4          |   Pretrained w/ AudioSet-20K  |   [Link](https://www.dropbox.com/s/y6x2ck2ca3tb7d9/audio_model_as20k.pth?dl=1)   |
|           5          |   Pretrained w/ AudioSet-2M   |   [Link](https://www.dropbox.com/s/m9p782df3faql1q/audio_model_as.pth?dl=1)   |
|           6          |   Pretrained Librispeech960   |   [Link](https://www.dropbox.com/s/f4bn2qelu3m8ksu/audio_model_librispeech.pth?dl=1)   |

Above links are dropbox direct download links (i.e., wget works), no dropbox registration or sign in needed. For those don't have access to Dropbox, use a VPN or use the [OneDrive Links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/EuAuTEZNYPhOmlLFFjRFvGUBcgnIXBqFgFE33GDK69h-Zw?e=d3MEgT) or [腾讯微云链接们](https://share.weiyun.com/C4GoAcv0).

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.

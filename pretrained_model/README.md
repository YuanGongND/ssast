
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

Above links are dropbox direct download links (i.e., wget works), no Dropbox registration or sign in needed. For those don't have access to Dropbox, use a VPN or use the [OneDrive Links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/EuAuTEZNYPhOmlLFFjRFvGUBcgnIXBqFgFE33GDK69h-Zw?e=d3MEgT) or [腾讯微云链接们](https://share.weiyun.com/C4GoAcv0).

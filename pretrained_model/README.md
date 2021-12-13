
We provide the following self-supervised pretrained models. All models are trained with full AudioSet + Librispeech. Click the model name to download. Tiny model should be able to pretrain and fine-tune on an 8GB GPU with a reasonable batch size.

|       Model Name      | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size | #Params | Avg Audio  Performance | Avg Speech  Performance |
|:---------------------:|:---------------:|:---------------:|:------------------:|:----------:|:-------:|:----------------------:|:-----------------------:|
|  [SSAST-Base-Patch-400](https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1) |        16       |        16       |         400        |    Base    |   89M   |          59.9          |           79.5          |
|  [SSAST-Base-Patch-250](https://www.dropbox.com/s/mxrm9qog6aj8hif/SSAST-Base-Patch-250.pth?dl=1) |        16       |        16       |         250        |    Base    |   89M   |          58.6          |           79.5          |
|  [SSAST-Base-Frame-400](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1) |       128       |        2        |         400        |    Base    |   89M   |          57.6          |           84.0          |
|  [SSAST-Base-Frame-250](https://www.dropbox.com/s/4e6l7ulhwrfoana/SSAST-Base-Frame-250.pth?dl=1) |       128       |        2        |         250        |    Base    |   89M   |          55.6          |           81.6          |
|  [SSAST-Small-Patch-400](https://www.dropbox.com/s/i24w446rl9pkf05/SSAST-Small-Patch-400.pth?dl=1) |        16       |        16       |         400        |    Small   |   23M   |          58.1          |           78.2          |
|  [SSAST-Tiny-Patch-400](https://www.dropbox.com/s/fkbtf78y94113wz/SSAST-Tiny-Patch-400.pth?dl=1) |        16       |        16       |         400        |    Tiny    |    6M   |          53.3          |           75.7          |
|  [SSAST-Tiny-Frame-400](https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1) |       128       |        2        |         400        |    Tiny    |    6M   |          47.8          |          untested          |

Above links are dropbox direct download links (i.e., wget works). For those don't have access to Dropbox, use a VPN or use the [OneDrive Links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/EuAuTEZNYPhOmlLFFjRFvGUBcgnIXBqFgFE33GDK69h-Zw?e=d3MEgT).

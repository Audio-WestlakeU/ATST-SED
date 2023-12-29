# ATST-SED
The official implementations of "Fine-tune the pretrained ATST model for sound event detection" (accepted by ICASSP 2024). 

This work is highly related to [ATST](https://arxiv.org/abs/2204.12076), [ATST-Frame](https://arxiv.org/abs/2306.04186). Please check these works if you want to find out the principles of the ATST-SED.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Paper :star_struck:](https://arxiv.org/abs/2309.08153) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/ATST-SED/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](https://saoyear.github.io)

# Introduction

In a nutshell, ATST-SED proposes a fine-tuning strategy for the pretrained model integrated with the CRNN SED system.
<div align="center">
<image src="/src/flowchart.png"  width="500" alt="The proposed fine-tuning method for ATST-SED" />
</div>

# Updating Notice

- **Real dataset download**: The 7000+ strongly-labelled audio clips extracted from the AudioSet is provided in [this issue](https://github.com/Audio-WestlakeU/ATST-SED/issues/5).

- **Strong val dataset**: This dataset meta files are now updated to the repo.

- **About batch sizes**: If you change the batch sizes when fine-tuning ATST-Frame (Stage 1/2), you might probably need to change the `n_epochs` and `n_epochs_warmup` in the configuration file `train/local/confs/stage2.yaml` correspondingly. The fine-tuning of ATST-SED is related to the batch sizes, you might not reproduce the reported results when using a smaller batch sizes. The ablation study of the batch size setups is shown in the model performance below.

# Comparing with DCASE code
For better understanding of SED community, our codes are developed based on the [baseline codes](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline) of [DCASE2023 challenge task 4](https://dcase.community/). Therefore, the training progress is build under [`pytorch-lightning`](https://lightning.ai/).

we changed 
- [dataset.dataio.dataset.py](https://github.com/Audio-WestlakeU/ATST-SED/blob/main/desed_task/dataio/datasets_atst_sed.py) with our implementation. 
- [dataset.data_augm.py](https://github.com/Audio-WestlakeU/ATST-SED/blob/main/desed_task/data_augm.py) with an extra mixup module for the pretrained features.

The other parts in the `desed_task` are left unchange

# Get started
0. To reproduce our experiments, please first ensure you have the full DESED dataset (including 3000+ strongly labelled real audio clips from the AudioSet).

1. Ensure you have the correct environment. The environment of this code is the same as the DCASE 2023 baseline, please refer to their [docs/codes](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline) to configure your environment.

2. Download the pretrained [ATST checkpoint (atst_as2M.ckpt)](https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view?usp=drive_link). Noted that this checkpoint is fine-tuned by the AudioSet-2M.

3. Clone the ATST-SED codes by:

```
git clone https://github.com/Audio-WestlakeU/ATST-SED.git
```

4. Install our desed_task package by:

```
cd ATST-SED
```

```
pip install -e .
```

5. Change all required paths in `train/local/confs/stage1.yaml` and `train/local/confs/stage2.yaml` to your own paths. Noted that the pretrained ATST checkpoint path should be changed in **both** files.

6. Start training stage 1 by:

```
python train_stage1.py --gpus YOUR_DEVICE_ID,
```

We also supply a pretrained stage 1 ckpt for you to fine-tune directly. [Stage_1.ckpt](https://drive.google.com/file/d/1_sGve3FySPEqZQKYDO_DVntZ-VWVhtWN/view?usp=drive_link). If you cannot run stage 1 without `accm_grad=1`, we recommend you to use this checkpoint first.

7. When finishing the stage 1 training, change the path of the `model_init` in `train/local/confs/stage2.yaml` to the stage 1 checkpoint path (we saved top-5 models in both stages of training, you could use the best one as the model initialization in the stage 2, but use any one of the top-5 models should give the similar results).

8. Start training stage 2 by:

```
python train_stage2.py --gpus YOUR_DEVICE_ID,
```


# Performance

We report both DESED development set and public evaluation set results. The external set is the extra data extracted from the [AudioSet](http://research.google.com/audioset/)/[AudioSetStrong](https://research.google.com/audioset/download_strong.html). Please do not mess it with the 3000+ strongly labelled real audio clips from the AudioSet.

Please note that ATST-SED also get top-ranked performance on the public evaluation dataset without using external dataset. But we did not report it in our paper since the limited writing space. Top-1 model used extra weakly-labelled data from AudioSet, we are still mining these part of the data to improve the model performance.


| Dataset | External set | PSDS_1 | PSDS_2 | ckpt |
| :--------: | :--: | :----: | :----: | :---: |
| DCASE dev. set | - | 0.583 | 0.810 | [Stage2_wo_ext.ckpt](https://drive.google.com/file/d/1yMv05N0Nz5mSzlQ4YBb_sqOjazPbPDhw/view?usp=sharing) |
| DCASE public eval. set | - | 0.631 | 0.833 | same as the above |
| DCASE dev. set | Used | 0.587 | 0.812 |[Stage2_w_ext.ckpt](https://drive.google.com/file/d/16BP00UCRlAcSPgk-1kr0qrA6sjZNzhTf/view?usp=sharing) |
| DCASE public eval. set | Used | 0.631 | 0.846 | same as the above |

Two fine-tuned ATST-SED checkpoints are also released in the table below. You can download them and use them directly.

If you want to check the performance of the fine-tuned checkpoint:

```
python train_stage2.py --gpus YOUR_DEVICE_ID, --test_from_checkpoint YOUR_CHECKPOINT_PATH
```

---

**Ablation on batch sizes**:

We report the model performances on the development set with the following setups:

| Batch sizes | `n_epochs` | `n_epochs_warmup` | `accm_grad` | PSDS_1 | PSDS_2 |
| :--------: | :--: | :--: | :--: | :----: | :---: |
| [4, 4, 8, 8] | 40 | 2 | \ | 0.535 | 0.784 |
| [8, 8, 16, 16] | 80 | 2 | \ | 0.562 | 0.802 |
| [12, 12, 24, 24] | 125 | 5 | \ | 0.570 |0.805 |
| [4, 4, 8, 8] | 250 | 10 | 6 | 0.579 | 0.811 |

As shown in the table, if you cannot afford the default batch sizes, please make sure that they are in a proper level. Or, we recommend you to use `accm_grad` hyperparameter in the `stage2.yaml` to enlarge the batch sizes. However, using `accm_grad` would also decay the model performances, due to its influcences to the batch norm layer of the CNN model. Comparing with the reported results, you might get a poorer result from 56%~58% in PSDS1 (using last ckpt for validation).

# Citation

If you want to cite this paper:

```
@misc{shao2023finetune,
      title={Fine-tune the pretrained ATST model for sound event detection}, 
      author={Nian Shao and Xian Li and Xiaofei Li},
      year={2023},
      eprint={2309.08153},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

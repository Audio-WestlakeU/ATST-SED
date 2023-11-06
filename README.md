# ATST-SED
This repo includes the official implementations of "Fine-tune the pretrained ATST model for sound event detection".

This work is submitted to ICASSP 2024. 

This work is highly related to [ATST](https://arxiv.org/abs/2204.12076), [ATST-Frame](https://arxiv.org/abs/2306.04186). Please check these works if you want to find out the principles of the ATST-SED.

[Paper :star_struck:](https://arxiv.org/abs/2309.08153) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/ATST-SED/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](sao_year@126.com)

# Introduction

In a nutshell, ATST-SED proposes a fine-tuning strategy for the pretrained model integrated with the CRNN SED system.
<div align="center">
<image src="/src/flowchart.png"  width="500" alt="The proposed fine-tuning method for ATST-SED" />
</div>

# Updating Notice

- **Strong val dataset**: There exists a strong real validation set in the code/configuration files. In our experiment, we split the strong real dataset with a ratio of 9:1 to generate the train/validation sets for strong real data. But such way of model selection does not promise to select the best model in the stage 1 (PS: we did not use it in the stage 2). You can remove the strong validation set, but please also change the `val_object` to some other metrics and remove the strong val dataset in the train files. To reproduce our work, we uploaded the train/valiation tsv files in [this issue](https://github.com/Audio-WestlakeU/ATST-SED/issues/2).

- **About bathc sizes**: If you change the batch sizes when fine-tuning ATST-Frame (Stage 2), you might probably need to change the `n_epochs` and `n_epochs_warmup` in the configuration file `train/local/confs/stage2.yaml` correspondingly. The fine-tuning of ATST-SED is related to the batch sizes, you might not reproduce the reported results when using a smaller batch sizes. The ablation study of the batch size setups is shown in the model performance below.

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

Ablation on batch sizes:

We report the model performances on the development set with the following setups:

| Batch sizes | `n_epochs` | `n_epochs_warmup` | PSDS_1 | PSDS_2 |
| :--------: | :--: | :----: | :----: | :---: |
| [4, 4, 8, 8] | 40 | 2 | 0.535 | 0.784 |
| [8, 8, 16, 16] | 80 | 2 | 0.562 | 0.802 |
| [12, 12, 24, 24] | 125 | 5 | 0.570 |0.805 |

As shown in the table, if you cannot use the default batch sizes, please make sure that they are in proper level. Or, we recommend you to use `accm_grad` parameter to enlarge the batch sizes. 

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

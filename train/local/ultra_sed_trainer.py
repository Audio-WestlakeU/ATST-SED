import os
import random
from copy import deepcopy
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from local.rrc_module import RandomResizeCrop
from desed_task.utils.scaler import TorchScaler
import numpy as np
import torchmetrics

from .utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores
)

import sed_scores_eval

class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'

class MinMax(CustomAudioTransform):
    def __init__(self, min, max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) *2. - 1.
        return input

class ATSTNorm(nn.Module):
    def __init__(self):
        super(ATSTNorm, self).__init__()
        # Audio feature extraction
        self.amp_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.scaler = MinMax(min=-79.6482,max=50.6842) # TorchScaler("instance", "minmax", [0, 1])

    def amp2db(self, spec):
        return self.amp_to_db(spec).clamp(min=-50, max=80)

    def forward(self, spec):
        spec = self.scaler(self.amp2db(spec))
        return spec


class SEDTask4(pl.LightningModule):
    def __init__(
        self,
        hparams,
        encoder,
        sed_student,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
        sed_teacher=None
    ):
        super(SEDTask4, self).__init__()
        self.hparams.update(hparams)

        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = sed_teacher if sed_teacher is not None else deepcopy(sed_student)

        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        self.atst_norm = ATSTNorm()
        for param in self.sed_teacher.parameters():
            param.detach_()


        # instantiating losses
        self.supervised_loss = torch.nn.BCELoss()
        if hparams["training"]["self_sup_loss"] == "mse":
            self.selfsup_loss = torch.nn.MSELoss()
        elif hparams["training"]["self_sup_loss"] == "bce":
            self.selfsup_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.get_weak_teacher_f1_seg_macro = torchmetrics.classification.f_beta.MultilabelF1Score(
            len(self.encoder.labels),
            average="macro",
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()
        # buffer for event based scores which we compute using sed-eval

        self.val_thds = [0.5]
        self.val_scores_postprocessed_buffer_student_synth = {}
        self.val_scores_postprocessed_buffer_teacher_synth = {}
        self.val_scores_postprocessed_buffer_student_real = {}
        self.val_scores_postprocessed_buffer_teacher_real = {}

        self.val_loss_weak_student = []
        self.val_loss_weak_teacher = []
        self.val_loss_synth_student = []
        self.val_loss_synth_teacher = []
        self.val_loss_real_student = []
        self.val_loss_real_teacher = []

        self.val_buffer_student_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_student = {k: pd.DataFrame() for k in test_thresholds}
        self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_student_05_buffer = pd.DataFrame()
        self.decoded_teacher_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_student = {}
        self.test_scores_raw_buffer_teacher = {}
        self.test_scores_postprocessed_buffer_student = {}
        self.test_scores_postprocessed_buffer_teacher = {}

        self.freq_warp = RandomResizeCrop((1,1.0),time_scale=(1.0,1.0))


    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def update_ema(self, alpha, global_step, model, ema_model):
        for (k, ema_params), params in zip(ema_model.named_parameters(), model.parameters()):      
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, pretrained_feats, model):
        return model(self.scaler(self.take_log(mel_feats)), pretrained_feats)

    def training_step(self, batch, batch_indx):
        # This file only contains the validation/test steps
        pass

    def on_before_zero_grad(self, *args, **kwargs):
        # update EMA teacher
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler[0]["scheduler"].step_num,
            self.sed_student,
            self.sed_teacher,
        )

    def validation_step(self, batch, batch_indx):
        audio, atst_feats, labels, padded_indxs, filenames = batch
        sed_feats = self.mel_spec(audio)
        # prediction for student
        atst_feats = self.atst_norm(atst_feats)
        strong_preds_student, weak_preds_student = self.detect(sed_feats, atst_feats, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(sed_feats, atst_feats, self.sed_teacher)

        mask_weak = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(sed_feats)
            .bool()
        )

        mask_synth = (
            torch.tensor(
                [
                    str(Path(x).parent) == str(Path(self.hparams["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
            .to(sed_feats)
            .bool()
        )

        mask_real = (
            torch.tensor(
                [
                    str(Path(x).parent) == str(Path(self.hparams["data"]["strong_folder"]))
                    for x in filenames
                ]
            )
            .to(sed_feats)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()

            loss_weak_student = self.supervised_loss(
                weak_preds_student[mask_weak], labels_weak
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[mask_weak], labels_weak
            )
            self.val_loss_weak_student.append(loss_weak_student.item())
            self.val_loss_weak_teacher.append(loss_weak_teacher.item())

            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_preds_student[mask_weak], labels_weak.long()
            )
            self.get_weak_teacher_f1_seg_macro(
                weak_preds_teacher[mask_weak], labels_weak.long()
            )

        if torch.any(mask_real):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_real], labels[mask_real]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_real], labels[mask_real]
            )

            self.val_loss_real_student.append(loss_strong_student.item())
            self.val_loss_real_teacher.append(loss_strong_teacher.item())

            filenames_real = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["strong_folder"])
            ]

            (
                scores_raw_student_strong, scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_real],
                filenames_real,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=self.val_thds,
            )

            self.val_scores_postprocessed_buffer_student_real.update(
                scores_postprocessed_student_strong
            )

            (
                scores_raw_teacher_strong, scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_real],
                filenames_real,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=self.val_thds,
            )

            self.val_scores_postprocessed_buffer_teacher_real.update(
                scores_postprocessed_teacher_strong
            )

        if torch.any(mask_synth):
            loss_strong_student = self.supervised_loss(
                strong_preds_student[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[mask_synth], labels[mask_synth]
            )

            self.val_loss_synth_student.append(loss_strong_student.item())
            self.val_loss_synth_teacher.append(loss_strong_teacher.item())

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            (
                scores_raw_student_strong, scores_postprocessed_student_strong,
                decoded_student_strong,
            ) = batched_decode_preds(
                strong_preds_student[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=self.val_thds,
            )
            (
                scores_raw_teacher_strong, scores_postprocessed_teacher_strong,
                decoded_teacher_strong,
            ) = batched_decode_preds(
                strong_preds_teacher[mask_synth],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=self.val_thds,
            )
            
            self.val_scores_postprocessed_buffer_student_synth.update(
                scores_postprocessed_student_strong
            )
            
            self.val_scores_postprocessed_buffer_teacher_synth.update(
                scores_postprocessed_teacher_strong
            )
        return 0

    def psds1(self, input, ground_truth, audio_durations):
        return compute_psds_from_scores(
            input,
            ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
        )
    
    def psds2(self, input, ground_truth, audio_durations):
        return compute_psds_from_scores(
            input,
            ground_truth,
            audio_durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            )

    def on_validation_epoch_end(self):
        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()
        # Strong real
        ground_truth = sed_scores_eval.io.read_ground_truth_events(self.hparams["data"]["strong_val_tsv"])
        audio_durations = sed_scores_eval.io.read_audio_durations(self.hparams["data"]["strong_val_dur"])        
        ground_truth = {
            audio_id: gt for audio_id, gt in ground_truth.items()
            if len(gt) > 0
        }
        audio_durations = {
            audio_id: audio_durations[audio_id]
            for audio_id in ground_truth.keys()
        }
        psds1_student_sed_scores_real = self.psds1(self.val_scores_postprocessed_buffer_student_real, ground_truth, audio_durations)
        psds2_student_sed_scores_real = self.psds2(self.val_scores_postprocessed_buffer_student_real, ground_truth, audio_durations)
        psds1_teacher_sed_scores_real = self.psds1(self.val_scores_postprocessed_buffer_teacher_real, ground_truth, audio_durations)
        psds2_teacher_sed_scores_real = self.psds2(self.val_scores_postprocessed_buffer_teacher_real, ground_truth, audio_durations)
        
        # Strong synthetic
        ground_truth = sed_scores_eval.io.read_ground_truth_events(self.hparams["data"]["synth_val_tsv"])
        audio_durations = sed_scores_eval.io.read_audio_durations(self.hparams["data"]["synth_val_dur"])
        
        ground_truth = {
            audio_id: gt for audio_id, gt in ground_truth.items()
            if len(gt) > 0
        }
        audio_durations = {
            audio_id: audio_durations[audio_id]
            for audio_id in ground_truth.keys()
        }
        psds1_student_sed_scores_synth = self.psds1(self.val_scores_postprocessed_buffer_student_synth, ground_truth, audio_durations)
        psds2_student_sed_scores_synth = self.psds2(self.val_scores_postprocessed_buffer_student_synth, ground_truth, audio_durations)
        psds1_teacher_sed_scores_synth = self.psds1(self.val_scores_postprocessed_buffer_teacher_synth, ground_truth, audio_durations)
        psds2_teacher_sed_scores_synth = self.psds2(self.val_scores_postprocessed_buffer_teacher_synth, ground_truth, audio_durations)

        real_teacher_weak_loss = sum(self.val_loss_weak_teacher) / len(self.val_loss_weak_teacher)
        obj_metric = psds1_teacher_sed_scores_synth + psds2_teacher_sed_scores_synth

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro)
        self.log("val/weak/student/loss_weak", sum(self.val_loss_weak_student) / len(self.val_loss_weak_student))
        self.log("val/weak/teacher/loss_weak", real_teacher_weak_loss)
        self.log("val/real/student/psds1_sed_scores_eval", psds1_student_sed_scores_real)
        self.log("val/real/student/psds2_sed_scores_eval", psds2_student_sed_scores_real)
        self.log("val/synth/student/psds1_sed_scores_eval", psds1_student_sed_scores_synth)
        self.log("val/synth/student/psds2_sed_scores_eval", psds2_student_sed_scores_synth)
        self.log("val/real/student/loss_strong", sum(self.val_loss_real_student) / len(self.val_loss_real_student))
        self.log("val/synth/student/loss_strong", sum(self.val_loss_synth_student) / len(self.val_loss_synth_student))
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro)
        self.log("val/real/teacher/psds1_sed_scores_eval", psds1_teacher_sed_scores_real)
        self.log("val/real/teacher/psds2_sed_scores_eval", psds2_teacher_sed_scores_real)
        self.log("val/synth/teacher/psds1_sed_scores_eval", psds1_teacher_sed_scores_synth)
        self.log("val/synth/teacher/psds2_sed_scores_eval", psds2_teacher_sed_scores_synth)
        self.log("val/real/teacher/loss_strong", sum(self.val_loss_real_teacher) / len(self.val_loss_real_teacher))
        self.log("val/synth/teacher/loss_strong", sum(self.val_loss_synth_teacher) / len(self.val_loss_synth_teacher))

        # free the buffers
        self.val_scores_postprocessed_buffer_student_synth = {}
        self.val_scores_postprocessed_buffer_teacher_synth = {}
        self.val_scores_postprocessed_buffer_student_real = {}
        self.val_scores_postprocessed_buffer_teacher_real = {}

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()

        # free loss lists
        self.val_loss_weak_student = []
        self.val_loss_weak_teacher = []
        self.val_loss_synth_student = []
        self.val_loss_synth_teacher = []
        self.val_loss_real_student = []
        self.val_loss_real_teacher = []
        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_student"] = self.sed_student.state_dict()
        checkpoint["sed_teacher"] = self.sed_teacher.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, atst_feats, labels, padded_indxs, filenames = batch
        sed_feats = self.mel_spec(audio)
        atst_feats = self.atst_norm(atst_feats)
        strong_preds_student, weak_preds_student = self.detect(sed_feats, atst_feats, self.sed_student)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(sed_feats, atst_feats, self.sed_teacher)

        if not self.evaluation:
            loss_strong_student = self.supervised_loss(strong_preds_student, labels)
            loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        (
            scores_raw_student_strong, scores_postprocessed_student_strong,
            decoded_student_strong,
        ) = batched_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()) + [.5],
        )

        self.test_scores_raw_buffer_student.update(scores_raw_student_strong)
        self.test_scores_postprocessed_buffer_student.update(
            scores_postprocessed_student_strong
        )
        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = pd.concat([self.test_psds_buffer_student[th], decoded_student_strong[th]], ignore_index=True)

        (
            scores_raw_teacher_strong, scores_postprocessed_teacher_strong,
            decoded_teacher_strong,
        ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()) + [.5],
        )

        self.test_scores_raw_buffer_teacher.update(scores_raw_teacher_strong)
        self.test_scores_postprocessed_buffer_teacher.update(
            scores_postprocessed_teacher_strong
        )
        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = pd.concat([self.test_psds_buffer_teacher[th], decoded_teacher_strong[th]], ignore_index=True)

        # compute f1 score
        self.decoded_student_05_buffer = pd.concat([self.decoded_student_05_buffer, decoded_student_strong[0.5]])
        self.decoded_teacher_05_buffer = pd.concat([self.decoded_teacher_05_buffer, decoded_teacher_strong[0.5]])

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")

        if self.evaluation:
            # only save prediction scores
            save_dir_student_raw = os.path.join(save_dir, "student_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_student, save_dir_student_raw)
            print(f"\nRaw scores for student saved in: {save_dir_student_raw}")

            save_dir_student_postprocessed = os.path.join(save_dir, "student_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_student, save_dir_student_postprocessed)
            print(f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}")

            save_dir_teacher_raw = os.path.join(save_dir, "teacher_scores", "raw")
            sed_scores_eval.io.write_sed_scores(self.test_scores_raw_buffer_teacher, save_dir_teacher_raw)
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_raw}")

            save_dir_teacher_postprocessed = os.path.join(save_dir, "teacher_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(self.test_scores_postprocessed_buffer_teacher, save_dir_teacher_postprocessed)
            print(f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}")

        else:
            # calculate the metrics
            ground_truth = sed_scores_eval.io.read_ground_truth_events(self.hparams["data"]["test_tsv"])
            audio_durations = sed_scores_eval.io.read_audio_durations(self.hparams["data"]["test_dur"])
            if self.fast_dev_run:
                ground_truth = {
                    audio_id: ground_truth[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
            else:
                # drop audios without events
                ground_truth = {
                    audio_id: gt for audio_id, gt in ground_truth.items()
                    if len(gt) > 0
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in ground_truth.keys()
                }
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds1_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds2_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # synth dataset
            intersection_f1_macro_student = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            best_test_result = torch.tensor(max(psds1_student_psds_eval, psds2_student_psds_eval))

            results = {
                "hp_metric": best_test_result,
                "test/student/psds1_psds_eval": psds1_student_psds_eval,
                "test/student/psds1_sed_scores_eval": psds1_student_sed_scores_eval,
                "test/student/psds2_psds_eval": psds2_student_psds_eval,
                "test/student/psds2_sed_scores_eval": psds2_student_sed_scores_eval,
                "test/teacher/psds1_psds_eval": psds1_teacher_psds_eval,
                "test/teacher/psds1_sed_scores_eval": psds1_teacher_sed_scores_eval,
                "test/teacher/psds2_psds_eval": psds2_teacher_psds_eval,
                "test/teacher/psds2_sed_scores_eval": psds2_teacher_sed_scores_eval,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_student,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
            }
        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)


        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.opt, self.scheduler

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader

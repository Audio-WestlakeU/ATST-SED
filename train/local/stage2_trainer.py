import random
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB

from desed_task.data_augm import mixup_w_pretrained
from .ultra_sed_trainer import SEDTask4


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


class SEDICT(SEDTask4):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(SEDICT, self).__init__(*args, **kwargs)

    _exp_dir = None


    def training_step(self, batch, batch_indx):
        audio, atst_feats, labels, padded_indxs, _ = batch
        if len(self.hparams["training"]["batch_size"]) == 4:
            indx_strong, indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
            indx_synth = indx_strong + indx_synth
        else:
            indx_synth, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]
        
        sed_feats = self.mel_spec(audio)
        
        batch_num = sed_feats.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(sed_feats).bool()
        weak_mask = torch.zeros(batch_num).to(sed_feats).bool()
        unlabelled_mask = torch.zeros(batch_num).to(sed_feats).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        unlabelled_mask[indx_weak + indx_synth :] = 1

        # deriving weak labels
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()

        mixup_type = self.hparams["training"].get("mixup")
        atst_org = atst_feats.clone()
        sed_org = sed_feats.clone()
        labels_org = labels.clone()
        labels_weak_org = labels_weak.clone()
        if mixup_type is not None and 0.5 > random.random():
            sed_feats[weak_mask], atst_feats[weak_mask], labels_weak, perm_weak, c_weak = mixup_w_pretrained(
                sed_feats[weak_mask], atst_feats[weak_mask], labels_weak, mixup_label_type="soft"
            )
            sed_feats[strong_mask], atst_feats[strong_mask], labels[strong_mask], perm_strong, c_strong = mixup_w_pretrained(
                sed_feats[strong_mask], atst_feats[strong_mask], labels[strong_mask], mixup_label_type="soft"
            )
            sed_feats[unlabelled_mask], atst_feats[unlabelled_mask], _, perm_unlabelled, c_unlabelled = mixup_w_pretrained(
                sed_feats[unlabelled_mask], atst_feats[unlabelled_mask], labels[unlabelled_mask], mixup_label_type="soft"
            )
        else:
            perm_weak = None
        atst_feats = self.atst_norm(atst_feats)
        atst_org = self.atst_norm(atst_org)
        if 0.5 < random.random():
            atst_feats[weak_mask] = self.freq_warp(atst_feats[weak_mask])
            atst_org[weak_mask] = self.freq_warp(atst_org[weak_mask])
            atst_feats[strong_mask] = self.freq_warp(atst_feats[strong_mask])
            atst_org[strong_mask] = self.freq_warp(atst_org[strong_mask])
            atst_feats[unlabelled_mask] = self.freq_warp(atst_feats[unlabelled_mask])
            atst_org[unlabelled_mask] = self.freq_warp(atst_org[unlabelled_mask])

        # sed student forward
        strong_preds_student, weak_preds_student = self.detect(
            sed_feats, atst_feats, self.sed_student,
        )
        # supervised loss on strong labels
        loss_strong = self.supervised_loss(strong_preds_student[strong_mask], labels[strong_mask])
        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds_student[weak_mask], labels_weak)
        # total supervised loss
        tot_loss_supervised = loss_strong + loss_weak / 2

        with torch.no_grad():
            strong_preds_teacher, weak_preds_teacher = self.detect(
                sed_org, atst_org, self.sed_teacher
            )
            loss_strong_teacher = self.supervised_loss(
                strong_preds_teacher[strong_mask], labels_org[strong_mask]
            )
            loss_weak_teacher = self.supervised_loss(
                weak_preds_teacher[weak_mask], labels_weak_org
            )

        warmup = self.scheduler[0]["scheduler"]._get_scaling_factor()
        weight = ( self.hparams["training"]["const_max"] * warmup ) # heavy unsupervised weights
        # compute ICT loss
        if perm_weak is not None:
            # strong_part
            strong_org, weak_org = strong_preds_teacher[strong_mask], weak_preds_teacher[strong_mask]
            strong_mix = c_strong * strong_org + (1 - c_strong) * strong_org[perm_strong]
            weak_mix = c_strong * weak_org + (1 - c_strong) * weak_org[perm_strong]
            loss_ict_strong = self.selfsup_loss(strong_preds_student[strong_mask], strong_mix.clamp(0, 1).detach()) + \
                self.selfsup_loss(weak_preds_student[strong_mask], weak_mix.clamp(0, 1).detach()) / 2
            # weak_part
            strong_org, weak_org = strong_preds_teacher[weak_mask], weak_preds_teacher[weak_mask]
            strong_mix = c_weak * strong_org + (1 - c_weak) * strong_org[perm_weak]
            weak_mix = c_weak * weak_org + (1 - c_weak) * weak_org[perm_weak]
            loss_ict_weak = self.selfsup_loss(strong_preds_student[weak_mask], strong_mix.clamp(0, 1).detach()) + \
                self.selfsup_loss(weak_preds_student[weak_mask], weak_mix.clamp(0, 1).detach()) / 2
            # unlabelled_part
            strong_org, weak_org = strong_preds_teacher[unlabelled_mask], weak_preds_teacher[unlabelled_mask]
            strong_mix = c_unlabelled * strong_org + (1 - c_unlabelled) * strong_org[perm_unlabelled]
            weak_mix = c_unlabelled * weak_org + (1 - c_unlabelled) * weak_org[perm_unlabelled]
            loss_ict_unlabelled = self.selfsup_loss(strong_preds_student[unlabelled_mask], strong_mix.clamp(0, 1).detach()) + \
                self.selfsup_loss(weak_preds_student[unlabelled_mask], weak_mix.clamp(0, 1).detach()) / 2

            loss_ict = (loss_ict_strong + loss_ict_weak + loss_ict_unlabelled) / 6
            tot_self_loss = loss_ict * weight / 4
            self.log("train/student/tot_ict_loss", loss_ict)
        # Else MeanTeacher loss
        else:
            strong_self_sup_loss = self.selfsup_loss(
            strong_preds_student, strong_preds_teacher.detach()
            )
            weak_self_sup_loss = self.selfsup_loss(
                weak_preds_student, weak_preds_teacher.detach()
            )
            
            tot_self_loss = (strong_self_sup_loss + weak_self_sup_loss / 2) * weight
            self.log("train/student/weak_self_sup_loss", weak_self_sup_loss)
            self.log("train/student/strong_self_sup_loss", strong_self_sup_loss)

        step_num = self.scheduler[0]["scheduler"].step_num
        cnn_lr = self.opt[-1].param_groups[0]["lr"]
        tfm_lr = self.opt[-1].param_groups[-1]["lr"]
            
        tot_loss = tot_loss_supervised + tot_self_loss

        self.log("train/student/loss_strong", loss_strong)
        self.log("train/student/loss_weak", loss_weak)
        self.log("train/teacher/loss_strong", loss_strong_teacher)
        self.log("train/teacher/loss_weak", loss_weak_teacher)
        self.log("train/step", step_num, prog_bar=True)
        self.log("train/student/tot_self_loss", tot_self_loss, prog_bar=True)
        self.log("train/weight", weight)
        self.log("train/student/tot_supervised", tot_loss_supervised, prog_bar=True)
        self.log("train/tfm_lr", tfm_lr)
        self.log("train/cnn_lr", cnn_lr, prog_bar=True)

        return tot_loss
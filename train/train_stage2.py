import argparse
import os
import pandas as pd
import torch
import yaml

from desed_task.dataio.sampler import ConcatDatasetBatchSampler
from desed_task.dataio.datasets_atst_sed import StronglyAnnotatedSet, UnlabeledSet, WeakSet
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.nnet.CRNN_e2e import CRNN
from local.scheduler import ExponentialWarmup
from local.classes_dict import classes_labels
from local.stage2_trainer import SEDICT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import warnings
warnings.filterwarnings("ignore")

def separate_opt_params(model):
    # group parameters
    cnn_params = []
    rnn_params = []
    tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for k, p in model.named_parameters():
        if "atst_frame" not in k:
            if "cnn" in k:
                cnn_params.append(p)
            else:
                rnn_params.append(p)
        else:
            if "blocks.0." in k:
                tfm_params[1].append(p)
            elif "blocks.1." in k:
                tfm_params[2].append(p)
            elif "blocks.2." in k:
                tfm_params[3].append(p)
            elif "blocks.3." in k:
                tfm_params[4].append(p)
            elif "blocks.4." in k:
                tfm_params[5].append(p)
            elif "blocks.5." in k:
                tfm_params[6].append(p)
            elif "blocks.6." in k:
                tfm_params[7].append(p)
            elif "blocks.7." in k:
                tfm_params[8].append(p)
            elif "blocks.8" in k:
                tfm_params[9].append(p)
            elif "blocks.9." in k:
                tfm_params[10].append(p)
            elif "blocks.10." in k:
                tfm_params[11].append(p)
            elif "blocks.11." in k:
                tfm_params[12].append(p)
            elif ".norm_frame." in k:
                tfm_params[13].append(p)
            else:
                tfm_params[0].append(p)
    return cnn_params, rnn_params, list(reversed(tfm_params))

def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None
):
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder,
            pad_to=None,
            return_filename=True,
            feat_params=config["feats"]
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    sed_student = CRNN(
        unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"], 
        **config["net"], 
        model_init=config["ultra"]["model_init"],
        atst_init=config["ultra"]["atst_init"],
        atst_dropout=config["ultra"]["atst_dropout"],
        mode="student")
    sed_teacher = CRNN(
        unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"], 
        **config["net"], 
        model_init=config["ultra"]["model_init"],
        atst_dropout=config["ultra"]["atst_dropout"],
        atst_init=config["ultra"]["atst_init"],
        mode="teacher")
    

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )

        strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
        strong_set = StronglyAnnotatedSet(
            config["data"]["strong_folder"],
            strong_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        # If you want to use external dataset, first extracted them and then change the dataset for the strongly-labelled part of the data
        # external_df = pd.read_csv(config["data"]["external_tsv"], sep="\t")
        # external_set = StronglyAnnotatedSet(
        #     config["data"]["external_folder"],
        #     external_df,
        #     encoder,
        #     return_filename=True,
        #     pad_to=config["data"]["audio_max_len"],
        #     feat_params=config["feats"]
        # )
        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        strong_df_val = pd.read_csv(config["data"]["strong_val_tsv"], sep="\t")
        strong_val = StronglyAnnotatedSet(
            config["data"]["strong_folder"],
            strong_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feat_params=config["feats"]
        )
        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            feat_params=config["feats"]
        )
        tot_train_data = [strong_set, synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)
        valid_dataset = torch.utils.data.ConcatDataset([weak_val, synth_val, strong_val])

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                    config["training"]["batch_size"][indx]
                    * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        cnn_params, rnn_params, tfm_params = separate_opt_params(sed_student)

        cnn_param_groups = [
            {"params": cnn_params, "lr": config["opt"]["cnn_lr"]}            
        ]
        rnn_param_groups = [
            {"params": rnn_params, "lr": config["opt"]["rnn_lr"]},
        ]
        tfm_trainable_params = []
        trainable_layers = config["opt"]["tfm_trainable_layers"]
        for i in range(trainable_layers):
            tfm_trainable_params.append(tfm_params[i])
            for p in tfm_params[i]:
                p.requires_grad = True

        init_lr = config["opt"]["tfm_lr"]
        lr_scale = config["opt"]["tfm_lr_scale"]
        scale_lrs = [init_lr * (lr_scale ** i) for i in range(trainable_layers)]
        tfm_param_groups = [ {"params": tfm_trainable_params[i], "lr": scale_lrs[i]} for i in range(len(tfm_trainable_params)) ]
        max_lrs = [ cnn_param_groups[0]["lr"] ] + [ rnn_param_groups[0]["lr"] ] + [ x["lr"] for x in tfm_param_groups ]

        param_groups = cnn_param_groups + rnn_param_groups + tfm_param_groups
        opt = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
        opt = [opt]

        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        tot_steps = config["training"]["n_epochs"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt[0], max_lrs, exp_steps, tot_steps=tot_steps),
            "interval": "step",
        }
        exp_scheduler = [exp_scheduler]
        
        # init opt
        init_lrs = exp_scheduler[0]["scheduler"]._get_lr()
        for i, lr in enumerate(init_lrs):
            opt[0].param_groups[i]["lr"] = lr

        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        if callbacks is None:
            callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="val/obj_metric",
                save_top_k=5,
                mode="max",
                save_last=True,
            ),
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDICT(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
        sed_teacher=sed_teacher
    )


    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    devices=gpus
    accelerator="gpu"

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
        use_distributed_sampler=False
    )
    if test_state_dict is None:

        # start tracking energy consumption
        trainer.fit(desed_training, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)

def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/stage2.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/stage2/",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
             "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
             "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint",
        default=None,
        help="Evaluate the model specified"
    )
    parser.add_argument(
        "--external",
        action="store_true",
        default=False,

    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )

    parser.add_argument(
        "--tfm_lr_scale",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--lr",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--loss_weight",
        default=None,
        type=float,
    )   

    parser.add_argument(
        "--warmup",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--ema",
        default=None,
        type=float,
    )

    args = parser.parse_args(argv)
    args.log_dir += args.prefix
    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    if args.tfm_lr_scale is not None:
        configs["opt"]["tfm_lr_scale"] = args.tfm_lr_scale
    if args.loss_weight is not None:
        configs["training"]["const_max"] = args.loss_weight
    if args.lr is not None:
        configs["opt"]["tfm_lr"] = args.lr
    if args.warmup is not None:
        configs["training"]["n_epochs_warmup"] = args.warmup
    if args.ema is not None:
        configs["training"]["ema_factor"] = args.ema

    evaluation = False 
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        configs_ckpt["training"]["median_window"] = configs["training"]["median_window"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]
        configs = configs_ckpt

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    return configs, args, test_model_state_dict, evaluation

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()

    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )

import gc
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import yaml
from coolname import generate_slug
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, WandbLogger

from src.config import COMP_NAME, CONFIG_PATH, OUTPUT_PATH


def prepare_args(config_path=CONFIG_PATH, default_config="default_run"):
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        help="Configuration scheme",
        default=default_config,
    )

    parser.add_argument(
        "--gpus",
        action="store",
        dest="gpus",
        help="Number of GPUs",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--timestamp",
        action="store",
        dest="timestamp",
        help="Timestamp for versioning",
        default=str(datetime.now().strftime("%Y%m%d-%H%M%S")),
        type=str,
    )

    parser.add_argument(
        "--fold",
        action="store",
        dest="fold",
        help="Fold number",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        help="Random seed",
        default=48,
        type=int,
    )

    parser.add_argument(
        "--slug",
        action="store",
        dest="slug",
        help="Human rememebrable run group",
        default=generate_slug(3),
        type=str,
    )

    parser.add_argument(
        "--logging",
        dest="logging",
        action="store_true",
        help="Flag to log to WandB (on by default)",
    )

    parser.add_argument(
        "--no-logging",
        dest="logging",
        action="store_false",
        help="Flag to prevent logging",
    )
    parser.set_defaults(logging=True)

    args = parser.parse_args()

    # Lookup the config from the YAML file and set args
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if args.config != default_config:
            print("Using", args.config, "configuration")

        for k, v in cfg[args.config].items():
            setattr(args, k, v)

    return args


def resume_helper(args):
    """
    To resume a run, add this to the YAML/args:

    checkpoint: "20210510-161949"
    wandb_id: 3j79kxq6

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    if hasattr(args, "checkpoint"):
        paths = (
            OUTPUT_PATH / args.checkpoint / args.encoder / f"fold_{args.fold - 1}"
        ).glob("*.*loss.ckpt")
        resume = list(paths)[0]

        if hasattr(args, "wandb_id"):
            run_id = args.wandb_id
        else:
            print("No wandb_id provided. Logging as new run")
            run_id = None
    else:
        resume = None
        run_id = None

    return resume, run_id


def prepare_loggers_and_callbacks(
    timestamp,
    encoder_name,
    fold,
    monitors=[],
    patience=None,
    tensorboard=False,
    wandb=False,
    neptune=False,
    run_id=None,
    save_weights_only=False,
):
    """
    Utility function to prepare loggers and callbacks

    Args:
        timestamp (str): Timestamp for folder name
        encoder_name (str): encoder_name for folder name
        fold (int): Fold number for folder nesting
        monitors (list, optional): For multiple monitors for ModelCheckpoint.
        patience (int, optional): patience for EarlyStopping
        List of tuples in form [(monitor, mode, suffix), ...],
        Defaults to [].
        tensorboard (bool): Flag to use Tensorboard logger
        wandb (bool): Flag to use Weight and Biases logger
        neptune (bool): Flag to use Neptune logger

    Returns:
        [type]: [description]
    """
    save_path = OUTPUT_PATH / timestamp

    callbacks, loggers = {}, {}

    callbacks["lr"] = LearningRateMonitor(logging_interval="step")

    if "/" in encoder_name:
        encoder_name = encoder_name.replace("/", "_")

    if patience:
        callbacks["early_stopping"] = EarlyStopping("loss/valid", patience=patience)

    for monitor, mode, suffix in monitors:

        if suffix is not None and suffix != "":
            filename = "{epoch:02d}-{f2:.4f}" + f"_{suffix}"
        elif len(monitors) == 1:
            filename = "{epoch:02d}"
        else:
            filename = "{epoch:02d}-{f2:.4f}"

        checkpoint = ModelCheckpoint(
            dirpath=save_path / encoder_name / f"fold_{fold}",
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_weights_only=save_weights_only,
            save_last=True,
        )
        callbacks[f"checkpoint_{monitor}"] = checkpoint

    if tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=save_path,
            name=encoder_name,
            version=f"fold_{fold}",
        )
        loggers["tensorboard"] = tb_logger

    if wandb:
        wandb_logger = WandbLogger(
            name=f"{timestamp}/fold{fold}",
            save_dir=OUTPUT_PATH,
            project=COMP_NAME,
            id=run_id,
        )
        loggers["wandb"] = wandb_logger

    if neptune:
        neptune_logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project_name=f"anjum48/{COMP_NAME}",
            experiment_name=f"{timestamp}-fold{fold}",
        )
        loggers["neptune"] = neptune_logger

    return loggers, callbacks


def memory_cleanup():
    """
    Cleans up GPU memory. Call after a fold is trained.
    https://github.com/huggingface/transformers/issues/1742
    """
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj
    gc.collect()
    torch.cuda.empty_cache()


# https://github.com/rwightman/pytorch-image-models/blob/ddc29da974023416ac2bf2468a80a18438c0090d/timm/optim/optim_factory.py#L31-L43
def add_weight_decay(
    model,
    weight_decay=1e-5,
    skip_list=("bias", "bn", "LayerNorm.bias", "LayerNorm.weight"),
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def mixup_data(x, y, alpha=1.0, return_idx=False):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, requires_grad=False).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    if return_idx:
        return mixed_x, y_a, y_b, lam, index
    else:
        return mixed_x, y_a, y_b, lam


class LogSummaryCallback(Callback):
    def __init__(self, metric_name, summary_type="min"):
        super().__init__()
        self.metric_name = metric_name
        self.summary_type = summary_type
        self.best_value = torch.inf if summary_type == "min" else -torch.inf

    def on_validation_epoch_end(self, trainer, pl_module):
        metric_value = trainer.callback_metrics[self.metric_name]

        if self.summary_type == "min" and metric_value < self.best_value:
            self.best_value = metric_value
        elif self.summary_type == "max" and metric_value > self.best_value:
            self.best_value = metric_value
        else:
            pass

        pl_module.log(f"{self.metric_name}_{self.summary_type}", self.best_value)


def mixup_data_multiobjective(x, y1, y2, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, requires_grad=False).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    return mixed_x, y1_a, y1_b, y2_a, y2_b, lam


# https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L227-L238
# https://arxiv.org/pdf/1905.04899.pdf
def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.shape[0]).to(x.device)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

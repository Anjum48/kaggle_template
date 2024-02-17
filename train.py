import wandb
import hydra
import lightning as L
import torch
from coolname import generate_slug
from omegaconf import DictConfig, open_dict

from src.config import MODEL_CACHE
from src.datasets import HMSDataModule, postprocess
from src.models import HMSModel
from src.utils import LogSummaryCallback, get_num_steps, prepare_loggers_and_callbacks

torch.set_float32_matmul_precision("high")
torch.hub.set_dir(MODEL_CACHE)
slug = generate_slug(3)


# @hydra.main(config_path="conf", config_name="config", version_base=None)
def run_fold(cfg: DictConfig):
    L.seed_everything(cfg.run.seed + cfg.run.fold, workers=True)

    resume, run_id = None, None
    monitor_list = [("loss/valid", "min", "loss")]

    loggers, callbacks = prepare_loggers_and_callbacks(
        fold=cfg.run.fold,
        run_name=cfg.run.run_name,
        output_dir=cfg.run.output_dir,
        monitors=monitor_list,
        tensorboard=cfg.run.logging,
        wandb=cfg.run.logging,
        run_id=run_id,
        save_weights_only=False,
    )

    callbacks["metric_summary"] = LogSummaryCallback("loss/valid", "min")

    dm = HMSDataModule(**cfg.model)
    dm.setup("fit", cfg.run.fold)

    model = HMSModel(
        T_max=get_num_steps(cfg, dm), **cfg.model, **cfg.trainer, **cfg.run
    )

    trainer = L.Trainer(
        logger=list(loggers.values()),
        callbacks=list(callbacks.values()),
        **cfg.trainer,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=resume)
    trainer.predict(model, datamodule=dm, return_predictions=False)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_cv(cfg: DictConfig):
    """Run a full cross-validation run.
    The fold number in run.fold is overidden here
    The parent MLFlow run is here, and the child runs are in run_fold()

    Args:
        cfg (DictConfig): OmegaConf config from Hydra
    """

    # fold_scores = []

    for f in range(cfg.run.n_folds):
        with open_dict(cfg):
            cfg.run.fold = f  # Set the fold number in cfg

            if cfg.run.run_name is None:
                cfg.run.run_name = slug

            run_fold(cfg)
            wandb.finish()

    y_preds, y_trues = postprocess()
    print(y_preds.shape, y_trues.shape)


if __name__ == "__main__":
    run_cv()

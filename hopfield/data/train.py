import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import os
import sys
os.environ["WANDB_MODE"] = "disabled"

# ============================================================
# Add project root to path so src.* imports work
# ============================================================
PROJECT_ROOT = "/system/user/studentwork/tscheidl/MHNfs"
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import FSMolDataModule
from src.mhnfs.models import MHNfs


@hydra.main(config_path="/system/user/studentwork/tscheidl/MHNfs/src/mhnfs/configs",
            config_name="cfg",
            version_base=None)
def train(cfg):
    """
    Training loop for MHNfs on FS-Mol.
    """
    # Set seed
    seed_everything(cfg.training.seed)

    # Load data module
    dm = FSMolDataModule(cfg)

    # Load model
    model = MHNfs(cfg)

    # Move model to device
    device = cfg.system.ressources.device
    model = model.to(device)

    # --------------------------------------------------------
    # Logger (wandb)
    # --------------------------------------------------------
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = pl_loggers.WandbLogger(
        save_dir=log_dir,
        name=cfg.experiment_name,
        project=cfg.project_name,
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    checkpoint_dauprc_val = ModelCheckpoint(
        monitor="dAUPRC_val", mode="max", save_top_k=1
    )
    checkpoint_dauprc_val_ma = ModelCheckpoint(
        monitor="dAUPRC_val_ma", mode="max", save_top_k=1
    )
    checkpoint_dauprc_delta = ModelCheckpoint(
        monitor="dAUPRC_train_val_delta", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stopping = EarlyStopping(
        monitor="dAUPRC_val_ma",
        patience=10,
        mode="max",
    )

    early_stopping_raw = EarlyStopping(
        monitor="dAUPRC_val",
        patience=5,
        mode="max",
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_dauprc_val,
            checkpoint_dauprc_val_ma,
            checkpoint_dauprc_delta,
            lr_monitor,
            early_stopping,
            EarlyStopping(monitor="dAUPRC_val", patience=5, mode="max"),
        ],
        max_epochs=cfg.training.epochs,
        # accumulate_grad_batches=5,
        reload_dataloaders_every_n_epochs=1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
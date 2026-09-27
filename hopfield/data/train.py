import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import os
import sys
os.environ["WANDB_MODE"] = "disabled"

# ============================================================
# Project setup
# ============================================================
# Add the project root to the Python path so that src.* modules
# can be imported when the training script is executed.
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
    # Set the random seed for reproducible training.
    seed_everything(cfg.training.seed)

    # Initialize the FS-Mol data module.
    # This handles loading and preparing the training, validation,
    # and test data.
    dm = FSMolDataModule(cfg)

    # Initialize the MHNfs model using the configuration.
    model = MHNfs(cfg)

    # Move the model to the configured device (GPU or CPU).
    device = cfg.system.ressources.device
    model = model.to(device)

    # --------------------------------------------------------
    # Experiment logging
    # --------------------------------------------------------
    # Store experiment logs in the project log directory.
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize the Weights & Biases logger for experiment tracking.
    logger = pl_loggers.WandbLogger(
        save_dir=log_dir,
        name=cfg.experiment_name,
        project=cfg.project_name,
    )

    # --------------------------------------------------------
    # Training callbacks
    # --------------------------------------------------------
    
    # Save the checkpoint with the highest validation dAUPRC.
    checkpoint_dauprc_val = ModelCheckpoint(
        monitor="dAUPRC_val",
        mode="max",
        save_top_k=1,
        dirpath="best_checkpoints/runB6_16shot",
        filename="best_raw-{epoch:02d}-{dAUPRC_val:.4f}",
    )

    # Alternative checkpoint configuration that would save
    # multiple checkpoints based on raw validation dAUPRC.
    #checkpoint_dauprc_val = ModelCheckpoint(
    #    monitor="dAUPRC_val", mode="max", save_top_k=5
    #)
    
    # Save the checkpoint with the highest moving-average
    # validation dAUPRC.
    checkpoint_dauprc_val_ma = ModelCheckpoint(
        monitor="dAUPRC_val_ma",
        mode="max",
        save_top_k=1,
        dirpath="best_checkpoints/runB6_16shot",
        filename="best-{epoch:02d}-{dAUPRC_val_ma:.4f}",
    )

    # Alternative checkpoint configuration that would save
    # multiple checkpoints based on the moving-average metric.
    #checkpoint_dauprc_val_ma = ModelCheckpoint(
    #    monitor="dAUPRC_val_ma", mode="max", save_top_k=5
    #)

    # Track the learning rate after each training epoch.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Stop training if the moving-average validation dAUPRC
    # does not improve for 30 epochs.
    early_stopping = EarlyStopping(
        monitor="dAUPRC_val_ma",
        patience=30,
        mode="max",
    )

    # Stop training based on the raw validation dAUPRC
    # if it does not improve for 30 epochs.
    early_stopping_raw = EarlyStopping(
        monitor="dAUPRC_val",
        patience=30,
        mode="max",
    )

    # --------------------------------------------------------
    # PyTorch Lightning trainer
    # --------------------------------------------------------
    # Configure the training process, including the device,
    # callbacks, number of epochs, and gradient accumulation.
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_dauprc_val,
            checkpoint_dauprc_val_ma,
            lr_monitor,
            early_stopping,
            EarlyStopping(monitor="dAUPRC_val", patience=30, mode="max"),
        ],
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
    )

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------
    # Start the training process using the MHNfs model and
    # the FS-Mol data module.
    trainer.fit(model, dm)


if __name__ == "__main__":
    # Run the training function when this file is executed directly.
    train()

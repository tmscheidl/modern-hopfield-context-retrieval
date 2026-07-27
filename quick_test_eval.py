# quick_test_eval.py
import sys, os
sys.path.insert(0, "/system/user/studentwork/tscheidl/MHNfs")
os.environ["WANDB_MODE"] = "disabled"
import hydra
import pytorch_lightning as pl

@hydra.main(config_path="/system/user/studentwork/tscheidl/MHNfs/src/mhnfs/configs",
            config_name="cfg", version_base=None)
def run_test(cfg):
    from src.data.dataloader import FSMolDataModule
    from src.mhnfs.models import MHNfs

    dm = FSMolDataModule(cfg)
    dm.setup()

    model = MHNfs.load_from_checkpoint(
        "/system/user/studentwork/tscheidl/MHNfs/best_checkpoints/v28_retrain/best-epoch=22-dAUPRC_val_ma=0.2575.ckpt",  # update after training
        cfg=cfg
    )
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    run_test()
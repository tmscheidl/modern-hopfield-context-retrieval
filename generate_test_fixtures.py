"""
Run this ONCE on your cluster (where the real data + checkpoints live) to
generate the reference fixtures that test_model_inference.py checks against.

Usage:
    python3 test/generate_test_fixtures.py \
        checkpoint_path=/path/to/your/v28_best_checkpoint.ckpt

After running, commit/keep these three files:
    test/assets/test_reference_data/model_input_batch.pt
    test/assets/test_reference_data/model_predictions.pt
    test/assets/mhnfs_data/cfg_snapshot.yaml
and copy the checkpoint itself to:
    test/assets/mhnfs_data/reference_checkpoint.ckpt

Re-run this script (and re-commit the outputs) whenever you deliberately want
to update what "correct" behavior means -- e.g. if you pick a new best
checkpoint. Until you do, test_model_inference.py will fail loudly if
anything changes the model's actual output, which is the point.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import os
import sys
import shutil
import torch
import hydra
from omegaconf import OmegaConf

PROJECT_ROOT = "/system/user/studentwork/tscheidl/MHNfs"
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import FSMolDataModule
from src.mhnfs.models import MHNfs

OUT_DIR = os.path.join(PROJECT_ROOT, "test", "assets")


@hydra.main(
    config_path="/system/user/studentwork/tscheidl/MHNfs/src/mhnfs/configs",
    config_name="cfg",
    version_base=None,
)
def generate(cfg):
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "Pass checkpoint_path=/abs/path/to/checkpoint.ckpt on the command line."
        )

    os.makedirs(os.path.join(OUT_DIR, "test_reference_data"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "mhnfs_data"), exist_ok=True)

    # --------------------------------------------------------
    # Load model from the checkpoint you want to freeze as "correct"
    # --------------------------------------------------------
    model = MHNfs.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # --------------------------------------------------------
    # Grab one real validation batch
    # --------------------------------------------------------
    dm = FSMolDataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }

    # --------------------------------------------------------
    # Run inference once, save inputs + outputs
    # --------------------------------------------------------
    with torch.no_grad():
        predictions = model(batch, use_fixed_context=True).detach().cpu()

    batch_cpu = {
        k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()
    }

    torch.save(batch_cpu, os.path.join(OUT_DIR, "test_reference_data", "model_input_batch.pt"))
    torch.save(predictions, os.path.join(OUT_DIR, "test_reference_data", "model_predictions.pt"))

    # Freeze the exact config used, so future config edits don't silently
    # invalidate this fixture without anyone noticing.
    OmegaConf.save(cfg, os.path.join(OUT_DIR, "mhnfs_data", "cfg_snapshot.yaml"))

    # Copy the checkpoint itself alongside the fixtures.
    shutil.copy(checkpoint_path, os.path.join(OUT_DIR, "mhnfs_data", "reference_checkpoint.ckpt"))

    print(f"Saved reference batch, predictions, cfg snapshot, and checkpoint to {OUT_DIR}")
    print(f"Predictions shape: {predictions.shape}")


if __name__ == "__main__":
    generate()
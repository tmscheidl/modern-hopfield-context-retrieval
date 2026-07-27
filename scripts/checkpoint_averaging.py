"""
Checkpoint averaging (SWA-style) for V21.
Averages weights from epoch=13 (best raw dAUPRC_val) and epoch=22
(part of the best moving-average window), then re-runs validation
to see if the averaged model beats either individual checkpoint.

Zero risk to existing checkpoints - this only reads them and creates
a new averaged model in memory to evaluate.
"""
import sys
import torch
import copy

PROJECT_ROOT = "/system/user/studentwork/tscheidl/MHNfs"
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.mhnfs.models import MHNfs
from src.data.dataloader import FSMolDataModule
import pytorch_lightning as pl

CKPT_PATHS = [
    f"{PROJECT_ROOT}/logs/MHNfs/6cSNymmXNiBF9v4Vh8pout/checkpoints/epoch=13-step=15540.ckpt",
    f"{PROJECT_ROOT}/logs/MHNfs/6cSNymmXNiBF9v4Vh8pout/checkpoints/epoch=22-step=25530.ckpt",
]

cfg = OmegaConf.load(f"{PROJECT_ROOT}/src/mhnfs/configs/cfg.yaml")

print("Loading checkpoints...")
state_dicts = []
for path in CKPT_PATHS:
    ckpt = torch.load(path, map_location="cpu")
    state_dicts.append(ckpt["state_dict"])
    print(f"  Loaded: {path.split('/')[-1]}")

print("\nAveraging weights...")
avg_state_dict = copy.deepcopy(state_dicts[0])
for key in avg_state_dict.keys():
    if avg_state_dict[key].dtype.is_floating_point:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = stacked.mean(dim=0)
    # non-float buffers (if any) just keep the first checkpoint's value

print("Building model and loading averaged weights...")
model = MHNfs(cfg)
model.load_state_dict(avg_state_dict)

print("\nRunning validation on the averaged model...")
dm = FSMolDataModule(cfg)
dm.setup()

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)
results = trainer.validate(model, datamodule=dm)
print("\n=== Averaged model validation results ===")
print(results)
import sys, os
sys.path.insert(0, "/system/user/studentwork/tscheidl/MHNfs")
os.environ["WANDB_MODE"] = "disabled"
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="/system/user/studentwork/tscheidl/MHNfs/src/mhnfs/configs",
            config_name="cfg", version_base=None)
def check(cfg):
    from src.mhnfs.models import MHNfs
    model = MHNfs(cfg)
    result = model.configure_optimizers()
    print("=== configure_optimizers() output ===")
    print(type(result))
    if isinstance(result, dict):
        print("optimizer:", result["optimizer"])
        print("lr_scheduler config:", result["lr_scheduler"])
    else:
        print("optimizer only, no scheduler dict returned:", result)

if __name__ == "__main__":
    check()

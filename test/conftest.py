"""
Needed objects for tests.

Adapted from the professor's conftest.py. The key difference: our MHNfs.forward()
takes a single `batch` dict (plus use_fixed_context), not five/seven positional
tensors, so fixtures are structured around one saved batch dict rather than
separate query/actives/inactives tensors.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import pytest
import torch
from omegaconf import OmegaConf
from src.mhnfs.models import MHNfs

#---------------------------------------------------------------------------------------
# Define fixtures
#---------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_input_batch():
    """
    A single validation batch (dict of tensors), saved once via
    test/generate_test_fixtures.py from a real dataloader.
    """
    current_loc = __file__.rsplit("/", 2)[0]
    return torch.load(
        current_loc + "/test/assets/test_reference_data/model_input_batch.pt"
    )


@pytest.fixture(scope="session")
def model_predictions():
    """
    Reference predictions produced by the checkpointed model on
    `model_input_batch`, saved at the same time as the batch.
    """
    current_loc = __file__.rsplit("/", 2)[0]
    return torch.load(
        current_loc + "/test/assets/test_reference_data/model_predictions.pt"
    )


@pytest.fixture(scope="session")
def model_cfg():
    """
    The exact cfg.yaml used to produce the reference checkpoint/predictions.
    Keep a frozen copy under test/assets/ rather than pointing at the live
    configs/cfg.yaml, so future config edits don't silently break this test.
    """
    current_loc = __file__.rsplit("/", 2)[0]
    return OmegaConf.load(current_loc + "/test/assets/mhnfs_data/cfg_snapshot.yaml")


@pytest.fixture(scope="session")
def model_trainingClass(model_cfg):
    """
    The trained MHNfs LightningModule, loaded from the checkpoint used to
    produce the reference predictions (e.g. your best v28 checkpoint).
    """
    current_loc = __file__.rsplit("/", 2)[0]
    model = MHNfs.load_from_checkpoint(
        current_loc + "/test/assets/mhnfs_data/reference_checkpoint.ckpt",
        cfg=model_cfg,
    )
    return model
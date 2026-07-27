"""
Regression test: verifies that the checkpointed MHNfs model still produces the
same predictions on a fixed reference batch as it did when the reference
predictions were generated (see test/generate_test_fixtures.py).

This guards against silent behavior changes from refactors, dependency
upgrades, or accidental config drift -- it does not test correctness of the
model itself, only that its behavior hasn't changed unexpectedly.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import torch

#---------------------------------------------------------------------------------------
# Define tests


def _move_batch(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


class TestMHNfsInference:

    def test_mhnfs_prediction_on_cpu(
        self, model_input_batch, model_predictions, model_trainingClass
    ):
        model = model_trainingClass.to("cpu")
        model.eval()

        batch = _move_batch(model_input_batch, "cpu")

        with torch.no_grad():
            predictions = model(batch, use_fixed_context=True).detach().cpu()

        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.0)

    def test_mhnfs_prediction_on_gpu(
        self, model_input_batch, model_predictions, model_trainingClass
    ):
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("CUDA not available on this machine.")

        model = model_trainingClass.to("cuda")
        model.eval()

        batch = _move_batch(model_input_batch, "cuda")

        with torch.no_grad():
            predictions = model(batch, use_fixed_context=True).detach().cpu()

        assert torch.allclose(predictions, model_predictions, atol=0.01, rtol=0.0)


#---------------------------------------------------------------------------------------
# debugging
if __name__ == "__main__":
    print(
        "Run via pytest, e.g.:\n"
        "    pytest test/test_model_inference.py -v\n"
        "This file needs fixtures from conftest.py to run standalone."
    )
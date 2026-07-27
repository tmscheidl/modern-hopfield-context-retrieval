# MHNfs/test — regression + unit tests

## Layout

```
MHNfs/
├── src/
│   └── metrics/
│       └── performance_metrics.py   # standalone metric functions (new)
└── test/
    ├── conftest.py                  # pytest fixtures
    ├── test_metrics.py              # unit tests for AUC / dAUPRC math
    ├── test_model_inference.py      # regression test: model output unchanged
    ├── generate_test_fixtures.py    # one-time script to create reference data
    └── assets/
        ├── test_reference_data/
        │   ├── model_input_batch.pt     (generated)
        │   └── model_predictions.pt     (generated)
        └── mhnfs_data/
            ├── cfg_snapshot.yaml        (generated)
            └── reference_checkpoint.ckpt (generated, copy of your chosen checkpoint)
```

## One-time setup (on the cluster, where data + checkpoints live)

1. Place `src/metrics/performance_metrics.py` in your source tree.
2. Refactor `MHNfs.on_validation_epoch_end()` in `models.py` to call
   `compute_dauprc_score(...)` from that new module instead of computing
   dAUPRC inline. (Behavior should be identical -- this is a pure refactor
   for testability.)
3. Run the fixture generator once, pointing at your best checkpoint
   (e.g. the v28 run):

   ```bash
   python3 test/generate_test_fixtures.py \
       checkpoint_path=/absolute/path/to/v28_best.ckpt
   ```

   This creates all four files under `test/assets/`.

## Running the tests

```bash
cd /system/user/studentwork/tscheidl/MHNfs
pytest test/ -v
```

- `test_metrics.py` needs no fixtures/checkpoints -- pure math, runs anywhere,
  fast.
- `test_model_inference.py` needs the four generated files above. It loads
  the frozen checkpoint + frozen batch and asserts predictions still match
  within `atol=0.01`.

## When to regenerate fixtures

Only when you deliberately pick a **new** reference/best checkpoint (e.g. you
find something that beats v28 and want to make that the new baseline).
Otherwise, if `test_model_inference.py` starts failing, that's the point --
it means something (a refactor, a library upgrade, an accidental config
edit) silently changed the model's behavior, and you should find out why
before trusting any new training numbers.

## Notes on adapting from the professor's original tests

The professor's `conftest.py` / `test_model_inference.py` /
`test_model_training_class.py` assume a `forward()` signature with several
positional tensor arguments (query, support actives, support inactives,
sizes, masks). Our `MHNfs.forward()` takes a single `batch` dict plus
`use_fixed_context`, so those fixtures and tests were rewritten rather than
copied directly -- the *pattern* (freeze inputs + outputs from a verified
checkpoint, assert future runs match) is the same, but the concrete fixture
shapes and call signatures differ because our model's API differs.
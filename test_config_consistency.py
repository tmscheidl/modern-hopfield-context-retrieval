# test/test_config_consistency.py
"""
Guards against silent config/behavior mismatches -- the kind that cost
real debugging time earlier in this project (e.g. a hardcoded top_k=64
silently overriding cfg.model.context_top_k=512).

This doesn't test correctness, just that the values documented in the
thesis comparison table are actually the values in effect.
"""
from omegaconf import OmegaConf


class TestConfigConsistency:

    def test_key_hyperparameters_match_reported_values(self, model_cfg):
        cfg = model_cfg
        assert cfg.model.associationSpace_dim == 1024
        assert abs(cfg.model.prediction_scaling - 0.044194173824159216) < 1e-9
        assert abs(cfg.model.hopfield.beta - 0.044) < 1e-9  # 0.044194173824159216
        assert cfg.model.hopfield.dropout == 0.5
        assert cfg.model.transformer.num_layers == 1
        assert cfg.model.similarityModule.numHeads == 1
        assert cfg.model.similarityModule.scaling == "1/N"
        assert cfg.model.context.ratio_training_molecules == 0.05
        assert cfg.training.learning_rate == 0.0001
        assert cfg.training.weight_decay == 0.0
        assert cfg.training.batch_size == 32
        assert cfg.training.accumulate_grad_batches == 16  # effective batch = 512

    def test_context_module_top_k_is_wired_correctly(self, model_trainingClass):
        # Regression guard for the specific bug where ContextModule(cfg, top_k=64)
        # silently overrode cfg.model.context_top_k.
        assert model_trainingClass.context_module.top_k == 512

    def test_similarity_module_has_no_learnable_projections(self, model_trainingClass):
        # Regression guard for input_dim=dim vs input_dim=None -- confirms
        # the parameter-free similarity design (matching the professor's
        # fixed dot-product) is actually in effect.
        assert model_trainingClass.similarity_module.query_proj is None
        assert model_trainingClass.similarity_module.support_proj is None
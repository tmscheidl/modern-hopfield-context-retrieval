"""
Tests that AUC and delta-AUPRC metrics are computed as expected.

Adapted from the professor's test_metrics.py, pointed at our own
standalone src/metrics/performance_metrics.py implementation.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import torch
from src.metrics.performance_metrics import compute_auc_score, compute_dauprc_score

#---------------------------------------------------------------------------------------
# Define tests


class TestMetrics:

    def test_compute_auc_score(self):

        # Test 1:
        # Single task setting: Perfect classifier
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])

        auc = 1.0
        computed_auc = compute_auc_score(predictions, labels, target_ids)[0]
        assert auc == computed_auc

        # Test 2:
        # Single task setting: Random classifier
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])

        auc = 0.5
        computed_auc = compute_auc_score(predictions, labels, target_ids)[0]
        assert auc == computed_auc

        # Test 3:
        # Multi-task setting: 2 tasks, perfect on the first, random on the second
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1,
                                0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1])

        idx = torch.randperm(predictions.shape[0])
        predictions = predictions[idx]
        labels = labels[idx]
        target_ids = target_ids[idx]

        aucs = [1.0, 0.5]
        mean_auc = 0.75

        computed_mean_auc, computed_aucs, _ = compute_auc_score(
            predictions, labels, target_ids
        )

        assert mean_auc == computed_mean_auc
        assert aucs == computed_aucs

    def test_compute_dauprc_score(self):

        # Test 1:
        # Single task setting: Perfect classifier
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])

        dauprc = 0.5
        computed_dauprc = compute_dauprc_score(predictions, labels, target_ids)[0]
        assert dauprc == computed_dauprc

        # Test 2:
        # Single task setting: Random classifier
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0])

        dauprc = 0.0
        computed_dauprc = compute_dauprc_score(predictions, labels, target_ids)[0]
        assert dauprc == computed_dauprc

        # Test 3:
        # Multi-task setting: 2 tasks, perfect on the first, random on the second
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        labels = torch.tensor([0, 0, 0, 1, 1, 1,
                                0, 0, 0, 1, 1, 1])
        target_ids = torch.tensor([0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1])

        idx = torch.randperm(predictions.shape[0])
        predictions = predictions[idx]
        labels = labels[idx]
        target_ids = target_ids[idx]

        dauprcs = [0.5, 0.0]
        mean_dauprc = 0.25

        computed_mean_dauprc, computed_dauprcs, _ = compute_dauprc_score(
            predictions, labels, target_ids
        )

        assert mean_dauprc == computed_mean_dauprc
        assert dauprcs == computed_dauprcs


#---------------------------------------------------------------------------------------
# debugging
if __name__ == "__main__":
    test = TestMetrics()
    test.test_compute_auc_score()
    test.test_compute_dauprc_score()
    print("All metric tests passed.")
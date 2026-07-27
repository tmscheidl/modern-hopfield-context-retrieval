"""
Standalone performance metric functions for MHNfs.

Extracted from the inline logic previously living in
MHNfs.on_validation_epoch_end(), so it can be:
  1) unit tested independently of the full training loop, and
  2) reused identically between validation, test, and any offline analysis.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _group_by_task(predictions, labels, target_ids):
    predictions = _to_numpy(predictions).reshape(-1)
    labels = _to_numpy(labels).reshape(-1)
    target_ids = _to_numpy(target_ids).reshape(-1)

    per_task = {}
    for p, l, t in zip(predictions, labels, target_ids):
        tid = int(t)
        per_task.setdefault(tid, {"probs": [], "labels": []})
        per_task[tid]["probs"].append(p)
        per_task[tid]["labels"].append(l)
    return per_task


def compute_auc_score(predictions, labels, target_ids):
    """
    Compute per-task ROC-AUC and the mean across tasks.

    Tasks with only one class present (all-active or all-inactive) are
    skipped, since AUC is undefined for them.

    Returns
    -------
    mean_auc : float
    aucs : list[float]          per-task AUC, in ascending task_id order
    task_ids : list[int]        task ids corresponding to `aucs`
    """
    per_task = _group_by_task(predictions, labels, target_ids)

    aucs, task_ids = [], []
    for tid in sorted(per_task.keys()):
        l = np.array(per_task[tid]["labels"])
        p = np.array(per_task[tid]["probs"])
        if 0 < l.sum() < len(l):
            aucs.append(float(roc_auc_score(l, p)))
            task_ids.append(tid)

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return mean_auc, aucs, task_ids


def compute_dauprc_score(predictions, labels, target_ids):
    """
    Compute per-task delta-AUPRC (AUPRC minus the random-classifier
    baseline, i.e. the positive rate) and the mean across tasks.

    Tasks with only one class present are skipped, matching
    compute_auc_score's behavior.

    Returns
    -------
    mean_dauprc : float
    dauprcs : list[float]       per-task delta-AUPRC, in ascending task_id order
    task_ids : list[int]        task ids corresponding to `dauprcs`
    """
    per_task = _group_by_task(predictions, labels, target_ids)

    dauprcs, task_ids = [], []
    for tid in sorted(per_task.keys()):
        l = np.array(per_task[tid]["labels"])
        p = np.array(per_task[tid]["probs"])
        if 0 < l.sum() < len(l):
            baseline = l.mean()
            auprc = average_precision_score(l, p)
            dauprcs.append(float(auprc - baseline))
            task_ids.append(tid)

    mean_dauprc = float(np.mean(dauprcs)) if dauprcs else 0.0
    return mean_dauprc, dauprcs, task_ids
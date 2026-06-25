import sys
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = "/system/user/studentwork/tscheidl/MHNfs"
sys.path.append(PROJECT_ROOT)

from src.mhnfs.cross_attention_module import CrossAttentionModule
from src.mhnfs.context_module import ContextModule
from src.mhnfs.similarity_module import SimilarityModule
from src.mhnfs.mhnfs_final_model import MHNfsFinalModel

import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

MOL_INPUTS_PATH = "/system/user/studentwork/tscheidl/MHNfs/src/data/preprocessed/training/mol_inputs.npy"


class MHNfs(pl.LightningModule):
    """
    MHNfs: Modern Hopfield Network for few-shot molecular activity prediction.
    Wraps CrossAttentionModule + ContextModule + SimilarityModule into a
    PyTorch Lightning module for training on FS-Mol.

    Context set follows the paper:
    - Training: randomly sample 5% of training molecules per batch
    - Validation/Test: fixed 5% subset (same seed)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dim = cfg.model.associationSpace_dim

        # --------------------------------------------------------
        # Input projection: mol_inputs (2248-dim) -> model dim
        # --------------------------------------------------------
        hidden_dim = cfg.model.encoder.number_hidden_neurons
        input_dropout = cfg.model.encoder.input_dropout
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.model.mol_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(input_dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        # --------------------------------------------------------
        # Submodules
        # --------------------------------------------------------
        #self.cross_attention = CrossAttentionModule(cfg)
        self.cross_attention = CrossAttentionModule(cfg, num_layers=cfg.model.transformer.num_layers)
        self.context_module = ContextModule(cfg, top_k=64)
        self.similarity_module = SimilarityModule(cfg, input_dim=dim)

        # --------------------------------------------------------
        # Final model
        # --------------------------------------------------------
        self.model = MHNfsFinalModel(
            cross_attention=self.cross_attention,
            context_module=self.context_module,
            similarity_module=self.similarity_module,
            prediction_scaling=cfg.model.prediction_scaling,
        )

        # --------------------------------------------------------
        # Load all training molecules into CPU memory once
        # Context is sampled from this on each batch
        # --------------------------------------------------------
        self._mol_inputs_all = torch.tensor(
            np.load(MOL_INPUTS_PATH),
            dtype=torch.float32,
        )  # [N, 2248] on CPU

        n_total = len(self._mol_inputs_all)
        self._n_context = int(cfg.model.context.ratio_training_molecules * n_total)

        # --------------------------------------------------------
        # Fixed context for validation/test (same seed as paper)
        # Pre-compute once and store
        # --------------------------------------------------------
        rng = np.random.default_rng(42)
        fixed_idx = rng.choice(n_total, size=self._n_context, replace=False)
        self.register_buffer(
            "_fixed_context_raw",
            self._mol_inputs_all[fixed_idx],  # [n_context, 2248]
        )

        # --------------------------------------------------------
        # Placeholder context embedding (will be set in forward)
        # --------------------------------------------------------
        self.context_embedding = None

        # --------------------------------------------------------
        # Loss
        # --------------------------------------------------------
        self.loss_fn = nn.BCEWithLogitsLoss()

        # --------------------------------------------------------
        # Tracking
        # --------------------------------------------------------
        self._val_dauprc_history = []
        self._val_outputs = []

        self._train_context_emb = None

    # --------------------------------------------------------
    # Called from train.py before training — no-op now since
    # context is handled per-batch in forward()
    # --------------------------------------------------------
    def _update_context_set_embedding(self):
        pass

    # --------------------------------------------------------
    # Sample random context for one batch (training)
    # --------------------------------------------------------
    def _get_fresh_train_context(self):
        idx = torch.randperm(len(self._mol_inputs_all))[:self._n_context]
        context_raw = self._mol_inputs_all[idx].to(self.device)
        with torch.no_grad():
            return self.input_projection(context_raw)

    # --------------------------------------------------------
    # Get fixed context for validation/test
    # --------------------------------------------------------
    def _get_fixed_context(self):
        context_raw = self._fixed_context_raw.to(self.device)  # [n_context, 2248]
        with torch.no_grad():
            context_emb = self.input_projection(context_raw)  # [n_context, dim]
        return context_emb

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, batch, use_fixed_context=False):
        query = self.input_projection(batch["queryMolecule"])
        actives = self.input_projection(batch["supportSetActives"])
        inactives = self.input_projection(batch["supportSetInactives"])

        # Use precomputed masks if available (training with dropout)
        # otherwise compute fresh masks (validation)
        if "_act_mask" in batch:
            act_mask = batch["_act_mask"]
            inact_mask = batch["_inact_mask"]
        else:
            act_size = batch["supportSetActivesSize"]
            inact_size = batch["supportSetInactivesSize"]
            B, Na, _ = actives.shape
            _, Ni, _ = inactives.shape
            act_mask = torch.arange(Na, device=self.device).unsqueeze(0) < act_size.unsqueeze(1)
            inact_mask = torch.arange(Ni, device=self.device).unsqueeze(0) < inact_size.unsqueeze(1)

        B, Na, _ = actives.shape
        _, Ni, _ = inactives.shape

        if use_fixed_context:
            context_emb = self._get_fixed_context()
        else:
            context_emb = self._get_fresh_train_context()

        context = context_emb.unsqueeze(0).expand(B, -1, -1)

        logits = self.model(
            query=query,
            support_actives=actives,
            support_inactives=inactives,
            mask_actives=act_mask,
            mask_inactives=inact_mask,
            context_memory=context,
        )
        return logits

    def on_train_epoch_start(self):
        """Resample context at the start of each training epoch."""
        idx = torch.randperm(len(self._mol_inputs_all))[:self._n_context]
        context_raw = self._mol_inputs_all[idx].to(self.device)
        with torch.no_grad():
            self._train_context_emb = self.input_projection(context_raw)

    # --------------------------------------------------------
    # Training step — random context per batch
    # --------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # Build masks
        act_size = batch["supportSetActivesSize"]
        inact_size = batch["supportSetInactivesSize"]
        B = act_size.shape[0]
        Na = batch["supportSetActives"].shape[1]
        Ni = batch["supportSetInactives"].shape[1]

        act_mask = torch.arange(Na, device=self.device).unsqueeze(0) < act_size.unsqueeze(1)
        inact_mask = torch.arange(Ni, device=self.device).unsqueeze(0) < inact_size.unsqueeze(1)

        # Support set dropout — randomly mask additional molecules during training
        # In training_step, replace the dropout masking block with this:
        ss_dropout = self.cfg.model.transformer.ss_dropout
        act_dropout_mask = torch.rand(act_mask.shape, device=self.device) < ss_dropout
        inact_dropout_mask = torch.rand(inact_mask.shape, device=self.device) < ss_dropout
        act_mask_dropped = act_mask & ~act_dropout_mask
        inact_mask_dropped = inact_mask & ~inact_dropout_mask

        # Guarantee at least 1 valid active and 1 valid inactive per sample
        for b in range(act_mask.shape[0]):
            if act_mask_dropped[b].sum() == 0:
                # restore a random valid one
                valid_idx = act_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_idx) > 0:
                    act_mask_dropped[b, valid_idx[torch.randint(len(valid_idx),(1,))]] = True
            if inact_mask_dropped[b].sum() == 0:
                valid_idx = inact_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_idx) > 0:
                    inact_mask_dropped[b, valid_idx[torch.randint(len(valid_idx),(1,))]] = True

        act_mask = act_mask_dropped
        inact_mask = inact_mask_dropped

        # Store masks in batch for forward
        batch["_act_mask"] = act_mask
        batch["_inact_mask"] = inact_mask

        logits = self(batch, use_fixed_context=False)
        labels = batch["label"].float().reshape(-1, 1)
        labels_smoothed = labels * 0.9 + 0.05
        loss = self.loss_fn(logits, labels_smoothed)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # --------------------------------------------------------
    # Validation step — fixed context
    # --------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        logits = self(batch, use_fixed_context=True)
        labels = batch["label"].float().reshape(-1, 1)
        loss = self.loss_fn(logits, labels)

        probs    = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)
        tasks_np  = batch["taskIdx"].detach().cpu().numpy().reshape(-1)  # ← add this

        self._val_outputs.append({
            "loss": loss,
            "probs": probs,
            "labels": labels_np,
            "tasks": tasks_np,   # ← add this
        })

    def on_validation_epoch_end(self):
        # group by task
        per_task = {}
        for o in self._val_outputs:
            for prob, label, task in zip(o["probs"], o["labels"], o["tasks"]):
                tid = int(task)
                per_task.setdefault(tid, {"probs": [], "labels": []})
                per_task[tid]["probs"].append(prob)
                per_task[tid]["labels"].append(label)

        # compute dAUPRC per task, then average
        dauprc_list = []
        for tid, d in per_task.items():
            p = np.array(d["probs"])
            l = np.array(d["labels"])
            if l.sum() > 0 and l.sum() < len(l):
                auprc = average_precision_score(l, p)
                baseline = l.mean()
                dauprc_list.append(auprc - baseline)

        dauprc = float(np.mean(dauprc_list)) if dauprc_list else 0.0

        avg_loss = torch.stack([o["loss"] for o in self._val_outputs]).mean()
        self._val_dauprc_history.append(dauprc)
        dauprc_ma = np.mean(self._val_dauprc_history[-10:])

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("dAUPRC_val", dauprc, prog_bar=True)
        self.log("dAUPRC_val_ma", dauprc_ma, prog_bar=True)
        self._val_outputs = []

    # --------------------------------------------------------
    # Training epoch end
    # --------------------------------------------------------
    def on_train_epoch_end(self):
        self.log("dAUPRC_train_val_delta", 0.0)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        # In models.py configure_optimizers(), replace scheduler with:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "dAUPRC_val",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            },
        }
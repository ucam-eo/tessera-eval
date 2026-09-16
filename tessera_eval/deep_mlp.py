"""A PyTorch MLP classifier reproducing the Tessera paper's own downstream-
eval architecture, for the classes in Louis Driver's Austrian-crop F1 gap
that scaling alone doesn't close.

Frank (the paper's downstream-eval author) described his setup directly,
2026-09-16, in response to that gap: 128->512->256->17, each hidden layer
Linear+BatchNorm1d+ReLU+Dropout(0.3), trained with AdamW(1e-3, weight_decay
0.01), batch 8192, 150 epochs, best checkpoint kept by validation weighted-F1
-- and explicitly, unprompted: "Louis's v1/v2/v3 don't seem to have
BatchNorm and that really matters I think, BN helps to get rid of
overfitting." classify.py's existing "mlp" (sklearn's MLPClassifier, wrapped
in _make_mlp with StandardScaler) has no BatchNorm/Dropout equivalent and no
checkpoint selection at all -- confirmed the gap survives that regardless of
how much training data TEE gives it (Louis's own learning-curve numbers:
best sklearn variant tops out at macro F1 0.598 at 80% training data, vs.
Frank's 0.7248 at 30%, and a *deeper* sklearn MLP sometimes scores *worse*
than a shallower one at the same training percentage -- the classic
signature of an unnormalized/unregularized net struggling to optimize, not
a model-capacity or task-difficulty problem). This module exists to let
classify.py offer an architecture that can actually reach that regime.
Interestingly, Frank *doesn't* scale his input features either -- BatchNorm1d
right after the first Linear layer does that job per-batch, which is also
why classify.py's own scaling fix (v1.11.4) only closed ~1-2 of the ~20+
point gap: it addressed the same symptom sklearn's MLP has for a different
reason (no BatchNorm to do it implicitly).

Deliberately classification-only for now -- that's the only task this was
diagnosed against; a regression counterpart is unimplemented and not part of
that evidence.

Requires PyTorch (not in requirements.txt / the base tessera-eval install --
same optional-extra pattern as unet.py; make_classifier's "deep_mlp" branch
imports this module lazily, so a fresh install without torch never touches
it unless deep_mlp is actually requested).
"""

import copy
import logging

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

_TORCH_MISSING = "deep_mlp requires PyTorch. Install with: pip install torch"


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(_TORCH_MISSING)


def _make_net(in_dim, hidden, n_classes, dropout):
    """Linear -> BatchNorm1d -> ReLU -> Dropout per hidden layer, ending in
    a plain Linear head (no softmax -- CrossEntropyLoss applies it
    internally, and predict_proba below applies it explicitly at inference).
    """
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


class DeepMLPClassifier:
    """scikit-learn-compatible (fit/predict/predict_proba) MLP classifier
    matching Frank's architecture -- see module docstring for the full
    diagnosis this exists to address.

    Distinct from classify.py's "mlp" (sklearn's MLPClassifier via
    _make_mlp): BatchNorm1d + Dropout per hidden layer, AdamW with weight
    decay, and best-checkpoint-by-validation-weighted-F1 rather than
    final-epoch weights.
    """

    def __init__(
        self,
        hidden=(512, 256),
        dropout=0.3,
        lr=1e-3,
        weight_decay=0.01,
        batch_size=8192,
        epochs=150,
        val_fraction=0.15,
        seed=42,
    ):
        _require_torch()
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.seed = seed
        self.model_ = None
        self.classes_ = None

    def _device(self):
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def fit(self, X, y):
        """Fit on X (N, dim) float features and y (N,) int labels.

        y doesn't need to be a dense 0..k-1 range -- self.classes_ (sorted
        np.unique(y)) is the label space predict()/predict_proba() report
        against, same convention as sklearn. (Callers in this codebase
        already pass a dense range via _fit_predict_relabeled, but this
        doesn't assume that.)
        """
        _require_torch()
        if len(X) == 0:
            raise ValueError("No training samples")

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        # Internal held-out split for checkpoint selection -- mirrors
        # Frank's own methodology (best checkpoint by val weighted-F1, not
        # the final epoch's weights). Falls back to training without one
        # (final-epoch weights) whenever a stratified split isn't possible:
        # small learning-curve percentages can leave some classes with only
        # 1-2 samples, too few to hold any out.
        X_va = y_va = None
        if len(X) >= max(20, 2 * n_classes):
            try:
                from sklearn.model_selection import train_test_split

                X, X_va, y_idx, y_va = train_test_split(
                    X, y_idx, test_size=self.val_fraction, random_state=self.seed, stratify=y_idx
                )
            except ValueError:
                pass  # too few members in some class to stratify -- no val split

        device = self._device()
        model = _make_net(X.shape[1], self.hidden, n_classes, self.dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx))
        batch_size = max(1, min(self.batch_size, len(train_ds)))
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        best_state = None
        best_f1 = -1.0
        for _epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                if xb.size(0) < 2:
                    continue  # BatchNorm1d needs >1 sample/batch in train mode
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            if X_va is not None:
                model.eval()
                with torch.no_grad():
                    preds = model(torch.from_numpy(X_va).to(device)).argmax(dim=1).cpu().numpy()
                from sklearn.metrics import f1_score

                val_f1 = f1_score(y_va, preds, average="weighted", zero_division=0)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        model.cpu().eval()
        self.model_ = model
        return self

    def _predict_logits(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.model_ is None:
            raise RuntimeError("DeepMLPClassifier.fit() must be called before predict()")
        self.model_.eval()
        chunks = []
        # Batched inference -- a full-map predict can be millions of pixels;
        # materialising every logit at once would be a needless multi-GB
        # allocation (n_classes floats per pixel).
        step = 65536
        with torch.no_grad():
            for i in range(0, len(X), step):
                chunks.append(self.model_(torch.from_numpy(X[i : i + step])).numpy())
        if not chunks:
            return np.zeros((0, len(self.classes_)), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def predict(self, X):
        idx = self._predict_logits(X).argmax(axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

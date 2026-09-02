"""Shared tokenization, class-imbalance, and batched-inference helpers for Roman
Urdu hate-speech classification. Tweets are short (<=380 chars) so, unlike CUAD's
long contracts, no sliding-window tokenization is needed -- just truncate to a
fixed max length and let a dynamic-padding collator handle the rest.

Used by both train_classifier.py (local) and train_kaggle.py (Kaggle GPU); inlined
into kaggle/train_kernel.ipynb by scripts/build_kaggle_notebook.py so the two never
drift.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import Trainer


def tokenize_batch(examples, tokenizer, max_length: int = 128):
    return tokenizer(examples["tweet"], truncation=True, max_length=max_length)


def compute_class_weights(labels, num_labels: int) -> torch.Tensor:
    weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=np.asarray(labels))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """A Trainer that applies a per-class weighted cross-entropy loss, so the
    5-class fine-grained task (Normal is ~8x more common than Profane) doesn't just
    collapse to predicting the majority class."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def predict_logits(model, features, batch_size: int, device: str, collate_fn) -> np.ndarray:
    """Batched forward pass over tokenized features (input_ids/attention_mask
    only), returning logits as a numpy array. Shared by evaluate.py and the Kaggle
    end-to-end driver."""
    loader = DataLoader(
        features.with_format("torch", columns=["input_ids", "attention_mask"]),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    all_logits = []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)

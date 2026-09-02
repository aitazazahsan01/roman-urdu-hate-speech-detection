"""End-to-end GPU driver: data prep + fine-tuning + test-set evaluation for the
Fine_Grained (primary) RUHSOLD task, in a single run. Meant for a Kaggle kernel
(free T4 GPU) -- see kaggle/kernel-metadata.json and scripts/build_kaggle_notebook.py,
which inlines this file (plus clf_utils.py, metrics_utils.py, and data_prep's
loading functions) into a single self-contained notebook, since `kaggle kernels
push` only pushes one file.

Coarse_Grained is cheap enough (binary, ~10k tweets) to train locally on CPU via
`train_classifier.py --config Coarse_Grained`, so it doesn't need its own GPU
kernel.

Written as flat top-level script code (no main() wrapper) so it can be dropped in
as notebook cells as-is. Also runs standalone: `python src/train_kaggle.py` on any
machine with a GPU and requirements.txt installed.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from clf_utils import WeightedLossTrainer, compute_class_weights, predict_logits, tokenize_batch
from data_prep import build_splits, label_names_for
from metrics_utils import compute_metrics_dict, compute_metrics_for_trainer

# ---- config ------------------------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
CONFIG_NAME = "Fine_Grained"
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SEED = 42
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5

ON_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).resolve().parent.parent
MODEL_OUTPUT_DIR = OUTPUT_ROOT / "outputs" / f"xlmr-{CONFIG_NAME.lower()}"
RESULTS_PATH = OUTPUT_ROOT / "results" / "metrics.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  Output root: {OUTPUT_ROOT}")

# ---- data ----------------------------------------------------------------------
set_seed(SEED)
splits = build_splits(CONFIG_NAME, VAL_FRACTION, TEST_FRACTION, SEED)
label_names = label_names_for(splits)
num_labels = len(label_names)
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}

train_ds = splits["train"].rename_column("label", "labels")
eval_ds = splits["validation"].rename_column("label", "labels")
test_ds = splits["test"]
print(f"train={len(train_ds)}  val={len(eval_ds)}  test={len(test_ds)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
)


def tokenize(examples):
    return tokenize_batch(examples, tokenizer, MAX_SEQ_LENGTH)


train_features = train_ds.map(tokenize, batched=True, remove_columns=["tweet"])
eval_features = eval_ds.map(tokenize, batched=True, remove_columns=["tweet"])
print(f"{len(train_features)} train / {len(eval_features)} val tokenized features")

class_weights = compute_class_weights(train_ds["labels"], num_labels)
print(f"Class weights ({label_names}): {class_weights.tolist()}")

# ---- train -----------------------------------------------------------------------
steps_per_epoch = -(-len(train_features) // TRAIN_BATCH_SIZE)  # ceil div
total_steps = int(steps_per_epoch * NUM_EPOCHS)
warmup_steps = int(total_steps * 0.1)

training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    fp16=(DEVICE == "cuda"),
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_features,
    eval_dataset=eval_features,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics_for_trainer,
    class_weights=class_weights,
)
trainer.train()

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
print(f"Saved model to {MODEL_OUTPUT_DIR}")

# ---- evaluate on held-out test split ----------------------------------------------
test_features = test_ds.map(tokenize, batched=True, remove_columns=["tweet"])
logits = predict_logits(model, test_features, EVAL_BATCH_SIZE, DEVICE, DataCollatorWithPadding(tokenizer))
preds = np.argmax(logits, axis=-1)

results = compute_metrics_dict(test_ds["label"], preds.tolist(), label_names, split="test")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
all_results[CONFIG_NAME] = results
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"Accuracy: {results['accuracy']:.4f}  |  Macro-F1: {results['macro_f1']:.4f}  |  Weighted-F1: {results['weighted_f1']:.4f}")
print(f"Saved metrics to {RESULTS_PATH}")

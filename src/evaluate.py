"""Evaluate a fine-tuned Roman Urdu hate-speech classifier: accuracy, macro/weighted
F1, per-class precision/recall/F1, and a confusion matrix. See
metrics_utils.compute_metrics_dict for the exact metric definitions.

Usage:
    python src/evaluate.py --model-dir outputs/xlmr-fine_grained --config Fine_Grained --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from clf_utils import predict_logits, tokenize_batch
from metrics_utils import compute_metrics_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data" / "processed"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--config", choices=["Coarse_Grained", "Fine_Grained"], default="Fine_Grained")
    p.add_argument("--data-dir", type=Path, default=None, help="Default: data/processed/<config>")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or DEFAULT_DATA_ROOT / args.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = DatasetDict.load_from_disk(str(data_dir))
    examples = splits[args.split]
    label_names = json.loads((data_dir / "label_names.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))

    if model.config.num_labels != len(label_names):
        raise ValueError(
            f"--model-dir has {model.config.num_labels} labels but --config {args.config} expects "
            f"{len(label_names)} -- did you mix up a Coarse_Grained checkpoint with Fine_Grained data "
            "(or vice versa)?"
        )

    def tokenize(batch):
        return tokenize_batch(batch, tokenizer, args.max_seq_length)

    features = examples.map(tokenize, batched=True, remove_columns=["tweet"])

    logits = predict_logits(model, features, args.batch_size, device, DataCollatorWithPadding(tokenizer))
    preds = np.argmax(logits, axis=-1)

    results = compute_metrics_dict(examples["label"], preds.tolist(), label_names, args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    all_results[args.config] = results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"Accuracy: {results['accuracy']:.4f}  |  Macro-F1: {results['macro_f1']:.4f}  |  "
        f"Weighted-F1: {results['weighted_f1']:.4f}"
    )
    print("Worst 3 classes by F1:")
    for row in results["per_class"][:3]:
        print(f"  {row['f1']:.3f}  (n={row['support']:>4})  {row['label']}")
    print(f"Saved full results to {args.output}")


if __name__ == "__main__":
    main()

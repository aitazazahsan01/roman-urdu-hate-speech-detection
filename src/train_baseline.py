"""TF-IDF + Logistic Regression baseline for Roman Urdu hate-speech classification.

Trains and evaluates in seconds on CPU -- run this for real (not just smoke-tested)
before ever touching Kaggle GPU quota, to get an honest "does the transformer
actually help" comparison row against the xlm-roberta-base fine-tune.

Usage:
    python src/train_baseline.py                       # both configs
    python src/train_baseline.py --config Fine_Grained
"""

import argparse
import json
from pathlib import Path

import joblib
from datasets import DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from metrics_utils import compute_metrics_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = ROOT / "outputs"
CONFIGS = ["Coarse_Grained", "Fine_Grained"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, default=None, help="Default: run both configs.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "baseline_metrics.json")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Where to save the fitted pipeline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(config_name: str, args) -> dict:
    data_dir = args.data_root / config_name
    splits = DatasetDict.load_from_disk(str(data_dir))
    label_names = json.loads((data_dir / "label_names.json").read_text(encoding="utf-8"))

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max))),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)),
        ]
    )
    pipeline.fit(splits["train"]["tweet"], splits["train"]["label"])

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"baseline-{config_name.lower()}.joblib"
    joblib.dump({"pipeline": pipeline, "label_names": label_names}, model_path)
    print(f"Saved fitted baseline pipeline to {model_path}")

    preds = pipeline.predict(splits["test"]["tweet"])
    return compute_metrics_dict(splits["test"]["label"], preds.tolist(), label_names, split="test")


def main():
    args = parse_args()
    configs = [args.config] if args.config else CONFIGS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}

    for config_name in configs:
        print(f"\n=== {config_name} (TF-IDF + Logistic Regression) ===")
        results = run_one(config_name, args)
        all_results[config_name] = results
        print(
            f"Accuracy: {results['accuracy']:.4f}  |  Macro-F1: {results['macro_f1']:.4f}  |  "
            f"Weighted-F1: {results['weighted_f1']:.4f}"
        )
        print("Worst 3 classes by F1:")
        for row in results["per_class"][:3]:
            print(f"  {row['f1']:.3f}  (n={row['support']:>4})  {row['label']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved baseline metrics to {args.output}")


if __name__ == "__main__":
    main()

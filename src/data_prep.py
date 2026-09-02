"""Download the RUHSOLD Roman Urdu hate-speech dataset and build train/val/test
splits for both the coarse-grained (binary) and fine-grained (5-class) label
schemes.

RUHSOLD (Rizwan et al., EMNLP 2020) ships on the Hugging Face Hub as two configs:
`Coarse_Grained` (Abusive/Offensive vs Normal) and `Fine_Grained` (adds Religious
Hate, Sexism, Profane/Untargeted). Each example is one tweet + one label.

Usage:
    python src/data_prep.py                      # both configs
    python src/data_prep.py --config Fine_Grained
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "community-datasets/roman_urdu_hate_speech"
CONFIGS = ["Coarse_Grained", "Fine_Grained"]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MIN_TWEET_LEN, MAX_TWEET_LEN = 6, 380


def load_ruhsold(config_name: str):
    # Some community datasets ship a legacy Python loading script that recent
    # `datasets` versions refuse to execute; fall back to the Hub's
    # auto-converted Parquet branch, which needs no remote code.
    try:
        return load_dataset(DATASET_NAME, config_name)
    except Exception:
        return load_dataset(DATASET_NAME, config_name, revision="refs/convert/parquet")


def _collect_labeled_pool(raw: DatasetDict):
    """This HF Hub port of RUHSOLD has real split-integrity issues, confirmed by
    inspection: `test` ships with every label withheld (None) for both configs --
    almost certainly reserved for a private shared-task leaderboard rather than a
    loading bug -- and `Fine_Grained`'s `validation` is an exact duplicate of its
    `train` (same 7,208 tweets, same labels, same order). `Coarse_Grained`'s
    `validation` is a genuine, distinct 800-tweet split.

    Rather than trusting the upstream split boundaries, pool every *uniquely
    labeled* tweet across `train` + `validation` (deduplicated by tweet text, which
    also neutralizes the Fine_Grained duplicate-validation bug) and re-split it
    ourselves below. `test` is dropped entirely -- it carries no usable labels.
    """
    tweets, labels = [], []
    seen = set()
    n_unlabeled, n_duplicate = 0, 0
    for split_name in ("train", "validation"):
        if split_name not in raw:
            continue
        for tweet, label in zip(raw[split_name]["tweet"], raw[split_name]["label"]):
            if label is None:
                n_unlabeled += 1
                continue
            if tweet in seen:
                n_duplicate += 1
                continue
            seen.add(tweet)
            tweets.append(tweet)
            labels.append(label)
    print(
        f"  pooled {len(tweets)} uniquely-labeled tweets from train+validation "
        f"(dropped {n_unlabeled} unlabeled, {n_duplicate} duplicate)"
    )
    return tweets, labels


def build_splits(config_name: str, val_fraction: float, test_fraction: float, seed: int) -> DatasetDict:
    raw = load_ruhsold(config_name)
    tweets, labels = _collect_labeled_pool(raw)

    class_label_feature = raw["train"].features["label"]
    pool = Dataset.from_dict({"tweet": tweets, "label": labels}).cast_column("label", class_label_feature)

    holdout_fraction = val_fraction + test_fraction
    train_holdout = pool.train_test_split(test_size=holdout_fraction, stratify_by_column="label", seed=seed)
    val_test = train_holdout["test"].train_test_split(
        test_size=test_fraction / holdout_fraction, stratify_by_column="label", seed=seed
    )

    return DatasetDict(train=train_holdout["train"], validation=val_test["train"], test=val_test["test"])


def label_names_for(splits: DatasetDict) -> list[str]:
    return splits["train"].features["label"].names


def sanity_check(splits: DatasetDict, label_names: list[str]) -> None:
    """Spot-check tweet lengths and label indices so a bad load fails loudly instead
    of silently training on garbage."""
    n_labels = len(label_names)
    for split_name, ds in splits.items():
        lengths = [len(t) for t in ds["tweet"]]
        out_of_range = sum(1 for length in lengths if not (MIN_TWEET_LEN <= length <= MAX_TWEET_LEN))
        bad_labels = sum(1 for lbl in ds["label"] if not (0 <= lbl < n_labels))
        if bad_labels:
            raise ValueError(f"{split_name}: {bad_labels} examples have an out-of-range label index")
        if out_of_range:
            print(
                f"  warning: {split_name} has {out_of_range}/{len(ds)} tweets outside the documented "
                f"[{MIN_TWEET_LEN}, {MAX_TWEET_LEN}] char range (dataset docs may be imprecise)"
            )
    print("Sanity check passed: label indices all valid.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, default=None, help="Default: process both configs.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    configs = [args.config] if args.config else CONFIGS

    for config_name in configs:
        print(f"\n=== {config_name} ===")
        print(f"Loading {DATASET_NAME} ({config_name}) from the Hugging Face Hub...")
        splits = build_splits(config_name, args.val_fraction, args.test_fraction, args.seed)
        label_names = label_names_for(splits)

        for split_name, ds in splits.items():
            counts = {name: 0 for name in label_names}
            for lbl in ds["label"]:
                counts[label_names[lbl]] += 1
            breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"{split_name:>10}: {len(ds):>6} examples | {breakdown}")

        sanity_check(splits, label_names)

        out_dir = args.out_dir / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(out_dir))
        with open(out_dir / "label_names.json", "w", encoding="utf-8") as f:
            json.dump(label_names, f, indent=2)
        print(f"Saved processed splits + label_names.json to {out_dir}")


if __name__ == "__main__":
    main()

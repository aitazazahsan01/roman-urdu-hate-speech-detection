"""Fine-tune xlm-roberta-base for Roman Urdu hate-speech classification.

Runs identically on CPU (--smoke-test, tiny model + tiny subset, for pipeline
correctness) or on a Kaggle/Colab GPU (full run). See kaggle/train_kernel.ipynb for
the GPU entry point.

Usage:
    python src/train_classifier.py --smoke-test
    python src/train_classifier.py --config Fine_Grained --output-dir outputs/xlmr-fine_grained --fp16
"""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from clf_utils import WeightedLossTrainer, compute_class_weights, tokenize_batch
from metrics_utils import compute_metrics_for_trainer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data" / "processed"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["Coarse_Grained", "Fine_Grained"], default="Fine_Grained")
    p.add_argument("--model-name-or-path", default="xlm-roberta-base")
    p.add_argument("--data-dir", type=Path, default=None, help="Default: data/processed/<config>")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/xlmr-<config lowercased>")
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--num-train-epochs", type=float, default=5.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=32)
    p.add_argument("--per-device-eval-batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny model + tiny data subset, CPU-friendly, to validate the pipeline before spending GPU quota.",
    )
    args = p.parse_args()
    if args.data_dir is None:
        args.data_dir = DEFAULT_DATA_ROOT / args.config
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"xlmr-{args.config.lower()}"
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 8
        args.per_device_eval_batch_size = 8
        args.fp16 = False
        args.output_dir = ROOT / "outputs" / f"smoke-test-{args.config.lower()}"
        print("Running in --smoke-test mode: tiny random-weight model (real tokenizer), tiny data, CPU-only.")

    splits = DatasetDict.load_from_disk(str(args.data_dir))
    label_names = json.loads((args.data_dir / "label_names.json").read_text(encoding="utf-8"))
    num_labels = len(label_names)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    train_ds = splits["train"].rename_column("label", "labels")
    eval_ds = splits["validation"].rename_column("label", "labels")

    if args.smoke_test:
        train_ds = train_ds.select(range(min(32, len(train_ds))))
        eval_ds = eval_ds.select(range(min(16, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.smoke_test:
        # Same tokenizer/architecture family as the real run (catches family-specific
        # bugs), but a randomly-initialized, shrunk config so it trains fast on CPU.
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        config.num_hidden_layers = 2
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = 2
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    def tokenize(examples):
        return tokenize_batch(examples, tokenizer, args.max_seq_length)

    train_features = train_ds.map(tokenize, batched=True, remove_columns=["tweet"])
    eval_features = eval_ds.map(tokenize, batched=True, remove_columns=["tweet"])
    print(f"{len(train_ds)} train / {len(eval_ds)} val examples, tokenized (max_length={args.max_seq_length})")

    class_weights = None
    if args.class_weights == "balanced":
        class_weights = compute_class_weights(train_ds["labels"], num_labels)
        print(f"Class weights ({label_names}): {class_weights.tolist()}")

    steps_per_epoch = -(-len(train_features) // args.per_device_train_batch_size)  # ceil div
    total_steps = int(steps_per_epoch * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()

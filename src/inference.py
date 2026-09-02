"""Single-example inference helper: given a Roman Urdu tweet/comment, predict its
hate-speech/offensive-language label. Used by app/app.py and for quick manual checks.

Usage:
    python src/inference.py --model-dir outputs/xlmr-fine_grained --text "..."
"""

import argparse

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BaselineHateSpeechModel:
    """Loads a TF-IDF + Logistic Regression pipeline saved by train_baseline.py.
    Same predict() shape as RomanUrduHateSpeechModel so callers (app.py) can treat
    the two interchangeably."""

    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.label_names = bundle["label_names"]

    def predict(self, text: str) -> dict:
        probs = self.pipeline.predict_proba([text])[0].tolist()
        probabilities = dict(zip(self.label_names, probs))
        best_label = max(probabilities, key=probabilities.get)
        return {
            "label": best_label,
            "confidence": probabilities[best_label],
            "probabilities": probabilities,
        }


class RomanUrduHateSpeechModel:
    def __init__(self, model_dir: str, max_seq_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)
        self.max_seq_length = max_seq_length

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        id2label = self.model.config.id2label
        probabilities = {id2label[i]: p for i, p in enumerate(probs)}
        best_id = int(logits.argmax())

        return {
            "label": id2label[best_id],
            "confidence": probabilities[id2label[best_id]],
            "probabilities": probabilities,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="HF model dir, or a .joblib baseline path with --baseline")
    p.add_argument("--baseline", action="store_true", help="Load a TF-IDF+LogReg pipeline instead of a transformer")
    p.add_argument("--text", required=True)
    args = p.parse_args()

    model = BaselineHateSpeechModel(args.model_dir) if args.baseline else RomanUrduHateSpeechModel(args.model_dir)
    result = model.predict(args.text)

    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.3f}")
    for label, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:>20}: {prob:.3f}")


if __name__ == "__main__":
    main()

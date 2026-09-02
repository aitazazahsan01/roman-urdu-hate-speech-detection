"""Gradio demo: type a Roman Urdu tweet/comment, get the predicted hate-speech/
offensive-language label with a per-class confidence breakdown.

Usage:
    python app/app.py --model-dir outputs/xlmr-fine_grained
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import BaselineHateSpeechModel, RomanUrduHateSpeechModel  # noqa: E402

EXAMPLES = [
    "Yeh video bohat acha tha, shukriya share karne ke liye!",
    "Tum jaise logon ko yahan se nikal dena chahiye, bekar qaum ho",
]


def build_app(model_dir: str, baseline: bool = False) -> gr.Blocks:
    try:
        clf = BaselineHateSpeechModel(model_dir) if baseline else RomanUrduHateSpeechModel(model_dir)
        load_error = None
    except Exception as exc:  # model not trained yet, bad path, etc.
        clf = None
        load_error = str(exc)

    def classify(text: str):
        if load_error:
            return {}, f"Model failed to load from '{model_dir}': {load_error}"
        if not text.strip():
            return {}, "Type a tweet/comment first."

        result = clf.predict(text)
        note = f"Predicted: **{result['label']}** (confidence {result['confidence']:.2f})"
        return result["probabilities"], note

    with gr.Blocks(title="Roman Urdu Hate Speech Detection") as demo:
        gr.Markdown(
            "# Roman Urdu Hate Speech Detection\n"
            "Type a Roman Urdu (or code-switched Urdu-English) tweet or comment and the model "
            "will classify it as Normal, Abusive/Offensive, Religious Hate, Sexism, or "
            "Profane/Untargeted."
        )
        with gr.Row():
            with gr.Column():
                text_box = gr.Textbox(label="Tweet / comment", lines=4, placeholder="Type a Roman Urdu tweet here...")
                submit_btn = gr.Button("Classify", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=text_box)
            with gr.Column():
                label_box = gr.Label(label="Class probabilities")
                note_box = gr.Markdown()

        submit_btn.click(classify, inputs=[text_box], outputs=[label_box, note_box])

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(ROOT / "outputs" / "xlmr-fine_grained"))
    p.add_argument("--baseline", action="store_true", help="Load a TF-IDF+LogReg pipeline instead of a transformer")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    demo = build_app(args.model_dir, baseline=args.baseline)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

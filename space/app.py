"""Hugging Face Space entry point: same demo as ../app/app.py, but loads the
transformer from the Hub (code-aitazaz/roman-urdu-hate-speech-xlmr-*) instead of a
local path, and offers both label schemes and both models (XLM-R vs. the TF-IDF
baseline, bundled here as .joblib files) via dropdowns -- including the honest
result that the baseline currently beats the transformer on Fine_Grained.
"""

import joblib
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HUB_MODELS = {
    "Fine_Grained": "code-aitazaz/roman-urdu-hate-speech-xlmr-fine-grained",
    "Coarse_Grained": "code-aitazaz/roman-urdu-hate-speech-xlmr-coarse-grained",
}
BASELINE_FILES = {
    "Fine_Grained": "baseline-fine_grained.joblib",
    "Coarse_Grained": "baseline-coarse_grained.joblib",
}
MODEL_CHOICES = ["XLM-R (transformer)", "TF-IDF + Logistic Regression"]
TASK_CHOICES = ["Fine_Grained", "Coarse_Grained"]

_cache = {}


def load_transformer(task: str):
    key = ("xlmr", task)
    if key not in _cache:
        repo_id = HUB_MODELS[task]
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id)
        model.eval()
        _cache[key] = (tokenizer, model)
    return _cache[key]


def load_baseline(task: str):
    key = ("baseline", task)
    if key not in _cache:
        _cache[key] = joblib.load(BASELINE_FILES[task])
    return _cache[key]


def predict_transformer(text: str, task: str) -> dict:
    tokenizer, model = load_transformer(task)
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    id2label = model.config.id2label
    return {id2label[i]: p for i, p in enumerate(probs)}


def predict_baseline(text: str, task: str) -> dict:
    bundle = load_baseline(task)
    probs = bundle["pipeline"].predict_proba([text])[0].tolist()
    return dict(zip(bundle["label_names"], probs))


def classify(text: str, task: str, model_choice: str):
    if not text.strip():
        return {}, "Type a tweet/comment first."
    probabilities = predict_transformer(text, task) if model_choice == MODEL_CHOICES[0] else predict_baseline(text, task)
    best_label = max(probabilities, key=probabilities.get)
    note = f"Predicted: **{best_label}** (confidence {probabilities[best_label]:.2f})"
    return probabilities, note


EXAMPLES = [
    ["Yeh video bohat acha tha, shukriya share karne ke liye!", "Fine_Grained", MODEL_CHOICES[0]],
    ["Tum jaise logon ko yahan se nikal dena chahiye, bekar qaum ho", "Fine_Grained", MODEL_CHOICES[1]],
]

with gr.Blocks(title="Roman Urdu Hate Speech Detection") as demo:
    gr.Markdown(
        "# Roman Urdu Hate Speech Detection\n"
        "Classify Roman Urdu / code-switched tweets and comments as **Normal** or a "
        "type of hate speech / offensive content. Built on "
        "[RUHSOLD](https://aclanthology.org/2020.emnlp-main.197) (Rizwan et al., "
        "EMNLP 2020). Two label schemes (`Fine_Grained`: 5-class, `Coarse_Grained`: "
        "binary) and two models -- notably, the classical TF-IDF + Logistic "
        "Regression baseline currently *beats* the fine-tuned XLM-R transformer on "
        "the harder `Fine_Grained` task (macro-F1 0.677 vs. 0.617), likely because "
        "5,760 training tweets across 5 imbalanced classes is little data for a "
        "270M-parameter model. Try both and compare."
    )
    with gr.Row():
        with gr.Column():
            text_box = gr.Textbox(label="Tweet / comment", lines=4, placeholder="Type a Roman Urdu tweet here...")
            task_dropdown = gr.Dropdown(TASK_CHOICES, value="Fine_Grained", label="Task")
            model_dropdown = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
            submit_btn = gr.Button("Classify", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[text_box, task_dropdown, model_dropdown])
        with gr.Column():
            label_box = gr.Label(label="Class probabilities")
            note_box = gr.Markdown()

    submit_btn.click(classify, inputs=[text_box, task_dropdown, model_dropdown], outputs=[label_box, note_box])

if __name__ == "__main__":
    demo.launch()

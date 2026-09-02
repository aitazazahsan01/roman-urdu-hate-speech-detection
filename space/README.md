---
title: Roman Urdu Hate Speech Detection
emoji: 🗣️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

Classify Roman Urdu / code-switched tweets and comments as Normal or a type of
hate speech / offensive content. Compares a fine-tuned XLM-R transformer against a
TF-IDF + Logistic Regression baseline across both the RUHSOLD `Fine_Grained`
(5-class) and `Coarse_Grained` (binary) label schemes.

Fine-tuned model weights: [Fine_Grained](https://huggingface.co/code-aitazaz/roman-urdu-hate-speech-xlmr-fine-grained) ·
[Coarse_Grained](https://huggingface.co/code-aitazaz/roman-urdu-hate-speech-xlmr-coarse-grained)

Trained on [RUHSOLD](https://aclanthology.org/2020.emnlp-main.197) (Rizwan et al., EMNLP 2020).

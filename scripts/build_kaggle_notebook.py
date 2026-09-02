"""Generate kaggle/train_kernel.ipynb by inlining src/clf_utils.py,
src/metrics_utils.py, data_prep.py's loading functions, and src/train_kaggle.py
into a single self-contained notebook.

`kaggle kernels push` only pushes one code file, so rather than hand-maintaining a
separate copy of the training pipeline for Kaggle (and letting it drift from
src/train_classifier.py's actual logic), this script assembles the notebook straight
from the real source files. Re-run it after changing any of the four files above.

Usage:
    python scripts/build_kaggle_notebook.py
    python scripts/build_kaggle_notebook.py --config Coarse_Grained
"""

import argparse
import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
KAGGLE_DIR = Path(__file__).resolve().parent.parent / "kaggle"
# Fine_Grained is the primary kernel at kaggle/train_kernel.ipynb; Coarse_Grained
# gets its own push directory (kaggle/coarse/) since `kaggle kernels push` needs a
# distinct kernel-metadata.json per kernel id.
OUT_BY_CONFIG = {
    "Fine_Grained": KAGGLE_DIR / "train_kernel.ipynb",
    "Coarse_Grained": KAGGLE_DIR / "coarse" / "train_kernel.ipynb",
}


def read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def strip_module_docstring(code: str) -> str:
    return re.sub(r'^"""[\s\S]*?"""\n+', "", code, count=1)


def top_imports(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if line.startswith(("import ", "from ")))


def extract_line(code: str, prefix: str) -> str:
    match = re.search(rf"^{re.escape(prefix)}.*$", code, re.MULTILINE)
    if not match:
        raise ValueError(f"couldn't find a line starting with {prefix!r}")
    return match.group(0)


def extract_function(code: str, func_name: str) -> str:
    pattern = rf"^def {func_name}\(.*?(?=^def |\Z)"
    match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"couldn't find function {func_name}")
    return match.group(0).rstrip() + "\n"


def strip_local_imports(code: str) -> str:
    return re.sub(r"^from (clf_utils|metrics_utils|data_prep) import .*\n", "", code, flags=re.MULTILINE)


def set_config_constant(code: str, config_name: str) -> str:
    return re.sub(r'^CONFIG_NAME = ".*"$', f'CONFIG_NAME = "{config_name}"', code, count=1, flags=re.MULTILINE)


_cell_counter = 0


def _next_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"cell-{_cell_counter}"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["Coarse_Grained", "Fine_Grained"], default="Fine_Grained")
    args = parser.parse_args()

    clf_utils_code = strip_module_docstring(read("clf_utils.py"))
    metrics_utils_code = strip_module_docstring(read("metrics_utils.py"))

    data_prep_full = read("data_prep.py")
    data_prep_code = "\n".join(
        [
            top_imports(data_prep_full),
            "",
            extract_line(data_prep_full, "DATASET_NAME ="),
            "",
            extract_function(data_prep_full, "load_ruhsold"),
            "",
            extract_function(data_prep_full, "_collect_labeled_pool"),
            "",
            extract_function(data_prep_full, "build_splits"),
            "",
            extract_function(data_prep_full, "label_names_for"),
        ]
    )

    train_kaggle_code = set_config_constant(
        strip_local_imports(strip_module_docstring(read("train_kaggle.py"))), args.config
    )

    cells = [
        md_cell(
            "# Roman Urdu Hate Speech Detection — Training (Kaggle GPU)\n\n"
            "Auto-generated from `src/` by `scripts/build_kaggle_notebook.py` — "
            "**do not hand-edit this notebook**; change the source files and rerun the "
            "builder instead.\n\n"
            "Before running: Settings → Accelerator → GPU T4 x1, and Internet → On."
        ),
        code_cell('!pip install -q -U "transformers>=4.42" "datasets>=2.19" accelerate scikit-learn sentencepiece\n'),
        code_cell(clf_utils_code),
        code_cell(metrics_utils_code),
        code_cell(data_prep_code),
        code_cell(train_kaggle_code),
        md_cell(
            "Outputs are written under `/kaggle/working/outputs` and `/kaggle/working/results`. "
            "Pull them back locally with:\n\n"
            "```\nkaggle kernels output <username>/<slug> -p ./kaggle_output\n```"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = OUT_BY_CONFIG[args.config]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

# Companion code: Generative AI Engineering Step by Step

Every experiment, figure, and verified listing in the book comes from a script here, and `muse/` is the
running project with its full test suite (32 tests).

## Layout
- `fs2.py`: figure style (large black labels, wrapped box text).
- `chNN/`: one folder per chapter. Experiment scripts write JSON results; `figs.py` draws the figures;
  `chNN.js` (not included) built the Word chapters.
- `muse/`: the Muse application: `src/muse/` (config, LLM client, prompts, RAG, safety, governance,
  ops controls, web front end), `tests/`, `evals/release_suite.py`, `Dockerfile`.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install torch numpy scipy scikit-learn matplotlib pillow tokenizers transformers jsonschema \
                jinja2 fastapi uvicorn httpx pytest mcp cryptography
    curl -L -o tiny.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Paths and order
Scripts use absolute paths from the authoring environment (`/home/claude/...`). Replace them with your
checkout path, e.g. `grep -rl "/home/claude/" . | xargs sed -i "s#/home/claude/gb/#$(pwd)/#g"`, and
place `tiny.txt` and the Chapter 2 model checkpoint where the scripts expect them.
Some chapters reuse earlier artifacts: Ch. 4 tokenizer → Ch. 5; Ch. 7 adapter → Ch. 8, 18, 19;
Ch. 9 models → Ch. 10, 12, 22; Ch. 14 RAG → Ch. 15–21.
Model checkpoints (*.pt) are not included; rerun the training scripts (each takes under 5 minutes on CPU).

## Running Muse
    cd muse && python -m pytest -q            # 32 tests
    python evals/release_suite.py             # release gate
    uvicorn muse.web.app:app                  # web front end at http://127.0.0.1:8000

#!/usr/bin/env bash
# Install semlabel: create .venv, install dependencies, download the embedding model.
# Safe to re-run. Usage: ./install.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
MODEL="Snowflake/snowflake-arctic-embed-xs"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Python environment
step "Python environment"
if command -v uv >/dev/null 2>&1; then
    echo "using uv ($(uv --version))"
    uv sync --frozen
else
    PY="${PYTHON:-python3}"
    command -v "$PY" >/dev/null 2>&1 || die "python3 not found (need 3.12+), or install uv: https://docs.astral.sh/uv/"
    "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
        || die "$("$PY" --version) is too old; semlabel needs Python 3.12+"
    echo "uv not found, using $("$PY" --version) + pip"
    [ -d .venv ] || "$PY" -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet sentence-transformers numpy scikit-learn
fi

# 2. Embedding model (~90 MB, cached by Hugging Face under ~/.cache/huggingface)
step "Embedding model: $MODEL"
.venv/bin/python - "$MODEL" <<'EOF'
import sys
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(sys.argv[1], device="cpu")
dim = model.encode(["hello"]).shape[1]
print(f"loaded, {dim}-dim embeddings")
EOF

# 3. Smoke test on the bundled sample data
step "Smoke test"
HIT="$(./semlabel search "new language model results" examples/sample.jsonl -k 1 --no-cache 2>/dev/null)" \
    || die "smoke test failed: ./semlabel search returned an error"
echo "$HIT" | .venv/bin/python -c "import json,sys; r=json.loads(sys.stdin.read()); assert '_score' in r; print('search OK')"
./semlabel embed-stop >/dev/null 2>&1 || true   # search auto-starts the embed daemon; don't leave it running

# 4. Optional: claude CLI for automatic labeling during training
step "Optional: LLM labeling"
if command -v claude >/dev/null 2>&1; then
    echo "claude CLI found: 'semlabel train --auto' will use it to label examples"
else
    echo "claude CLI not found: 'semlabel train' will ask you to label examples by hand"
    echo "(install Claude Code to enable --auto: https://docs.claude.com/en/docs/claude-code)"
fi

printf '\n\033[32mDone.\033[0m Try: ./semlabel search "chip inference cost" examples/sample.jsonl -k 3\n'
printf 'To run fully offline from now on: export HF_HUB_OFFLINE=1\n'

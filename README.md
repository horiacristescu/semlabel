# semlabel

Label JSONL records with concepts you define in plain language, then run those labels in production for the cost of a dot product.

```bash
./semlabel train concepts/science.json posts.jsonl --auto   # an LLM labels ~150 examples, once
./semlabel tag concepts/ posts.jsonl > tagged.jsonl         # one dot product per concept, then a calibrated confidence
```

It runs locally on CPU, reads and writes JSONL, and fits in shell pipelines.

## How it works

A concept is a direction in embedding space. Every record is embedded once into a unit vector (384 dimensions with the default model). A concept is another unit vector `w`. A record's score for that concept is `embedding · w`: positive means it belongs, negative means it doesn't, and the magnitude says how clearly.

You don't write `w` by hand. You write a description, like *"Major results in biotech, medicine, mathematics. Genuine surprises only, not incremental progress."*, and `train` finds the direction that separates records matching it from records that don't:

1. **Seed.** The description is embedded, and the 50 records closest to it (with near-duplicates removed) are sent for labeling.
2. **Label.** An LLM (`claude -p`) reads the description and the batch and marks each record positive or negative. You can also do the labeling yourself.
3. **Fit.** A linear classifier is fit to the labels so far: ridge regression on targets +1/−1, with class-balanced sample weights and no bias term. The result is normalized to unit length and becomes `w`.
4. **Refine.** The next 40 records are the ones closest to the current boundary (score near 0), where the classifier is least sure. They get labeled, and the classifier is refit. Three to five rounds is usually enough.

5. **Calibrate.** Each labeled example is re-scored by a `w` fit without it (5-fold cross-validation). These held-out scores are stored with the concept and turn raw scores into confidences (below).

The concept file ends up holding the description, the labeled examples, the calibration scores and `w`. From then on, labeling new data needs no LLM.

### Reasoning moves to training time

An LLM classifier reads every record at inference time, so you pay for reasoning on every item forever. semlabel spends that reasoning once, during training, on a few hundred examples the LLM labels. It then compresses the result into a 384-number vector. The expensive judgment happens once, and what ships to production is a lookup.

That gives very different cost profiles:

| | LLM per record | semlabel |
|---|---|---|
| Training | none | ~4 LLM calls, ~150–200 labeled records, a few minutes |
| Per record, 1 concept | one LLM call, ~1 s | one embedding (1–11 ms on CPU, see below) + one dot product |
| Per record, 20 concepts | 20 calls, or one long prompt | the same embedding + a 384×20 matrix multiply |
| Re-scoring an archive | full cost again | free if embeddings are cached |
| Runs offline / data stays local | usually not | yes |

Adding a concept costs nothing at inference: records are embedded once, and every concept reuses that embedding. A thousand records against twenty concepts is a single matrix multiply.

What you give up is expressiveness. A linear direction captures *topics* and *registers* well: "AI hardware news", "clickbait", "sports coverage". It can't capture judgments that depend on reasoning about the specific content, like "is this claim true" or "does this post contradict the previous one". If a concept keeps failing after a few rounds of `add` corrections, it probably isn't linear in the embedding space.

### Each concept is calibrated by its own training set

Plain embedding search needs a global similarity threshold ("anything above 0.35 counts"). No single threshold works for every topic, because some topics are tight clusters and others are diffuse. semlabel doesn't use one.

Each concept gets its own small training set, chosen by active learning around **that concept's** boundary. Most of the labels land on hard cases: records that look related but aren't, and real matches that look unusual. That set does two jobs. It fits `w`, and it calibrates the scores `w` produces, using **conformal prediction**.

For calibration, every labeled example gets a *held-out* score from a `w` fit without it. Scoring examples with a weight that trained on them would look too confident. A new record's raw score `s` is then placed in the sorted held-out scores of each class:

- **FN side:** the share of known positives that scored ≤ `s`. If almost none did, real matches rarely score this low.
- **FP side:** the share of known negatives that scored ≥ `s`. If almost none did, non-matches rarely score this high.

`tag` reports the confidence `p_pos / (p_pos + p_neg)`, where `p_pos` and `p_neg` are those two shares, with a shared smoothing term so that neither class dominates the tails. The confidence runs from 0 to 1 and increases with the raw score. It doesn't depend on how many positives and negatives were labeled. Each concept has its own threshold, taken from its own data. On the concepts we've trained, confidence 0.5 falls anywhere from −0.06 to +0.08 in raw cosine, which is exactly the per-concept correction that a global threshold can't make.

What the numbers mean:

- **0.5 is a balanced point**: the score is equally typical of the concept's positives and its negatives. It ignores how common the concept is. For a rare concept, `> 0.5` flags more records than you probably want. Use `> 0.9` when you need precision.
- **It's rank-based, not a probability.** Conformal prediction assumes the calibration examples look like the data you tag. Active learning oversamples boundary cases, so the confidence is sharpest near the boundary and approximate elsewhere. It is still measured on your data, for this concept, and you can check it.
- **`show` reports held-out recall and specificity** for each concept: an honest estimate on its hardest examples, a lower bound for easier data.
- The labeled examples stay in the concept file. `show` lets you inspect them, `add` fixes individual labels (and recalibrates), and re-running `train` continues from them. You can see exactly which examples define a concept.
- A concept is only as good as its description and its roughly 10+ positives. Narrow, specific descriptions train cleaner than broad ones.

`tag --raw` outputs the uncalibrated cosine instead.

### Compared with Jev

[Jev](https://flaviocopes.com/jev/) from TypeSafe AI is marketed as a "System One" decision model. You pass it a state and a question, and it returns a typed answer (yes/no, a choice, or a score) with a probability, in about 100 ms. Under the "decision engine" name, it's a general classifier that takes its question at call time: in effect a cross-encoder reranker with several output heads, one step above embedding-based RAG. semlabel attacks the same problem, cheap typed decisions over text, from the opposite direction.

**Where the reasoning lives.** Jev answers in a single forward pass. It never reads its own output back, so it can't work through a judgment step by step, and whatever it misses stays missed. semlabel puts the reasoning in a model that *does* iterate. `claude -p` labels in rounds: each round's labels move the boundary, and the next round asks about the records that the move made uncertain. Only the result, one vector, runs in production. The scorer can be a single step because the iteration already happened.

**Calibration.** Jev promises calibrated probabilities from one general model, but it publishes nothing about in-domain evaluation or fine-tuning. A probability is only accurate relative to the data it was calibrated on. If "relevant", "sensitive" or "allow" is rarer or more common in your domain, the same numbers come out systematically too high or too low, and you have no labeled set to check them against. semlabel makes a narrower claim you can check. Its confidence is calibrated by conformal prediction against held-out examples from **your** data, and those labeled examples ship inside the concept file, where you can inspect, correct and retrain them.

| | Jev | semlabel |
|---|---|---|
| Model | one general hosted model, same weights for everyone | one small vector per concept, fit on your data |
| Where judgment happens | at every call, in one forward pass | once, offline, in an LLM labeling loop |
| Defining a class | a question at call time; no training | a description + a few minutes of training |
| Output | typed answer + probability (calibration claimed, not shown in-domain) | conformal confidence from held-out, in-domain examples |
| Evidence behind a decision | none exposed | the stored positive/negative examples |
| Changing the taxonomy | edit the question, instantly | retrain the concept (minutes) |
| Latency | ~70–500 ms per call, per question (network) | 5–11 ms per record, warm, single; +0.15 ms for 20 concepts |
| Cost | ~$0.042 per million input tokens | local compute, after a few LLM calls for training |
| Deployment | hosted API only | local; data stays on your machine after training |

Where each fits: Jev (or an LLM) when the question changes every call and there's nothing to train on, for example "is this chunk relevant to *this* query". semlabel when the concept is stable, the volume is high, the data has to stay local, or you need to know why a record was labeled the way it was. The two also combine: Jev or an LLM can label the training rounds, and semlabel turns those labels into a classifier that costs nothing to run.

Jev figures are from public descriptions of TypeSafe's published numbers, as of September 2026.

### Speed

Measured on an Apple M5 Pro, CPU only, with the default model and real posts (full text averages 1,070 characters):

| Stage | Time |
|---|---|
| Cold start (import + model load) | 4.2 s, once. `embed-server` keeps the model loaded |
| Embed a single record, warm | 11 ms full text, 5 ms title only (p50) |
| Embed in batch | 9 ms/record full text (109/s), 1 ms/record title only (1,000/s) |
| Score 20 concepts, dot product + conformal | 0.15 ms for a single record, 1.2 µs/record in batch (16M decisions/s) |

Embedding takes nearly all the time. Once a record is embedded, the decisions are practically free, so adding concepts costs almost nothing, and re-scoring a cached archive runs only the last row.



## Install

```bash
git clone https://github.com/horiacristescu/semlabel.git && cd semlabel
./install.sh
```

`install.sh`:

1. creates `.venv/`, using [uv](https://docs.astral.sh/uv/) if installed, otherwise `python3 -m venv` + pip;
2. downloads the embedding model (about 90 MB, cached in `~/.cache/huggingface`) and checks that it loads;
3. runs a smoke-test search on `examples/sample.jsonl`;
4. checks for the `claude` CLI, which is optional.

After install everything runs offline. Set `HF_HUB_OFFLINE=1` to guarantee no network calls.

Requirements:

- Python 3.12+
- Optional: the [`claude` CLI](https://docs.claude.com/en/docs/claude-code) for automatic labeling (`train --auto`). Labeling runs `claude -p` with your Claude Code default model. Without the CLI, `train` asks you to label each batch by hand.

`./semlabel` is a small wrapper that runs `src/semlabel.py` with the `.venv` next to it, so you can symlink it onto your `PATH` and call it from anywhere.

`sentence-transformers` pulls in PyTorch. On Linux, pip's default PyTorch includes CUDA and is large. On a CPU-only machine, install the CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Usage

### Input format

One JSON object per line. semlabel reads three fields and passes the rest through unchanged:

```json
{"id": "a1b2c3", "title": "Octopuses learn by watching each other",
 "comments": [{"author": "Article", "text": "The study suggests..."},
              {"author": "someone", "text": "Fascinating result."}]}
```

The embedded text is the title, then comments whose author is `Article` or `Abstract` (treated as the body), then the remaining comments, cut at 2000 characters. A record with only a `title` works. `id` is optional for `tag` and `search`, but `add` and `search --id` use it to find records.

### Workflow

```bash
# 1. See what's in the data
./semlabel discover data.jsonl -n 10          # 10 clusters, 5 examples each
./semlabel search "chip inference cost" data.jsonl -k 5

# 2. Describe a concept
echo '{"description": "New AI accelerator chips and inference cost. Not general GPU gaming news."}' \
  > concepts/ai_hardware.json

# 3. Train it
./semlabel train concepts/ai_hardware.json data.jsonl --auto

# 4. Inspect, correct
./semlabel show concepts/ai_hardware.json
./semlabel add concepts/ai_hardware.json data.jsonl a1b2c3 - --dry-run   # what changes if a1b2c3 is negative?
./semlabel add concepts/ai_hardware.json data.jsonl a1b2c3 -

# 5. Tag, in batch or in a pipeline
./semlabel tag concepts/ data.jsonl > tagged.jsonl
cat today.jsonl | ./semlabel tag concepts/ | jq -c 'select(.tags.ai_hardware > 0.9)'
```

`concepts/` includes five untrained example descriptions, and `examples/sample.jsonl` is a small made-up dataset for trying `search` and `discover`. Training needs real data, at least a few hundred records.

### Commands

| Command | What it does |
|---|---|
| `train CONCEPT DATA [--auto]` | Active-learning training (above). Re-running continues from the existing labels, which is also how you extend a concept to new data. `--trace FILE` logs LLM prompts and replies; `--log-texts FILE` logs the exact embedded texts. |
| `tag CONCEPTS [DATA]` | Adds `"tags": {"concept": confidence, ...}` to each record (0–1, conformal; `--raw` for the cosine). `CONCEPTS` is a file or a directory; untrained concepts are skipped. Reads stdin when `DATA` is omitted (all of it before writing output, so it batches rather than streams). `--tag-name NAME` renames the field. |
| `search QUERY DATA [-k N] [--id]` | Nearest neighbors by cosine similarity, printed as JSONL with `_score`. With `--id`, `QUERY` is a record id ("more like this"). No concept needed. |
| `discover DATA [-n N] [-k K]` | k-means into `N` clusters, `K` examples each. A quick survey of what a dataset contains. |
| `add CONCEPT DATA ID +/- ...` | Adds records as positives (`+`) or negatives (`-`), then refits `w` and the calibration. `--dry-run` shows which records would change sides. |
| `show CONCEPTS` | Description, example counts, held-out recall/specificity, and labeled examples for each concept. |
| `calibrate CONCEPTS` | Computes and stores calibration for concepts trained before it existed. `tag` computes a missing calibration on the fly, but doesn't save it. |
| `embed-server` / `embed-stop` | Keeps the model in memory on a unix socket (`/tmp/semlabel-embed.sock`). Other commands use it automatically when it's running, which saves a few seconds of model loading per call. |

Manual labeling: without `--auto`, `train` prints each batch with short ids, and at the `pos>` prompt you type the ids that match, separated by spaces. Everything else in the batch is a negative.

### Caching

The first time a data file is embedded, semlabel writes `data.jsonl.embeds.npy` next to it, and later runs skip the embedding. The cache is keyed by byte offset, so appending records is fine. If you edit records in place, delete the `.npy` file. `--no-cache` bypasses the cache.

## Choosing an embedding model

The default is [`Snowflake/snowflake-arctic-embed-xs`](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs): 22M parameters, 384 dimensions, fast on CPU and good at topical separation in English. Because the learned concept direction does the fine discrimination, a small model usually goes further than you'd expect. Consider switching when:

| Need | Candidates (dims) |
|---|---|
| Faster / smaller | `sentence-transformers/all-MiniLM-L6-v2` (384) |
| Better English quality, still small | `BAAI/bge-small-en-v1.5` (384), `Snowflake/snowflake-arctic-embed-s` (384) |
| Non-English or mixed-language data | `intfloat/multilingual-e5-small` (384), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384) |
| Higher quality, slower | `Snowflake/snowflake-arctic-embed-m-v1.5` (768), `BAAI/bge-base-en-v1.5` (768) |

The quickest test for a candidate is to train one concept you care about with each model on the same data, then compare with `show`. If a model is better for your data, the positives come out cleaner and fewer boundary records need correcting.

To switch models, set `MODEL_NAME` at the top of `src/semlabel.py`. If the new model's dimension isn't 384, also replace the `384` literals in the same file (`grep -n 384 src/semlabel.py`). Also set `QUERY_PREFIX`, the text added to search queries and concept descriptions before they're embedded. Arctic-embed and BGE models expect an instruction prefix there (see the model card). Symmetric models like MiniLM expect `""`. Then:

- **Retrain every concept.** A concept vector only means something in the space of the model that produced it. The file records the model in its `model` field.
- **Delete the `*.embeds.npy` caches.** They don't record which model produced them.
- **Restart the embed server** if it's running (`./semlabel embed-stop`).

## Using it from Claude Code

`skills/semlabel/SKILL.md` is a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) that teaches an agent the whole workflow: survey the data, write a concept description, train, inspect, correct labels by id, and tag new data with sensible thresholds. It also covers the **agent-as-labeler** mode. There, the agent reads candidates from `search`, labels them itself with `add`, and repeats on the uncertain band, with no separate `claude -p` calls. To install it:

```bash
mkdir -p ~/.claude/skills && ln -s "$PWD/skills/semlabel" ~/.claude/skills/semlabel
```

Then ask Claude Code something like *"use semlabel to build a concept for AI hardware news from data.jsonl and tag today's feed"*. The skill expects `SEMLABEL` to point at the `semlabel` wrapper in this repo.

## Limitations

- Concepts are linear. They capture topics and styles, not judgments that need reasoning about the content.
- Confidences are rank-based and calibrated on a boundary-heavy sample, not exact probabilities. 0.5 ignores the base rate; use a higher cutoff for rare concepts.
- Input is truncated at 2000 characters. Long documents are represented by their beginning.
- Training is only as good as the labels. With `--auto`, precision depends on the description being specific enough for the LLM to judge edge cases.

## License

Apache License 2.0. See [LICENSE](LICENSE).

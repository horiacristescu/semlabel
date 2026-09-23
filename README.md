# semlabel

**Semantic text classification you train once and run for free.** You describe a concept in plain language. An LLM labels a few hundred examples from your own data, and semlabel condenses those labels into a single vector. From then on, classifying a record is one embedding plus one dot product per concept, turned into a confidence calibrated on that concept's own examples. It runs locally on CPU and reads and writes JSONL, so it fits in shell pipelines.

```bash
./semlabel train concepts/ai_hardware.json posts.jsonl --auto   # an LLM labels ~150–400 examples, once
./semlabel tag concepts/ today.jsonl > tagged.jsonl             # every concept, ~10 ms per record
```

Each record comes back with a confidence per concept (example output):

```json
{"id": "a1b2", "title": "New inference chip claims 3x lower cost per token", "tags": {"ai_hardware": 0.97, "slop": 0.04}}
```

## Install

```bash
git clone https://github.com/horiacristescu/semlabel.git && cd semlabel
./install.sh
```

`install.sh` sets up `.venv/` (with [uv](https://docs.astral.sh/uv/) if present, otherwise `python3 -m venv` + pip) and downloads the embedding model: `Snowflake/snowflake-arctic-embed-xs`, about 90 MB, cached in `~/.cache/huggingface`. It then runs a smoke test and checks for the `claude` CLI. After that everything runs offline; set `HF_HUB_OFFLINE=1` to guarantee it.

- **Python 3.12+.** On a CPU-only Linux machine, install the CPU build of PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). The default build bundles CUDA and is large.
- **Optional: the [`claude` CLI](https://docs.claude.com/en/docs/claude-code)** for automatic labeling during training (`train --auto`). It uses your Claude Code default model. Without it, you label examples yourself: at a prompt, or by id with `add`.
- `./semlabel` is a wrapper around `src/semlabel.py` and the `.venv` next to it. Symlink it onto your `PATH` to call it from anywhere.

## Compared with Jev

[Jev](https://flaviocopes.com/jev/) from TypeSafe AI is marketed as a "System One" decision model. You send it a piece of text and a question, and it returns a typed answer (yes/no, a choice, or a score) with a probability. Under the "decision engine" name, it's a general classifier that takes its question at call time: in effect a cross-encoder with several output heads, one step above embedding-based RAG. semlabel attacks the same problem, cheap typed decisions over text, from the other end.

### Speed

| | Jev | semlabel |
|---|---|---|
| One record, one question | ~70–500 ms (hosted API) | 5–11 ms to embed + 0.01 ms to score |
| One record, 20 questions | 20 calls | the same embedding + 0.15 ms for all 20 |
| Batch throughput | network-bound | 109 records/s full text, 1,000/s titles (CPU); 16M scoring decisions/s once embedded |
| Re-scoring an archive after changing a concept | full cost again | scoring only, since embeddings are cached |
| Cost | ~$0.042 per million input tokens | local compute, after a few LLM calls for training |

The semlabel numbers are measured on an Apple M5 Pro, CPU only, with real posts averaging 1,070 characters:

| Stage | Time |
|---|---|
| Cold start (import + model load) | 4.2 s once. `embed-server` keeps the model loaded between calls |
| Embed a single record, warm | 11 ms full text, 5 ms title only (p50) |
| Embed in batch | 9 ms/record full text, 1 ms/record title only |
| Score 20 concepts, dot product + conformal | 0.15 ms for a single record, 1.2 µs/record in batch |

Embedding takes nearly all of the time. Once a record is embedded, decisions are practically free.

### Domain calibration

Jev promises calibrated probabilities from one general model, the same weights for every customer, but publishes nothing about in-domain evaluation or fine-tuning. A probability is only accurate relative to the data it was calibrated on. If "relevant", "sensitive" or "allow" is rarer or more common in your domain than in theirs, the same numbers come out systematically too high or too low, and you have no labeled set to check them against.

semlabel is calibrated on **your** data by construction. Each concept's confidence comes from ranking a new score against held-out scores of that concept's own labeled examples (conformal prediction, below). Those examples ship inside the concept file, where you can inspect, correct and retrain them. `show` reports held-out recall and specificity for every concept.

| | Jev | semlabel |
|---|---|---|
| Model | one general hosted model | one small vector per concept, fit on your data |
| Where judgment happens | at every call, in one forward pass | once, offline, in an LLM labeling loop |
| Defining a class | a question at call time; no training | a description + a few minutes of training |
| Output | typed answer + probability (calibration claimed, not shown in-domain) | conformal confidence from held-out, in-domain examples |
| Evidence behind a decision | none exposed | the stored positive and negative examples |
| Changing the taxonomy | edit the question, instantly | retrain the concept (minutes) |
| Deployment | hosted API only | local; data stays on your machine after training |
| Expressiveness | can judge content that needs some reasoning | topical and stylistic directions only |

**Where each fits.** Jev, or an LLM, fits when the question changes on every call and there's nothing to train on, such as "is this chunk relevant to *this* query". semlabel fits when the concept is stable, the volume is high, the data has to stay local, or you need to know why a record got its label. The two also combine: an LLM or Jev labels the training rounds, and semlabel turns those labels into a classifier that costs nothing to run.

Jev figures are from public descriptions of TypeSafe's published numbers, as of September 2026.

## Core idea 1: move the reasoning to training time

An LLM classifier reads every record at inference time, so you pay for reasoning on every item, forever. A single-pass classifier like Jev is fast, but it never reads its own output back. It can't work through a judgment step by step, and whatever it misses stays missed.

semlabel puts the reasoning where it can iterate, and runs it once. Training is active learning, with an LLM (`claude -p`) as the labeler:

1. **Seed.** The concept description is embedded, and the 50 closest records (with near-duplicates removed) go to the LLM, which labels each one positive or negative against the description.
2. **Fit.** A linear classifier is fit on the labels so far (core idea 2).
3. **Refine.** The next 40 records are the ones nearest the current boundary, where the classifier is least sure. The LLM labels them, and the classifier is refit. Each round's labels move the boundary, and the next round asks about the records that move made uncertain. It usually settles in 3 to 5 rounds.

The judgment happens at training time, where a capable model can take its time. What ships to production is a lookup: no LLM call, no network, about 10 ms per record, plus almost nothing for each further concept.

The trade-off is expressiveness. A linear direction captures *topics* and *registers* well, such as "AI hardware news", "clickbait" or "sports coverage". It can't capture judgments that require reasoning about a particular record, like "is this claim true" or "does this contradict the previous post". If a concept stays poor after a few rounds of corrections, it probably isn't linear in the embedding space, and it belongs with an LLM.

## Core idea 2: regression induces concept vectors that act like regular embeddings

A trained linear classifier on embeddings is just its weight vector, and that vector lives in the same space as the embeddings. semlabel fits it with ridge regression on targets +1/−1, with class-balanced sample weights and **no bias term**, then normalizes it to unit length. So the concept *is* an embedding-shaped vector `w`, and a record's raw score is `embedding · w`, a cosine.

- **Use it anywhere an embedding goes.** `w` works as a query vector in any vector index built with the same model. Ranking records by `w` retrieves the concept's clearest members. It often does this better than embedding the description itself. On our data, the "science" concept vector surfaced a receptor study and a new bamboo plastic, while its description, embedded as a query, surfaced an ARC-AGI post and a date header.
- **A direction, not a location.** `w` is nearly orthogonal to the embedding of its own description (cosine about 0.0–0.1). Regression removes what all texts have in common and keeps what separates members from non-members. Absolute scores are small, but the ranking is sharp.
- **Concepts are comparable.** The cosine between two concept vectors measures their overlap. On our concepts, "health" and "science" come out at +0.41, while "slop" and "web media" come out at −0.16. Overlap between concepts is structural (many posts span two topics), not a bug.
- **A boundary, not just a center.** A nearest-centroid approach knows where a topic is. The regression also learns where it *ends*, because most of its labels are hard negatives: records that look related but aren't.

## Core idea 3: each concept carries its own training set

A concept file holds the description, every labeled example, the calibration scores, and `w`. Everything that defines the concept travels with it.

- **Retraining on new data is incremental.** `train concept.json new.jsonl --auto` starts from the stored examples, matched by id or re-embedded from their stored text, and adds boundary cases from the new data. A concept can follow a drifting feed without starting over.
- **Corrections are local.** `add concept.json data.jsonl <id> -` moves one record to the negatives and refits. `--dry-run` shows which records would change sides first.
- **Scores are calibrated by the training set itself (conformal prediction).** Each labeled example gets a *held-out* score from a `w` fit without it (5-fold cross-validation). A new record's raw score `s` is ranked against those held-out scores:
  - **FN side:** the share of known positives that scored ≤ `s`. When it's low, real members rarely score this low.
  - **FP side:** the share of known negatives that scored ≥ `s`. When it's low, non-members rarely score this high.

  `tag` reports the confidence `p_pos / (p_pos + p_neg)`, where `p_pos` and `p_neg` are those two shares, with a shared smoothing term so neither class dominates the tails. It runs from 0 to 1, increases with the raw score, and doesn't depend on how many examples of each class were labeled. Each concept gets its own threshold from its own data: on our concepts, confidence 0.5 falls anywhere from −0.06 to +0.08 in raw cosine. A single global threshold can't make that per-concept correction.

What the confidence means in practice:

- **0.5 is a balanced point.** The score is equally typical of the concept's positives and its negatives. It ignores how common the concept is, so for a rare concept `> 0.5` flags more than you want. Use **`> 0.9`** for precision, and treat 0.5–0.9 as a band to review.
- **It's rank-based, not an exact probability.** Conformal prediction assumes the calibration examples look like the data being tagged. Active learning oversamples boundary cases, so the confidence is sharpest near the boundary and approximate elsewhere. It is still measured on your data, for this concept, and you can check it.
- **The held-out numbers are honest lower bounds.** `show` prints held-out recall and specificity. They're measured on the concept's hardest examples, so accuracy on the full stream is higher.

`tag --raw` outputs the uncalibrated cosine.

## Usage patterns

Anywhere you'd otherwise run an LLM over every item to sort text into stable categories:

- **News and social feeds.** Tag every incoming post against a set of concepts, like "AI hardware", "policy", "clickbait" or "genuine research result", and filter, rank or route on the confidences. Training uses a day or two of the feed. After that, each new day costs about 10 ms per post, and concepts are retrained as the feed drifts.
  ```bash
  ./semlabel tag concepts/ today.jsonl | jq -c 'select(.tags.slop < 0.2 and .tags.ai_hardware > 0.9)'
  ```
- **Agent outputs.** Agents produce far more text than anyone reads: transcripts, tool outputs, final reports, logs. Concepts like "stuck in a retry loop", "refused or gave up", "asks the user for input", "touches credentials" or "off-task" can triage thousands of runs in seconds, pick which transcripts a human or an LLM judge should read, or annotate chunks of context by type before they reach a model. All of these are about the *kind* of text, which is what a linear concept captures well. Whether the agent was *right* is a reasoning question, and still needs a judge.
- **Large text collections.** Archives, support tickets, documents, a scraped corpus, or eval and fine-tuning data. `discover` and `search` show what's in the collection. A handful of concepts slices it, and because embeddings are cached, re-tagging the whole collection after a concept changes takes seconds, not a new LLM bill. The labeled examples each concept accumulates double as a small, curated eval set for that category.

A record is a JSON object with a `title` and optional `comments` (see *Input format*). Map other schemas with one `jq` line.

## Reference

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

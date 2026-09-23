---
name: semlabel
description: >
  Train and apply semantic concept classifiers over JSONL with the semlabel CLI.
  A concept is a plain-language description turned into a vector in embedding space by
  LLM-labeled active learning; tagging new data is one dot product per concept plus a
  conformal confidence. Use when the user wants to label, filter, route or monitor
  JSONL records by topic or style, define or refine a concept, or score new data.
argument-hint: [train|tag|add|show|search|discover|calibrate]
---

# semlabel: concept classifiers you train once and run for free

A **concept** is a JSON file holding a description, the labeled examples that define it,
calibration scores, and a unit vector `w`. Scoring a record means embedding it once and
taking the dot product with `w`. `tag` converts that dot product into a **confidence** in
(0, 1) by ranking it against the concept's own held-out examples (conformal prediction).

The reasoning happens once, at training time, when an LLM labels a few hundred examples.
After that, classification needs no LLM: about 5 to 11 ms per record on CPU, and further
concepts cost almost nothing.

## Invocation

Always call the wrapper script. It runs the right venv and strips the Claude Code
environment variables, so the nested `claude -p` used for labeling works from inside a session.

```bash
SEMLABEL=/path/to/semlabel/semlabel     # the wrapper in the repo root, not src/semlabel.py
$SEMLABEL --help
```

If `.venv/` is missing, run `./install.sh` in the repo first.

## Data

JSONL, one record per line. Only three fields are used, and every other field passes through unchanged:

```json
{"id": "a1b2", "title": "...", "comments": [{"author": "Article", "text": "body..."}, {"author": "x", "text": "..."}]}
```

The embedded text is the title, then `Article`/`Abstract` comments (the body), then the other comments, capped at 2000
characters. Titles alone work. `add` and `search --id` need `id` values. If the user's data has other
field names (for example `text`, `body` or `headline`), convert it first:

```bash
jq -c '{id: .uuid, title: .headline, comments: [{author: "Article", text: .body}]}' raw.jsonl > data.jsonl
```

The first command run on a file writes a `data.jsonl.embeds.npy` cache next to it. If records are edited in place,
delete the cache. Appending records is fine.

## Workflow: train a concept

**1. Survey the data before writing a description.**
```bash
$SEMLABEL discover data.jsonl -n 12 -k 5          # clusters with examples: what's actually in there
$SEMLABEL search "gpu inference cost" data.jsonl -k 10 | jq -r '"\(._score|.*100|round/100)  \(.title)"'
```
If search finds almost nothing relevant, the concept won't train. Positives must exist in the data.

**2. Write the description.** It's the only definition the labeler sees, so write it like a brief for
a human annotator: what counts, what doesn't, and the near misses.

```bash
cat > concepts/ai_hardware.json <<'EOF'
{"description": "New AI accelerator chips, inference hardware, and the cost or speed of running models. NOT: general GPU gaming news, stock-price stories, or model releases that mention hardware only in passing."}
EOF
```
Narrow, concrete descriptions train clean concepts. Broad or sentiment-based ones ("anything negative
about AI") drift. A concept needs about 10 or more distinct positives in the data.

**3. Train.**
```bash
$SEMLABEL train concepts/ai_hardware.json data.jsonl --auto --trace trace.md
```
`--auto` labels with `claude -p`, using the Claude Code default model. Round 1 labels the 50 records closest to the
description. Later rounds label 40 records near the current boundary. The classifier is refit after each round,
and training stops when the direction stabilizes, usually in 3 to 5 rounds. The concept file is saved after
every round. Without `--auto`, training prompts `pos>` for manual labels, which an agent can't answer, so as an agent
always use `--auto` or the `add` workflow below. Training needs a few hundred records or more.

**4. Inspect.**
```bash
$SEMLABEL show concepts/ai_hardware.json
```
Check the following:
- the `calibration:` line: held-out recall and specificity on the labeled set. That set is mostly hard boundary cases, so treat these as lower bounds. Recall below about 0.4 means the concept is too broad, too rare, or not linear.
- positives that don't fit the description (labeler false positives);
- near-duplicate positives, which inflate the count without adding information;
- negatives that are really positives.

**5. Correct.** Fix labels by record id, and preview the effect first:
```bash
$SEMLABEL add concepts/ai_hardware.json data.jsonl a1b2 - c3d4 + --dry-run   # which records would change side
$SEMLABEL add concepts/ai_hardware.json data.jsonl a1b2 - c3d4 +
```
`add` refits `w` and the calibration. Rewording the description and re-running `train` continues from
the existing labels.

### Agent as labeler (no `claude -p`)

You can build a concept entirely yourself, reading records and deciding. This is useful when you
have the context the description lacks, or when `--auto` isn't available:

```bash
$SEMLABEL search "accelerator chip inference" data.jsonl -k 30 > cand.jsonl   # candidates
jq -r '"\(.id)\t\(.title)"' cand.jsonl                                          # read, decide
$SEMLABEL add concepts/ai_hardware.json data.jsonl id1 + id2 + id3 - id4 -      # label by id
$SEMLABEL tag concepts/ai_hardware.json data.jsonl \
  | jq -r 'select(.tags.ai_hardware > 0.3 and .tags.ai_hardware < 0.7) | "\(.id)\t\(.title)"'   # uncertain ones
```
Repeat: label the uncertain records, `add`, re-tag. That's the same active-learning loop `train` runs.
Include hard negatives, meaning records that look related but aren't. They define the boundary. Aim for
20 or more positives and at least as many negatives.

## Workflow: apply to new data

```bash
$SEMLABEL tag concepts/ new.jsonl > tagged.jsonl                         # every trained concept in the dir
$SEMLABEL tag concepts/ new.jsonl | jq -c 'select(.tags.ai_hardware > 0.9)'   # high-precision filter
cat new.jsonl | $SEMLABEL tag concepts/ai_hardware.json --tag-name labels  # stdin, custom field
```

The output is each input record with `"tags": {"concept": confidence}` added.

- **Confidence** is `p_pos / (p_pos + p_neg)`: how typical the raw score is of the concept's held-out positives
  compared with its held-out negatives. It increases with the raw score and doesn't depend on class balance.
- **0.5** is where a score is equally typical of both classes. It ignores base rates, so for rare concepts
  `> 0.5` over-selects. Use **> 0.9** for precision, and 0.5 to 0.9 as a "maybe" band to review.
- `--raw` outputs the cosine instead: in [-1, 1], with the boundary at 0.
- `tag` reads all of its input before writing anything. It batches; it doesn't stream.
- For repeated calls, run `$SEMLABEL embed-server &` so the model stays loaded, which saves about 4 s per call.
  `embed-stop` stops it.

Concept files from before calibration existed: run `$SEMLABEL calibrate concepts/` once. Otherwise `tag`
recomputes the calibration on every run.

**Drift:** when a new batch of data looks different, spot-check its `0.5 < conf < 0.9` band. If
too many labels there are wrong, run `train concepts/x.json new.jsonl --auto` to add labels from the new data.
Training continues from the stored examples, so the concept keeps what it already learned.

## Concept file

```json
{
  "description": "...",
  "model": "Snowflake/snowflake-arctic-embed-xs",
  "trained_at": "2026-09-23T10:00:00Z",
  "positives": [{"id": "a1b2", "text": "..."}],
  "negatives": [{"id": "c3d4", "text": "..."}],
  "n_positive": 40, "n_negative": 180,
  "calibration": {"method": "conformal, 5-fold out-of-fold", "pos": [...], "neg": [...]},
  "weight": [384 floats]
}
```
Concepts are tied to the embedding model that trained them. After changing `MODEL_NAME`, retrain every concept and
delete the `.embeds.npy` caches.

## When not to use it

A concept is a linear direction. It captures topics, registers and styles ("clickbait", "AI hardware",
"sports coverage"). It can't capture judgments that need reasoning about the specific content ("is this claim
true?", "does this contradict the previous post?"). If a concept stays poor after a few rounds of
corrections, say so and use an LLM for that decision instead.

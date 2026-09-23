import hashlib
import json
import os
import re
import subprocess
import tempfile

import numpy as np

MODEL_NAME = "Snowflake/snowflake-arctic-embed-xs"
# Arctic-embed is asymmetric: short queries matched against records need this prefix
# (the model's own "query" prompt). Records are embedded without it. Set to "" for
# symmetric models such as all-MiniLM-L6-v2.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_SOCKET = "/tmp/semlabel-embed.sock"
EMBED_DTYPE = np.dtype([('offset', 'i8'), ('embedding', 'f4', (384,))])

_model = None


def read_jsonl(path):
    """Read JSONL file, returning (records, offsets). Offsets are byte positions for seeking."""
    records = []
    offsets = []
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
                offsets.append(offset)
    return records, offsets


def build_text(record):
    """Build text for embedding from a record: title + article body + comment texts."""
    parts = []
    title = record.get("title", "")
    if title:
        parts.append(title)
    # Article/Abstract comments first (body text), then regular comments
    articles = []
    comments = []
    for comment in record.get("comments", []):
        text = comment.get("text", "")
        if not text:
            continue
        if comment.get("author") in ("Article", "Abstract"):
            articles.append(text)
        else:
            comments.append(text)
    parts.extend(articles)
    parts.extend(comments)
    return "\n".join(parts)[:2000]


def _npy_path(json_cache_path):
    """Convert .embeds path to .npy path."""
    return json_cache_path.replace(".embeds", ".embeds.npy")


def _load_cache(path):
    """Load embedding cache. Tries .npy first, falls back to JSON .embeds."""
    npy = _npy_path(path) if path else None
    if npy and os.path.exists(npy):
        arr = np.load(npy, allow_pickle=False)
        if 'offset' in arr.dtype.names:
            return {str(row['offset']): row['embedding'].copy() for row in arr}
        else:
            return {str(row['hash']): row['embedding'].copy() for row in arr}
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError, KeyError):
        return {}


def _save_npy(npy_path, cache):
    """Save cache dict to .npy structured array. Atomic write."""
    arr = np.empty(len(cache), dtype=EMBED_DTYPE)
    for i, (k, v) in enumerate(cache.items()):
        arr[i]['offset'] = int(k) if k.isdigit() else hash(k) % (2**63)
        arr[i]['embedding'] = v
    dir_name = os.path.dirname(os.path.abspath(npy_path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, npy_path)
    except:
        os.unlink(tmp_path)
        raise


def _save_cache(path, cache):
    """Save embedding cache. Writes .npy when keys are offsets (numeric), always writes JSON."""
    # Only write .npy if keys look like offsets (numeric)
    offset_cache = {k: v for k, v in cache.items() if k.isdigit()}
    if offset_cache:
        _save_npy(_npy_path(path), offset_cache)

    # Always save JSON
    data = {k: v.tolist() for k, v in cache.items()}
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except:
        os.unlink(tmp_path)
        raise


EMBED_PID_FILE = "/tmp/semlabel-embed.pid"
EMBED_TTL = 24 * 3600  # daemon auto-exits after 24h


def _daemon_alive():
    """Check if a daemon is already running. Returns pid if alive, None otherwise."""
    if not os.path.exists(EMBED_PID_FILE):
        return None
    try:
        with open(EMBED_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ProcessLookupError, ValueError, OSError):
        # Stale PID file — clean up
        for p in (EMBED_PID_FILE, EMBED_SOCKET):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return None


def _start_daemon():
    """Fork an embed daemon in the background. Returns True if started."""
    import sys

    # Already running?
    if _daemon_alive():
        return True

    # Find our own script path
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semlabel.py")
    python = sys.executable
    pid = os.fork()
    if pid == 0:
        # Child: detach and exec
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        log = os.open("/tmp/semlabel-embed.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log, 2)
        os.close(devnull)
        os.close(log)
        os.execv(python, [python, script, "embed-server"])
    else:
        # Parent: wait for PID file (written early by server) then socket
        import time
        for _ in range(100):  # up to 10s (model load can be slow)
            time.sleep(0.1)
            if os.path.exists(EMBED_SOCKET):
                # Socket exists — try connecting to confirm it's accepting
                import socket
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(EMBED_SOCKET)
                    s.close()
                    return True
                except (socket.error, OSError):
                    pass  # socket bound but not yet accepting — keep waiting
        return False


def _embed_via_daemon(texts):
    """Try to embed via running daemon. Auto-starts daemon if not running. Returns np.ndarray or None."""
    import socket
    import struct

    for attempt in range(3):  # try connect, maybe start, retry with patience
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30)  # model load can take time on first request
            sock.connect(EMBED_SOCKET)
            payload = json.dumps(texts).encode()
            sock.sendall(struct.pack("!I", len(payload)))
            sock.sendall(payload)
            count_data = _recv_exact(sock, 4)
            count = struct.unpack("!I", count_data)[0]
            nbytes = count * 384 * 4
            data = _recv_exact(sock, nbytes)
            sock.close()
            return np.frombuffer(data, dtype=np.float32).reshape(count, 384).copy()
        except (socket.error, ConnectionRefusedError, FileNotFoundError, OSError):
            if attempt == 0 and not _daemon_alive():
                import sys
                print("  starting embed daemon...", file=sys.stderr, flush=True)
                if not _start_daemon():
                    return None
            elif attempt < 2:
                # Daemon exists but not ready yet — wait a bit
                import time
                time.sleep(1)
            else:
                return None
    return None


def _recv_exact(sock, nbytes):
    """Receive exactly nbytes from socket."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def embed(texts, cache_path=None, keys=None):
    """Embed texts, using cache if provided. Only loads model if there are cache misses.

    keys: optional list of cache keys (e.g. str(offset)). If None, uses sha256 of text.
    """
    global _model

    # Use provided keys or fall back to content hashes
    hashes = keys if keys else [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    # Load cache and find misses
    cache = _load_cache(cache_path) if cache_path else {}
    miss_indices = [i for i, h in enumerate(hashes) if h not in cache]

    # Encode only misses
    if miss_indices:
        import sys
        miss_texts = [texts[i] for i in miss_indices]

        # Try daemon first
        daemon_result = _embed_via_daemon(miss_texts)
        if daemon_result is not None:
            print(f"  embedded {len(miss_indices)}/{len(texts)} via daemon", file=sys.stderr)
            miss_embeddings = daemon_result
        else:
            if _model is None:
                import io
                from sentence_transformers import SentenceTransformer
                print(f"  loading embed model ({len(miss_indices)}/{len(texts)} uncached)...", file=sys.stderr, flush=True)
                _real_stderr = sys.stderr
                sys.stderr = io.StringIO()
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
                sys.stderr = _real_stderr
            else:
                print(f"  embedding {len(miss_indices)}/{len(texts)} uncached...", file=sys.stderr)
            miss_embeddings = _model.encode(miss_texts, normalize_embeddings=True, show_progress_bar=False)

        for j, i in enumerate(miss_indices):
            cache[hashes[i]] = miss_embeddings[j]
        if cache_path:
            _save_cache(cache_path, cache)
    elif cache_path and keys and not os.path.exists(_npy_path(cache_path)):
        # Warm cache but .npy doesn't exist yet — migrate from hash-only JSON
        _save_cache(cache_path, cache)

    # Assemble result in order
    return np.array([cache[h] for h in hashes])


def save_concept(path, weight, positives, negatives, description="", calibration=None):
    """Save a concept to its own JSON file. Weight is written last for readability.

    Args:
        path: path to concept JSON file (filename stem = concept name)
        positives: list of {"id": str, "text": str} — positive support vectors
        negatives: list of {"id": str, "text": str} — negative support vectors
        calibration: out-of-fold scores from _calibrate(), used for conformal scoring
    """
    from datetime import datetime, timezone
    entry = {
        "description": description,
        "model": MODEL_NAME,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "positives": positives,
        "negatives": negatives,
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }
    if calibration:
        entry["calibration"] = calibration
    entry["weight"] = weight.tolist() if isinstance(weight, np.ndarray) else list(weight)
    _write_json_atomic(path, entry)


def _write_json_atomic(path, entry):
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, indent=2)
        os.replace(tmp_path, path)
    except:
        os.unlink(tmp_path)
        raise


def show_command(path):
    """Show a compact summary of concepts. Path can be a single concept file or a directory."""
    import sys

    if not os.path.exists(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(path):
        import glob as glob_mod
        files = sorted(glob_mod.glob(os.path.join(path, "*.json")))
    else:
        files = [path]

    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        c = load_concept(fpath)
        desc = c.get("description", "")
        n_pos = c.get("n_positive", len(c.get("positives", [])))
        n_neg = c.get("n_negative", len(c.get("negatives", [])))
        w_dim = len(c.get("weight", [])) if "weight" in c else 0
        print(f"\033[1;32m=== {name} ({n_pos}+/{n_neg}-) dim={w_dim} ===\033[0m", file=sys.stderr)
        if desc:
            print(f"\033[32m  {desc}\033[0m", file=sys.stderr)
        cal = c.get("calibration")
        if cal:
            recall = sum(s > 0 for s in cal["pos"]) / len(cal["pos"])
            specificity = sum(s <= 0 for s in cal["neg"]) / len(cal["neg"])
            print(f"  calibration: {cal['method']}, held-out recall {recall:.2f}, "
                  f"specificity {specificity:.2f}", file=sys.stderr)
        elif "weight" in c:
            print("  calibration: none (run `semlabel calibrate`)", file=sys.stderr)
        for i, sv in enumerate(c.get("positives", []), 1):
            txt = sv.get("text", "")[:250].replace("\n", " ")
            print(f"  \033[32m{i:3d} POS\033[0m {txt}", file=sys.stderr)
        for i, sv in enumerate(c.get("negatives", []), 1):
            txt = sv.get("text", "")[:250].replace("\n", " ")
            print(f"  \033[31m{i:3d} NEG\033[0m {txt}", file=sys.stderr)
        print(file=sys.stderr)


def calibrate_command(path):
    """Compute and store conformal calibration for trained concepts (file or directory)."""
    import sys
    import glob as glob_mod

    files = sorted(glob_mod.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath) as f:
            data = json.load(f)
        if not data.get("weight"):
            print(f"  {name}: untrained, skipped", file=sys.stderr)
            continue
        calibration = _calibrate_from_examples(data)
        if calibration is None:
            print(f"  {name}: fewer than 2 examples per class, skipped", file=sys.stderr)
            continue
        weight = data.pop("weight")
        data["calibration"] = calibration
        data["weight"] = weight  # keep weight last
        _write_json_atomic(fpath, data)
        print(f"  {name}: {len(calibration['pos'])}+/{len(calibration['neg'])}- ({calibration['method']})",
              file=sys.stderr)


def tag_command(concepts_path, data_path=None, no_cache=False, tag_name="tags", raw=False):
    """Read JSONL from file or stdin, score against concepts, write tagged JSONL to stdout.
    concepts_path can be a directory (loads all trained concepts) or a single concept file.
    Scores are conformal confidences in (0, 1); raw=True outputs cosine scores instead."""
    import sys

    if os.path.isdir(concepts_path):
        concepts = load_concepts_dir(concepts_path)
    else:
        name = os.path.splitext(os.path.basename(concepts_path))[0]
        c = load_concept(concepts_path)
        if "weight" not in c:
            raise ValueError(f"{concepts_path} has no trained weight")
        concepts = {name: c}

    # Read all records
    records = []
    offsets = []
    if data_path:
        records, offsets = read_jsonl(data_path)
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    # Embed all records using build_text
    texts = [build_text(rec) for rec in records]
    cache_path = (data_path + ".embeds") if data_path and not no_cache else None
    embeddings = embed(texts, cache_path=cache_path, keys=[str(o) for o in offsets] if offsets else None)

    # Score each concept
    for name, c in concepts.items():
        scores = embeddings @ c["weight"]
        if not raw:
            calibration = c.get("calibration") or _calibrate_from_examples(c)
            if calibration is None:
                print(f"warning: {name} has no calibration and too few examples; writing raw scores",
                      file=sys.stderr)
            else:
                if "calibration" not in c:
                    print(f"note: {name} has no stored calibration, computed from its examples "
                          f"(run `semlabel calibrate` to save it)", file=sys.stderr)
                scores = conformal_confidence(scores, calibration)
        for i, score in enumerate(scores):
            records[i].setdefault(tag_name, {})[name] = float(score)

    # Write output
    for rec in records:
        print(json.dumps(rec), file=sys.stdout)


def load_concept(path):
    """Load a single concept from its JSON file, converting weight to numpy array."""
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    concept = {
        "description": data.get("description", ""),
        "positives": data.get("positives", []),
        "negatives": data.get("negatives", []),
        "n_positive": int(data.get("n_positive", len(data.get("positives", [])))),
        "n_negative": int(data.get("n_negative", len(data.get("negatives", [])))),
    }
    if "weight" in data and len(data["weight"]) > 0:
        concept["weight"] = np.array(data["weight"])
    if data.get("calibration"):
        concept["calibration"] = data["calibration"]
    return concept


def load_concepts_dir(dir_path):
    """Load all trained concepts from a directory. Returns {name: concept_dict}.
    Skips files without a weight (untrained concepts)."""
    import glob as glob_mod
    concepts = {}
    for path in sorted(glob_mod.glob(os.path.join(dir_path, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        concept = load_concept(path)
        if "weight" in concept:
            concepts[name] = concept
    return concepts




def ask_llm(description, texts, trace_path=None):
    """Ask claude to label texts as matching a topic description. Returns set of positive indices (0-indexed)."""
    # Flatten each item to single line — embedded newlines break the numbered list
    flat = [t[:200].replace("\n", " ") for t in texts]
    numbered = "\n".join(f"{i+1}. {flat[i]}" for i in range(len(flat)))
    prompt = (
        f"You are labeling training examples for a semantic concept classifier. "
        f"Your positive/negative picks become support vectors that define the concept direction in embedding space. "
        f"Precision matters more than recall — a false positive is worse than a missed true positive.\n\n"
        f"Topic: {description}\n\n"
        f"Here are {len(texts)} items:\n{numbered}\n\n"
        f"Classify each item as positive (matches topic) or negative (does not match). "
        f"Reply in exactly this format:\npos: 2, 6, 36\nneg: 1, 3, 4, 5\n"
        f"List ALL item numbers — every item must appear in exactly one line."
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude failed (exit {result.returncode}): {result.stderr.strip()}")

    stdout = result.stdout.strip()

    if trace_path:
        with open(trace_path, "a") as tf:
            tf.write(f"## Prompt\n\n{prompt}\n\n## Response\n\n{stdout}\n\n---\n\n")

    if not stdout or stdout.lower() == "none":
        return set()

    # Parse pos: line from response
    selected = set()
    pos_match = re.search(r"(?i)^pos:[^\S\n]*([^\n]*)", stdout, re.MULTILINE)
    if pos_match is not None and re.search(r"(?i)^neg:", stdout, re.MULTILINE):
        for m in re.finditer(r"\d+", pos_match.group(1)):
            idx = int(m.group()) - 1  # 1-indexed → 0-indexed
            if 0 <= idx < len(texts):
                selected.add(idx)
    else:
        # Fallback: treat whole response as comma-separated numbers
        for m in re.finditer(r"\d+", stdout):
            idx = int(m.group()) - 1
            if 0 <= idx < len(texts):
                selected.add(idx)

    return selected


def _dedup_indices(embeddings, indices, threshold=0.80):
    """Filter out near-duplicate items. Greedy: keep first, skip if cosine > threshold to any kept."""
    kept = []
    for i in indices:
        if any(float(embeddings[i] @ embeddings[k]) > threshold for k in kept):
            continue
        kept.append(i)
    return kept


def _short_id(record):
    """First 4 chars of record's id field."""
    return str(record.get("id", ""))[:4]


def _show_items(records, indices, scores=None, out=None):
    """Print items with short IDs to out (defaults to stderr)."""
    import sys
    out = out or sys.stderr
    for i in indices:
        sid = _short_id(records[i])
        title = records[i].get("title", "")[:80]
        if scores is not None:
            print(f"{sid}  [{scores[i]:.2f}]  {title}", file=out)
        else:
            print(f"{sid}  {title}", file=out)


def _read_positive_ids(candidate_indices, records, input_fn=None):
    """Read a line of space-separated short IDs, return set of matching global indices."""
    import sys
    if input_fn:
        line = input_fn()
    else:
        try:
            line = input("pos> ")
        except EOFError:
            return None  # signal abort

    tokens = line.strip().split()
    if not tokens:
        return set()

    # Build lookup: short_id -> global index (only among candidates)
    lookup = {}
    for i in candidate_indices:
        sid = _short_id(records[i])
        lookup[sid] = i

    positive = set()
    for tok in tokens:
        if tok in lookup:
            positive.add(lookup[tok])
    return positive


def train_command(data_path, concept_path, input_fn=None, auto=False, trace_path=None, no_cache=False, log_texts_path=None):
    """Train a concept vector via interactive active learning and save it."""
    import sys

    # Derive concept name from filename, load description from concept file
    concept = os.path.splitext(os.path.basename(concept_path))[0]
    existing_concept = load_concept(concept_path)
    description = existing_concept.get("description", "")
    if not description:
        raise ValueError(f"No description in {concept_path}")

    # Load JSONL and build texts
    records, offsets = read_jsonl(data_path)

    texts = [build_text(r) for r in records]
    if not texts:
        raise ValueError(f"No records found in {data_path}")

    if log_texts_path:
        with open(log_texts_path, "w") as lf:
            for i, text in enumerate(texts):
                sid = _short_id(records[i])
                lf.write(f"## {i} [{sid}]\n{text}\n---\n")
        print(f"Wrote {len(texts)} texts to {log_texts_path}", file=sys.stderr)

    cache_path = (data_path + ".embeds") if not no_cache else None
    offset_keys = [str(o) for o in offsets]
    embeddings = embed(texts, cache_path=cache_path, keys=offset_keys)

    # Embed seed description (no offset — uses content hash)
    seed_emb = embed([QUERY_PREFIX + description], cache_path=cache_path)[0]

    # Load existing labels if continuing training
    labels = {}  # index -> bool
    weight = None
    existing_ids = set()  # IDs already labeled from previous training

    if "weight" in existing_concept:
        # Build ID-to-index lookup for current dataset
        id_to_idx = {}
        for i, r in enumerate(records):
            rid = r.get("id", "")
            if rid:
                id_to_idx[rid] = i

        # Match existing vectors to current dataset by ID
        n_matched = 0
        extra_texts = []
        extra_labels = []
        for sv in existing_concept.get("positives", []):
            rid = sv.get("id", "")
            if rid in id_to_idx:
                labels[id_to_idx[rid]] = True
                existing_ids.add(id_to_idx[rid])
                n_matched += 1
            elif sv.get("text"):
                extra_texts.append(sv["text"])
                extra_labels.append(True)
        for sv in existing_concept.get("negatives", []):
            rid = sv.get("id", "")
            if rid in id_to_idx:
                labels[id_to_idx[rid]] = False
                existing_ids.add(id_to_idx[rid])
                n_matched += 1
            elif sv.get("text"):
                extra_texts.append(sv["text"])
                extra_labels.append(False)

        # Append unmatched vectors as extra rows
        if extra_texts:
            extra_embs = embed(extra_texts, cache_path=cache_path)
            base = len(records)
            for j, (txt, lbl) in enumerate(zip(extra_texts, extra_labels)):
                idx = base + j
                records.append({"id": "", "title": ""})
                texts.append(txt)
                labels[idx] = lbl
                existing_ids.add(idx)
            embeddings = np.vstack([embeddings, extra_embs])

        # Fit initial weight from existing labels
        if len(set(labels.values())) >= 2:
            weight = _fit_classifier(embeddings, labels, quiet=True)

    # Header: concept name + existing vector counts
    n_prev_pos = sum(1 for v in labels.values() if v)
    n_prev_neg = sum(1 for v in labels.values() if not v)
    if labels:
        print(f"\033[1m=== {concept} ({n_prev_pos}+/{n_prev_neg}- existing, {len(texts)} records) ===\033[0m", file=sys.stderr)
    else:
        print(f"\033[1m=== {concept} (new, {len(texts)} records) ===\033[0m", file=sys.stderr)

    for round_num in range(1, 6):
        if round_num == 1 and weight is not None:
            # Skip seed round — use boundary sampling with existing weight
            candidate_indices = select_samples(
                embeddings @ weight, set(labels.keys()), 2, len(texts), n=40
            )
            if not candidate_indices:
                print(f"  R{round_num} no unlabeled samples left", file=sys.stderr)
                break
            candidate_indices = _dedup_indices(embeddings, candidate_indices)
        elif round_num == 1:
            # Round 1: top 50 by seed similarity
            sims = embeddings @ seed_emb
            ranked = np.argsort(-sims)
            candidate_indices = ranked[:50].tolist() if auto else _dedup_indices(embeddings, ranked[:50].tolist())
        elif weight is not None:
            # Round 2+: boundary cases near 0
            candidate_indices = select_samples(
                embeddings @ weight, set(labels.keys()), round_num, len(texts), n=40
            )
            if not candidate_indices:
                print(f"  R{round_num} no unlabeled samples left", file=sys.stderr)
                break
            if not auto:
                candidate_indices = _dedup_indices(embeddings, candidate_indices)
        else:
            # No weight yet (only one class seen) — use seed similarity, skip already labeled
            sims = embeddings @ seed_emb
            ranked = np.argsort(-sims)
            candidate_indices = [i for i in ranked if i not in labels][:50]

        # Label candidates
        if auto:
            candidate_texts = [texts[i] for i in candidate_indices]
            llm_positives = ask_llm(description, candidate_texts, trace_path=trace_path)
            positive_set = {candidate_indices[j] for j in llm_positives}
        else:
            # Show candidates for manual labeling
            _show_items(records, candidate_indices,
                        scores=(embeddings @ weight) if weight is not None else None)
            positive_set = _read_positive_ids(candidate_indices, records, input_fn=input_fn)
            if positive_set is None:
                print("\nAborted.", file=sys.stderr)
                return

        # Label: cited = positive, rest of candidates = negative
        for i in candidate_indices:
            labels[i] = i in positive_set

        # Show labeled items
        for i in candidate_indices:
            if labels[i]:
                tag = "\033[32mPOS\033[0m"
            else:
                tag = "\033[31mNEG\033[0m"
            txt = texts[i][:250].replace("\n", " ")
            print(f"  R{round_num} {tag} {txt}", file=sys.stderr)

        n_pos = sum(1 for v in labels.values() if v)
        n_neg = sum(1 for v in labels.values() if not v)

        if n_pos == 0:
            if round_num == 1:
                print(f"  R{round_num} 0 positives found — skipping concept", file=sys.stderr)
                return
            # Later rounds: all negative is fine, just means boundary is clean

        # Need both classes to fit
        if len(set(labels.values())) < 2:
            continue

        # Fit classifier
        prev_weight = weight
        weight = _fit_classifier(embeddings, labels)
        if weight.ndim == 0 or np.linalg.norm(weight) < 1e-10:
            weight = seed_emb  # degenerate fit — fall back to seed

        # Save after each round — protects against crashes/timeouts mid-training
        positives = [{"id": records[i].get("id", ""), "text": texts[i]} for i in sorted(labels) if labels[i]]
        negatives = [{"id": records[i].get("id", ""), "text": texts[i]} for i in sorted(labels) if not labels[i]]
        save_concept(concept_path, weight, positives, negatives, description=description,
                     calibration=_calibrate(embeddings, labels))

        # Convergence check
        if prev_weight is not None:
            cos_sim = float(np.dot(weight, prev_weight))
            if cos_sim > 0.95:
                print(f"\033[32m  R{round_num} converged (cosine={cos_sim:.4f}), {n_pos}+/{n_neg}-\033[0m", file=sys.stderr)
                break
            else:
                print(f"\033[33m  R{round_num} {n_pos}+/{n_neg}- cosine={cos_sim:.4f}\033[0m", file=sys.stderr)
        else:
            print(f"\033[33m  R{round_num} {n_pos}+/{n_neg}-\033[0m", file=sys.stderr)

    if weight is None:
        print(f"  no classifier trained (need both positive and negative labels)", file=sys.stderr)
        return
    print(f"  saved {n_pos}+/{n_neg}-", file=sys.stderr)


def _fit_classifier(embeddings, labels, alpha=1.0, quiet=False):
    """Fit concept direction via ridge regression with class balancing. Returns unit-norm weight vector."""
    indices = sorted(labels.keys())
    X = embeddings[indices]
    y = np.array([1.0 if labels[i] else -1.0 for i in indices])

    n_pos = int((y > 0).sum())
    n_neg = len(y) - n_pos
    n = len(y)

    # Class-balanced sample weights: minority class weighs more
    sample_weights = np.where(y > 0, n / (2 * n_pos), n / (2 * n_neg))
    W_sqrt = np.sqrt(sample_weights)

    # Apply sample weights
    X_w = X * W_sqrt[:, None]
    y_w = y * W_sqrt

    # Ridge: (X_w^T X_w + αI)^{-1} X_w^T y_w
    reg = alpha * np.eye(X_w.shape[1])
    w = np.linalg.solve(X_w.T @ X_w + reg, X_w.T @ y_w)

    # Normalize to unit norm — scores become cosine similarities
    weight = w / np.linalg.norm(w)

    return weight


CALIB_FOLDS = 5


def _calibrate(embeddings, labels, k=CALIB_FOLDS, seed=0):
    """Out-of-fold scores of the labeled examples, the calibration set for conformal scoring.

    Stratified k-fold: each example is scored by a weight fit without it, so its score is
    what an unseen record would get (in-sample scores sit too far from the boundary).
    Returns {"method", "pos": sorted scores, "neg": sorted scores}, or None if either
    class has fewer than 2 examples.
    """
    idx = [int(i) for i in sorted(labels)]
    pos = [i for i in idx if labels[i]]
    neg = [i for i in idx if not labels[i]]
    k = min(k, len(pos), len(neg))
    if k < 2:
        return None

    rng = np.random.default_rng(seed)
    fold = {}
    for group in (pos, neg):
        for j, i in enumerate(rng.permutation(group)):
            fold[int(i)] = j % k

    oof = {}
    for f in range(k):
        w = _fit_classifier(embeddings, {i: labels[i] for i in idx if fold[i] != f}, quiet=True)
        for i in idx:
            if fold[i] == f:
                oof[i] = float(embeddings[i] @ w)

    return {
        "method": f"conformal, {k}-fold out-of-fold",
        "pos": sorted(round(oof[i], 5) for i in pos),
        "neg": sorted(round(oof[i], 5) for i in neg),
    }


def conformal_pvalues(scores, calibration):
    """Rank raw scores against the calibration set. Returns (p_pos, p_neg) arrays.

    p_pos: share of held-out positives scoring <= s. Low means real matches rarely score
           this low: calling it positive would be unusual (false-negative side).
    p_neg: share of held-out negatives scoring >= s. Low means non-matches rarely score
           this high (false-positive side).
    """
    scores = np.asarray(scores, dtype=float)
    pos = np.asarray(calibration["pos"])
    neg = np.asarray(calibration["neg"])
    p_pos = (np.searchsorted(pos, scores, side="right") + 1) / (len(pos) + 1)
    p_neg = (len(neg) - np.searchsorted(neg, scores, side="left") + 1) / (len(neg) + 1)
    return p_pos, p_neg


def conformal_confidence(scores, calibration):
    """Confidence in (0, 1) that a score belongs to the positive class: p_pos / (p_pos + p_neg).

    Uses the same rank fractions as conformal_pvalues but with one smoothing term shared by
    both classes, eps = 1/(n_min + 1). With per-class +1 smoothing the floors would differ
    (1/31 vs 1/301 for 30+/300-), and a score outside both calibration sets would land at
    0.91 instead of 0.5: class imbalance leaking back in. With a shared eps the result is
    monotone in the raw score and independent of class balance; 0.5 is where a score is
    equally typical of both classes.
    """
    scores = np.asarray(scores, dtype=float)
    pos = np.asarray(calibration["pos"])
    neg = np.asarray(calibration["neg"])
    eps = 1.0 / (min(len(pos), len(neg)) + 1)
    f_pos = np.searchsorted(pos, scores, side="right") / len(pos) + eps
    f_neg = (len(neg) - np.searchsorted(neg, scores, side="left")) / len(neg) + eps
    return f_pos / (f_pos + f_neg)


def _calibrate_from_examples(concept):
    """Build a calibration set by embedding a concept's stored positive/negative texts."""
    pos = [sv["text"] for sv in concept.get("positives", []) if sv.get("text")]
    neg = [sv["text"] for sv in concept.get("negatives", []) if sv.get("text")]
    if len(pos) < 2 or len(neg) < 2:
        return None
    embeddings = embed(pos + neg)
    labels = {i: i < len(pos) for i in range(len(pos) + len(neg))}
    return _calibrate(embeddings, labels)


def select_samples(scores, labeled_indices, round_num, n_total, n=40):
    """Select sample indices for labeling.

    Round 1: random from unlabeled.
    Round 2+: boundary cases near decision threshold (0), balanced above/below.
    """
    unlabeled = [i for i in range(n_total) if i not in labeled_indices]
    if not unlabeled:
        return []

    if round_num == 1 or scores is None:
        rng = np.random.default_rng()
        chosen = rng.choice(unlabeled, size=min(n, len(unlabeled)), replace=False)
        return chosen.tolist()

    # Boundary: split into above/below 0, take n/2 closest from each side
    half = n // 2
    above = [(i, float(scores[i])) for i in unlabeled if scores[i] >= 0]
    below = [(i, float(scores[i])) for i in unlabeled if scores[i] < 0]
    above.sort(key=lambda x: x[1])       # closest to 0 first
    below.sort(key=lambda x: -x[1])      # closest to 0 first
    picked = [i for i, _ in above[:half]] + [i for i, _ in below[:half]]
    # If one side has fewer, fill from the other
    if len(above) < half:
        extra = half - len(above)
        picked = [i for i, _ in above] + [i for i, _ in below[:half + extra]]
    elif len(below) < half:
        extra = half - len(below)
        picked = [i for i, _ in above[:half + extra]] + [i for i, _ in below]
    return picked[:n]


def add_command(concept_path, data_path, pairs, no_cache=False, dry_run=False):
    """Add records from JSONL to a concept's positives or negatives, recompute weight, show reclassification diff.

    pairs: list of (record_id, is_positive) tuples.
    """
    import sys

    # Load concept and capture old weight
    concept = load_concept(concept_path)
    description = concept.get("description", "")
    positives = concept.get("positives", [])
    negatives = concept.get("negatives", [])
    old_weight = concept.get("weight")

    # Load all records from JSONL for ID lookup
    all_records = {}
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", ""))
            if rid:
                all_records[rid] = r

    # Add each pair (dedup: remove from old list before appending to new)
    for record_id, label in pairs:
        rid = str(record_id)
        if rid not in all_records:
            raise ValueError(f"Record {rid} not found in {data_path}")
        record = all_records[rid]
        text = build_text(record)
        entry = {"id": record.get("id", ""), "text": text}
        positives = [sv for sv in positives if str(sv.get("id", "")) != rid]
        negatives = [sv for sv in negatives if str(sv.get("id", "")) != rid]
        if label:
            positives.append(entry)
        else:
            negatives.append(entry)
        tag = "+" if label else "-"
        print(f"  {tag} {record_id}", file=sys.stderr)

    # Recompute weight if we have both classes
    if not (positives and negatives):
        if not dry_run:
            save_concept(concept_path, concept.get("weight", np.zeros(384)), positives, negatives, description=description)
        print(f"  → {os.path.basename(concept_path)} ({len(positives)}+/{len(negatives)}-)", file=sys.stderr)
        return

    all_texts = [sv["text"] for sv in positives] + [sv["text"] for sv in negatives]
    cache_path = (data_path + ".embeds") if not no_cache else None
    sv_embeddings = embed(all_texts, cache_path=cache_path)
    labels_map = {}
    for i in range(len(positives)):
        labels_map[i] = True
    for i in range(len(positives), len(all_texts)):
        labels_map[i] = False
    new_weight = _fit_classifier(sv_embeddings, labels_map, quiet=True)

    # Reclassification diff: score all source records with old and new weight
    if old_weight is not None:
        records, rc_offsets = read_jsonl(data_path)
        texts = [build_text(r) for r in records]
        embeddings = embed(texts, cache_path=cache_path, keys=[str(o) for o in rc_offsets])

        old_scores = embeddings @ old_weight
        new_scores = embeddings @ new_weight

        # Find boundary crossings
        gained = []  # was negative, now positive
        lost = []    # was positive, now negative
        for i in range(len(records)):
            if old_scores[i] <= 0 and new_scores[i] > 0:
                gained.append((new_scores[i], i))
            elif old_scores[i] > 0 and new_scores[i] <= 0:
                lost.append((old_scores[i], i))

        if gained or lost:
            prefix = "[dry-run] " if dry_run else ""
            if gained:
                gained.sort(reverse=True)
                print(f"\n{prefix}Gained ({len(gained)} now positive):", file=sys.stderr)
                for score, i in gained:
                    sid = _short_id(records[i])
                    txt = texts[i][:250].replace("\n", " ")
                    print(f"  \033[32m+{score:+.3f}\033[0m [{sid}] {txt}", file=sys.stderr)
            if lost:
                lost.sort(reverse=True)
                print(f"\n{prefix}Lost ({len(lost)} now negative):", file=sys.stderr)
                for score, i in lost:
                    sid = _short_id(records[i])
                    txt = texts[i][:250].replace("\n", " ")
                    print(f"  \033[31m-{score:+.3f}\033[0m [{sid}] {txt}", file=sys.stderr)
        else:
            print(f"\n{'[dry-run] ' if dry_run else ''}No reclassifications.", file=sys.stderr)

    if not dry_run:
        save_concept(concept_path, new_weight, positives, negatives, description=description,
                     calibration=_calibrate(sv_embeddings, labels_map))

    mode = " (dry-run)" if dry_run else ""
    print(f"  → {os.path.basename(concept_path)} ({len(positives)}+/{len(negatives)}-){mode}", file=sys.stderr)


def search_command(query, data_path, k=10, no_cache=False, by_id=False):
    """Semantic search over JSONL records. Query can be text or a record ID (with --id flag).
    Scores by cosine similarity, outputs top K as JSONL to stdout."""
    import sys

    # Fast path: if .npy index exists and not --id mode, skip full JSONL load
    npy_path = _npy_path(data_path + ".embeds") if not no_cache else None
    if npy_path and os.path.exists(npy_path) and not by_id:
        arr = np.load(npy_path, allow_pickle=False)
        if 'offset' in arr.dtype.names:
            npy_offsets = arr['offset']
            embeddings = arr['embedding']
            query_emb = embed([QUERY_PREFIX + query])[0]
            scores = embeddings @ query_emb
            top_indices = np.argsort(-scores)[:k]
            # Seek to byte offsets for just the top-k results
            with open(data_path, "rb") as f:
                for i in top_indices:
                    f.seek(int(npy_offsets[i]))
                    line = f.readline()
                    rec = json.loads(line)
                    rec["_score"] = float(scores[i])
                    print(json.dumps(rec), file=sys.stdout)
            return

    # Slow path: load all records (needed for --id mode or no index)
    records, offsets = read_jsonl(data_path)

    if not records:
        return

    texts = [build_text(rec) for rec in records]
    cache_path = (data_path + ".embeds") if not no_cache else None
    embeddings = embed(texts, cache_path=cache_path, keys=[str(o) for o in offsets])

    if by_id:
        # Find record by ID, use its embedding as query
        match_idx = None
        for i, rec in enumerate(records):
            if str(rec.get("id", "")) == query:
                match_idx = i
                break
        if match_idx is None:
            print(f"Record ID {query!r} not found in {data_path}", file=sys.stderr)
            sys.exit(1)
        query_emb = embeddings[match_idx]
    else:
        query_emb = embed([QUERY_PREFIX + query])[0]

    scores = embeddings @ query_emb

    top_indices = np.argsort(-scores)[:k]
    for i in top_indices:
        rec = dict(records[i])
        rec["_score"] = float(scores[i])
        print(json.dumps(rec), file=sys.stdout)


def discover_command(data_path, n_clusters=10, k=5, no_cache=False):
    """Cluster records by embedding similarity and print top-k examples per cluster as markdown."""
    from sklearn.cluster import KMeans

    records, offsets = read_jsonl(data_path)

    texts = [build_text(r) for r in records]
    if not texts:
        raise ValueError(f"No records found in {data_path}")

    cache_path = (data_path + ".embeds") if not no_cache else None
    embeddings = embed(texts, cache_path=cache_path, keys=[str(o) for o in offsets])

    n_clusters = min(n_clusters, len(texts))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(embeddings)

    # For each cluster, find the k items closest to the centroid
    centroids = km.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    for ci in range(n_clusters):
        member_indices = [i for i, lbl in enumerate(km.labels_) if lbl == ci]
        sims = embeddings[member_indices] @ centroids[ci]
        ranked = np.argsort(-sims)
        n = len(ranked)
        # Distribute k picks evenly from top (0) to middle (n//2)
        half = max(n // 2, 1)
        pick_positions = [int(half * i / max(k - 1, 1)) for i in range(k)] if k > 1 else [0]
        picks = [ranked[min(p, n - 1)] for p in pick_positions]

        print(f"## Cluster {ci + 1} ({len(member_indices)} records)\n")
        for rank, idx_in_members in enumerate(picks, 1):
            i = member_indices[idx_in_members]
            print(f"{rank}. {texts[i][:250].replace(chr(10), ' ')}")
        print()


def embed_server_command():
    """Run embed daemon: load model once, serve embeddings over unix socket."""
    import io
    import signal
    import socket
    import struct
    import sys

    # Write PID immediately — signals to other callers that a daemon is starting
    with open(EMBED_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Bind socket before model load — existence signals "starting up"
    if os.path.exists(EMBED_SOCKET):
        os.unlink(EMBED_SOCKET)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(EMBED_SOCKET)
    server.listen(5)
    server.settimeout(60)  # wake every 60s to check TTL

    # Now load model (slow — 3-5s)
    print("Loading model...", file=sys.stderr, flush=True)
    from sentence_transformers import SentenceTransformer
    _real_stderr = sys.stderr
    sys.stderr = io.StringIO()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    sys.stderr = _real_stderr
    print("Model loaded. Listening on", EMBED_SOCKET, file=sys.stderr, flush=True)

    start_time = import_time = __import__("time").time()

    def cleanup(sig=None, frame=None):
        server.close()
        for f in (EMBED_SOCKET, EMBED_PID_FILE):
            if os.path.exists(f):
                os.unlink(f)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while True:
        # Check TTL
        if __import__("time").time() - start_time > EMBED_TTL:
            print("TTL expired, shutting down.", file=sys.stderr)
            cleanup()

        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue

        try:
            # Read length-prefixed JSON
            len_data = _recv_exact(conn, 4)
            msg_len = struct.unpack("!I", len_data)[0]
            payload = _recv_exact(conn, msg_len)
            texts = json.loads(payload.decode())

            # Embed
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Send back: count + raw float32 bytes
            conn.sendall(struct.pack("!I", len(texts)))
            conn.sendall(embeddings.tobytes())
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        finally:
            conn.close()


def embed_stop_command():
    """Stop the running embed daemon."""
    import signal
    import sys
    if not os.path.exists(EMBED_PID_FILE):
        print("No daemon running (no PID file).", file=sys.stderr)
        return
    with open(EMBED_PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped embed daemon (pid {pid}).", file=sys.stderr)
    except ProcessLookupError:
        print(f"Daemon pid {pid} not running, cleaning up.", file=sys.stderr)
    for p in (EMBED_PID_FILE, EMBED_SOCKET):
        if os.path.exists(p):
            os.unlink(p)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="semlabel", description="Semantic labeling via concept vectors")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a concept vector via active learning")
    p_train.add_argument("concept", help="Concept JSON file (filename = concept name)")
    p_train.add_argument("data", help="JSONL data file")
    p_train.add_argument("--auto", action="store_true", help="Use LLM (claude -p) to label instead of manual input")
    p_train.add_argument("--trace", help="Write LLM prompt/response trace to this file")
    p_train.add_argument("--no-cache", action="store_true", help="Skip embedding cache")
    p_train.add_argument("--log-texts", help="Write embedding input texts to this file")

    # tag
    p_tag = sub.add_parser("tag", help="Score JSONL against concepts")
    p_tag.add_argument("concepts", help="Concept file or directory of concept files")
    p_tag.add_argument("data", nargs="?", help="JSONL data file (reads stdin if omitted)")
    p_tag.add_argument("--no-cache", action="store_true", help="Skip embedding cache")
    p_tag.add_argument("--tag-name", default="tags", help="Output field name (default: tags)")
    p_tag.add_argument("--raw", action="store_true", help="Output raw cosine scores instead of conformal confidence")

    # show
    p_show = sub.add_parser("show", help="Show compact summary of concepts")
    p_show.add_argument("path", help="Concept file or directory of concept files")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Compute conformal calibration for trained concepts")
    p_cal.add_argument("path", help="Concept file or directory of concept files")

    # add
    p_add = sub.add_parser("add", help="Add a record to a concept's positives or negatives")
    p_add.add_argument("concept", help="Concept JSON file")
    p_add.add_argument("data", help="JSONL data file (to look up record by ID)")
    p_add.add_argument("entries", nargs="*", help="Pairs of: record_id +/- (e.g. abc123 + def456 -). Omit to just recompute weight.")
    p_add.add_argument("--no-cache", action="store_true", help="Skip embedding cache")
    p_add.add_argument("--dry-run", action="store_true", help="Show reclassification diff without saving")

    # search
    p_search = sub.add_parser("search", help="Semantic search over JSONL records")
    p_search.add_argument("query", help="Search query string (or record ID with --id)")
    p_search.add_argument("data", help="JSONL data file")
    p_search.add_argument("-k", type=int, default=10, help="Number of results (default: 10)")
    p_search.add_argument("--id", action="store_true", help="Treat query as a record ID")
    p_search.add_argument("--no-cache", action="store_true", help="Skip embedding cache")

    # embed-server
    sub.add_parser("embed-server", help="Run embed daemon (keeps model warm for fast queries)")

    # embed-stop
    sub.add_parser("embed-stop", help="Stop the running embed daemon")

    # discover
    p_disc = sub.add_parser("discover", help="Cluster records and show top examples per cluster")
    p_disc.add_argument("data", help="JSONL data file")
    p_disc.add_argument("-n", "--clusters", type=int, default=10, help="Number of clusters (default: 10)")
    p_disc.add_argument("-k", type=int, default=5, help="Examples per cluster (default: 5)")
    p_disc.add_argument("--no-cache", action="store_true", help="Skip embedding cache")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args.data, args.concept, auto=args.auto, trace_path=args.trace, no_cache=args.no_cache, log_texts_path=args.log_texts)
    elif args.command == "tag":
        tag_command(args.concepts, data_path=args.data, no_cache=args.no_cache, tag_name=args.tag_name, raw=args.raw)
    elif args.command == "show":
        show_command(args.path)
    elif args.command == "calibrate":
        calibrate_command(args.path)
    elif args.command == "add":
        # Parse id/label pairs from flat list: id1 + id2 - id3 +
        entries = args.entries or []
        if len(entries) % 2 != 0:
            parser.error("entries must be pairs of: record_id +/-")
        pairs = []
        for i in range(0, len(entries), 2):
            rid, lbl = entries[i], entries[i + 1]
            if lbl not in ("+", "-"):
                parser.error(f"expected +/- after {rid}, got {lbl}")
            pairs.append((rid, lbl == "+"))
        add_command(args.concept, args.data, pairs, no_cache=args.no_cache, dry_run=args.dry_run)
    elif args.command == "search":
        search_command(args.query, args.data, k=args.k, no_cache=args.no_cache, by_id=args.id)
    elif args.command == "embed-server":
        embed_server_command()
    elif args.command == "embed-stop":
        embed_stop_command()
    elif args.command == "discover":
        discover_command(args.data, n_clusters=args.clusters, k=args.k, no_cache=args.no_cache)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

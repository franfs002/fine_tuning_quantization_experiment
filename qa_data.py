"""Extractive QA data: SQuAD-format loading, features, post-processing and metrics.

Every question keeps the id of its passage (`pid`), so the bootstrap can resample
passages instead of questions (questions on the same passage are not independent).
"""
import collections
import hashlib
import json
import re
import string

import numpy as np
import torch


def load_squad_json(path):
    """Read a SQuAD v1-style file ({"data": [{"paragraphs": [{"context", "qas"}]}]})."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    articles = raw["data"] if isinstance(raw, dict) else raw
    examples = []
    for art in articles:
        for par in art["paragraphs"]:
            ctx = par["context"]
            pid = hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12]
            for qa in par["qas"]:
                answers = [{"text": a["text"], "answer_start": int(a["answer_start"])}
                           for a in qa.get("answers", [])]
                examples.append({"id": str(qa["id"]), "pid": pid, "context": ctx,
                                 "question": qa["question"].strip(), "answers": answers})
    return examples


def _encode(examples, tok, max_len, stride):
    return tok([e["question"] for e in examples], [e["context"] for e in examples],
               truncation="only_second", max_length=max_len, stride=stride,
               return_overflowing_tokens=True, return_offsets_mapping=True,
               padding="max_length")


def _tensors(enc):
    keys = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in enc else [])
    return {k: torch.tensor(enc[k], dtype=torch.long) for k in keys}


def prepare_train_features(examples, tok, max_len=384, stride=128):
    """Windows over each context, labelled with the answer's token span (CLS if cut off)."""
    enc = _encode(examples, tok, max_len, stride)
    sample_map = enc["overflow_to_sample_mapping"]
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        ex = examples[sample_map[i]]
        if not ex["answers"] or 1 not in seq_ids:
            starts.append(0); ends.append(0); continue
        a = ex["answers"][0]
        s_char, e_char = a["answer_start"], a["answer_start"] + len(a["text"])
        ts = 0
        while seq_ids[ts] != 1:
            ts += 1
        te = len(offsets) - 1
        while seq_ids[te] != 1:
            te -= 1
        if not (offsets[ts][0] <= s_char and offsets[te][1] >= e_char):
            starts.append(0); ends.append(0); continue
        while ts < len(offsets) and seq_ids[ts] == 1 and offsets[ts][0] <= s_char:
            ts += 1
        starts.append(ts - 1)
        while seq_ids[te] == 1 and offsets[te][1] >= e_char:
            te -= 1
        ends.append(te + 1)
    t = _tensors(enc)
    t["start_positions"] = torch.tensor(starts, dtype=torch.long)
    t["end_positions"] = torch.tensor(ends, dtype=torch.long)
    return t


def prepare_eval_features(examples, tok, max_len=384, stride=128):
    """Windows plus what post-processing needs: the example of each window and its context offsets."""
    enc = _encode(examples, tok, max_len, stride)
    offsets = []
    for i, off in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offsets.append([o if seq_ids[k] == 1 else None for k, o in enumerate(off)])
    return {"tensors": _tensors(enc), "example_index": list(enc["overflow_to_sample_mapping"]),
            "offsets": offsets}


def postprocess(examples, feats, start_logits, end_logits, n_best=20, max_answer_len=30):
    """Best-scoring valid span per question across all of its windows."""
    by_example = collections.defaultdict(list)
    for fi, ei in enumerate(feats["example_index"]):
        by_example[ei].append(fi)
    preds = []
    for ei, ex in enumerate(examples):
        best_text, best_score = "", -np.inf
        for fi in by_example[ei]:
            sl, el, off = start_logits[fi], end_logits[fi], feats["offsets"][fi]
            for s in np.argsort(sl)[::-1][:n_best]:
                if off[s] is None:
                    continue
                for e in np.argsort(el)[::-1][:n_best]:
                    if off[e] is None or e < s or e - s + 1 > max_answer_len:
                        continue
                    score = sl[s] + el[e]
                    if score > best_score:
                        best_score, best_text = score, ex["context"][off[s][0]:off[e][1]]
        preds.append(best_text)
    return preds


_PUNCT = set(string.punctuation) | set("¿¡«»“”‘’…–—")
_ARTICLES = re.compile(r"\b(el|la|los|las|un|una|unos|unas)\b")


def normalize(text):
    """SQuAD normalization adapted to Spanish articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def f1_em(pred, golds):
    if not golds:
        return (100.0, 100.0) if not pred else (0.0, 0.0)
    best_f1 = best_em = 0.0
    p_tok = normalize(pred).split()
    for g in golds:
        g_tok = normalize(g).split()
        em = float(p_tok == g_tok)
        common = collections.Counter(p_tok) & collections.Counter(g_tok)
        same = sum(common.values())
        if same == 0:
            f1 = 0.0
        else:
            prec, rec = same / len(p_tok), same / len(g_tok)
            f1 = 2 * prec * rec / (prec + rec)
        best_f1, best_em = max(best_f1, f1), max(best_em, em)
    return 100.0 * best_f1, 100.0 * best_em


def score_examples(examples, preds):
    """One row per question: id, passage, F1 and EM in points (0-100)."""
    rows = []
    for ex, p in zip(examples, preds):
        f1, em = f1_em(p, [a["text"] for a in ex["answers"]])
        rows.append({"qid": ex["id"], "pid": ex["pid"], "f1": f1, "em": em})
    return rows

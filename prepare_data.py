"""Get the QA data into three SQuAD-format files: train.json, dev.json (selection), test.json.

Accepted inputs: SQuAD JSON ({"data": [...]}), a JSON list or JSON-lines file of Hugging Face-style
records (id, title, context, question, answers{text: [...], answer_start: [...]}), or a folder saved
with datasets.save_to_disk.
"""
import argparse
import collections
import json
import os
import random
import shutil


def _records_to_articles(records):
    arts = collections.OrderedDict()
    for r in records:
        ans = r.get("answers", {})
        if isinstance(ans, dict):
            ans = [{"text": t, "answer_start": s} for t, s in zip(ans.get("text", []), ans.get("answer_start", []))]
        title = r.get("title", "")
        ctxs = arts.setdefault(title, collections.OrderedDict())
        ctxs.setdefault(r["context"], []).append({"id": str(r["id"]), "question": r["question"], "answers": ans})
    return [{"title": t, "paragraphs": [{"context": c, "qas": q} for c, q in ctxs.items()]} for t, ctxs in arts.items()]


def read_any(path):
    if os.path.isdir(path):
        from datasets import load_from_disk
        ds = load_from_disk(path)
        return _records_to_articles(ds)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return _records_to_articles(json.loads(l) for l in text.splitlines() if l.strip())
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, list) and raw and "paragraphs" in raw[0]:
        return raw
    return _records_to_articles(raw)


def repair(articles):
    """Check every answer span against its context; fix the offset by search, else drop."""
    kept = fixed = dropped_a = dropped_q = 0
    for art in articles:
        for par in art["paragraphs"]:
            ctx, good_qas = par["context"], []
            for qa in par["qas"]:
                good = []
                for a in qa["answers"]:
                    t, s = a["text"], int(a["answer_start"])
                    if ctx[s:s + len(t)] == t:
                        good.append({"text": t, "answer_start": s})
                    elif t and ctx.find(t) >= 0:
                        good.append({"text": t, "answer_start": ctx.find(t)}); fixed += 1
                    else:
                        dropped_a += 1
                if good:
                    qa["answers"] = good; good_qas.append(qa); kept += 1
                else:
                    dropped_q += 1
            par["qas"] = good_qas
    print(f"answers repaired: {fixed} | answers dropped: {dropped_a} | questions dropped: {dropped_q} | kept: {kept}")
    return articles


def _passages(articles):
    return [(a.get("title", ""), p) for a in articles for p in a["paragraphs"] if p["qas"]]


def _take(units, n_questions):
    out, n = [], 0
    for u in units:
        if n >= n_questions:
            break
        out.append(u); n += len(u[1]["qas"])
    return out


def _write(units, path):
    data = [{"title": t, "paragraphs": [p]} for t, p in units]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": "prepared", "data": data}, f, ensure_ascii=False)


def stats(folder):
    for split in ("train", "dev", "test"):
        with open(os.path.join(folder, f"{split}.json"), encoding="utf-8") as f:
            data = json.load(f)["data"]
        pars = [p for a in data for p in a["paragraphs"]]
        q = sum(len(p["qas"]) for p in pars)
        titles = len({a.get("title", "") for a in data})
        print(f"{split:5s}: {q:6d} questions over {len(pars):5d} passages from {titles:4d} articles")


def squad_es(train, dev, out, n_train, n_eval, seed):
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    tr = _passages(repair(read_any(train)))
    rng.shuffle(tr)
    _write(_take(tr, n_train), os.path.join(out, "train.json"))
    dv = repair(read_any(dev))
    titles = sorted({a.get("title", "") for a in dv})
    rng.shuffle(titles)
    half = set(titles[: len(titles) // 2])
    sel = [u for u in _passages(dv) if u[0] in half]
    tst = [u for u in _passages(dv) if u[0] not in half]
    rng.shuffle(sel); rng.shuffle(tst)
    _write(_take(sel, n_eval), os.path.join(out, "dev.json"))
    _write(_take(tst, n_eval), os.path.join(out, "test.json"))
    stats(out)


def sqac(out):
    from huggingface_hub import hf_hub_download
    os.makedirs(out, exist_ok=True)
    for split in ("train", "dev", "test"):
        p = hf_hub_download("PlanTL-GOB-ES/SQAC", f"{split}.json", repo_type="dataset")
        shutil.copy(p, os.path.join(out, f"{split}.json"))
    stats(out)


def split(path, out, seed, frac=(0.8, 0.1, 0.1)):
    units = _passages(repair(read_any(path)))
    random.Random(seed).shuffle(units)
    n = len(units)
    cut1, cut2 = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    os.makedirs(out, exist_ok=True)
    for name, part in (("train", units[:cut1]), ("dev", units[cut1:cut2]), ("test", units[cut2:])):
        _write(part, os.path.join(out, f"{name}.json"))
    stats(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("squad_es")
    s.add_argument("--train", required=True); s.add_argument("--dev", required=True)
    s.add_argument("--out", required=True); s.add_argument("--n-train", type=int, default=15000)
    s.add_argument("--n-eval", type=int, default=2000); s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("sqac"); a.add_argument("--out", required=True)
    b = sub.add_parser("split"); b.add_argument("--input", required=True); b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("stats"); c.add_argument("--dir", required=True)
    args = ap.parse_args()
    if args.cmd == "squad_es":
        squad_es(args.train, args.dev, args.out, args.n_train, args.n_eval, args.seed)
    elif args.cmd == "sqac":
        sqac(args.out)
    elif args.cmd == "split":
        split(args.input, args.out, args.seed)
    else:
        stats(args.dir)

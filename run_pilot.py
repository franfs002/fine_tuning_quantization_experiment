"""
fine-tunes the pretrained model on the training split;
scores FP32 on the selection (dev) and test splits;
for each quantizer (RTN, GPTQ) and bit-width, quantizes the FP32 weights and scores again;
writes one row per question to scores/<run>.csv.gz.

"""
import argparse
import json
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

import qa_data
import quantize


class DictDataset(Dataset):
    def __init__(self, tensors):
        self.t = tensors
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(tag, lr, wd, seed):
    return f"{tag}__lr{lr:g}__wd{wd:g}__s{seed}"


def load_features(cfg, tag, tok, cache_dir):
    """Tokenize the splits once per model (tokenizers differ between models) and cache them."""
    path = os.path.join(cache_dir, f"{tag}_features.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = cfg["data"]
    ex = {s: qa_data.load_squad_json(d[s]) for s in ("train", "dev", "test")}
    if cfg.get("limit_train"):
        ex["train"] = ex["train"][: cfg["limit_train"]]
    ml, st = cfg["max_len"], cfg["stride"]
    train = qa_data.prepare_train_features(ex["train"], tok, ml, st)
    g = torch.Generator().manual_seed(cfg["calib_seed"])
    idx = torch.randperm(train["input_ids"].shape[0], generator=g)[: cfg["calib_n"]]
    calib = {k: v[idx] for k, v in train.items() if k not in ("start_positions", "end_positions")}
    feats = {"examples": ex, "train": train, "calib": calib,
             "dev": qa_data.prepare_eval_features(ex["dev"], tok, ml, st),
             "test": qa_data.prepare_eval_features(ex["test"], tok, ml, st)}
    torch.save(feats, path)
    return feats


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train(model, feats, lr, wd, seed, cfg, device):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(DictDataset(feats), batch_size=cfg["batch_size"], shuffle=True, generator=g)
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    total = int(np.ceil(len(dl) * cfg["epochs"]))  # epochs may be fractional (e.g. 3.5)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.get("warmup", 0.1) * total), total)
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    model.train()
    last = []
    step = 0
    while step < total:
        for batch in dl:
            if step >= total:
                break
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                loss = model(**batch).loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            last.append(loss.item())
    return float(np.mean(last[-50:]))


@torch.no_grad()
def evaluate(model, examples, feats, device, bs):
    """FP32 inference (no autocast, no TF32) so only the weights differ between precisions."""
    model.eval()
    starts, ends = [], []
    t = feats["tensors"]
    n = t["input_ids"].shape[0]
    for i in range(0, n, bs):
        out = model(**{k: v[i:i + bs].to(device) for k, v in t.items()})
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
    preds = qa_data.postprocess(examples, feats, np.concatenate(starts), np.concatenate(ends))
    return qa_data.score_examples(examples, preds)


def one_run(cfg, mdl, lr, wd, seed, feats, device, out_dir):
    name = run_name(mdl["tag"], lr, wd, seed)
    t0 = time.time()
    set_seed(seed)
    # e.g. {"hidden_dropout_prob": 0.1}: BERTIN ships with dropout 0.0, the BSC models with 0.1
    model = AutoModelForQuestionAnswering.from_pretrained(
        mdl["path"], **mdl.get("config_overrides", {})).to(device)
    loss = train(model, feats["train"], lr, wd, seed, cfg, device)
    t_train = time.time() - t0
    fp32 = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    rows = []

    def score(quant, bits):
        for split in ("dev", "test"):
            for r in evaluate(model, feats["examples"][split], feats[split], device, cfg["eval_batch_size"]):
                rows.append({"model": mdl["tag"], "lr": lr, "wd": wd, "seed": seed,
                             "quant": quant, "bits": bits, "split": split, **r})

    score("none", "32")
    for q in cfg["quantizers"]:
        for b in cfg["bits"]:
            model.load_state_dict(fp32)
            if q == "rtn":
                quantize.quantize_rtn(model, b)
            elif q == "gptq":
                quantize.quantize_gptq(model, b, feats["calib"], device)
            else:
                raise ValueError(q)
            score(q, b)
    if cfg.get("save_models"):  # insurance: lets you re-quantize later without retraining
        model.load_state_dict(fp32)
        if cfg.get("save_dtype", "float16") == "float16":
            model = model.half()  # ~250 MB per model instead of ~500 MB
        model.save_pretrained(os.path.join(out_dir, "models", name))
    df = pd.DataFrame(rows)
    tmp = os.path.join(out_dir, "scores", name + ".tmp.csv.gz")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(out_dir, "scores", name + ".csv.gz"))
    meta = {"run": name, "train_loss_last50": loss, "train_seconds": round(t_train, 1),
            "total_seconds": round(time.time() - t0, 1),
            "fp32_dev_f1": float(df.query("quant == 'none' and split == 'dev'").f1.mean()),
            "fp32_test_f1": float(df.query("quant == 'none' and split == 'test'").f1.mean()),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    with open(os.path.join(out_dir, "runs", name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3: run only these seeds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    out_dir = cfg["out_dir"]
    for sub in ("scores", "runs", "cache"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with open(os.path.join(out_dir, "grid_used.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    jobs = [(m, lr, wd, s) for s in seeds for m in cfg["models"] for lr in cfg["lrs"] for wd in cfg["wds"]]
    i, n = map(int, args.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(out_dir, "scores", run_name(j[0]["tag"], j[1], j[2], j[3]) + ".csv.gz"))]
    print(f"{len(jobs)} jobs in this shard, {len(todo)} still to run")
    if args.dry_run:
        for j in todo:
            print(run_name(j[0]["tag"], j[1], j[2], j[3]))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    feats_by_tag = {}
    for m, lr, wd, s in todo:
        if m["tag"] not in feats_by_tag:
            tok = AutoTokenizer.from_pretrained(m["path"])
            feats_by_tag[m["tag"]] = load_features(cfg, m["tag"], tok, os.path.join(out_dir, "cache"))
        meta = one_run(cfg, m, lr, wd, s, feats_by_tag[m["tag"]], device, out_dir)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()

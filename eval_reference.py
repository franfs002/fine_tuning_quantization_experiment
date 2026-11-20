"""Quantize a model that is ALREADY fine-tuned and score it, without training.

It answers a direct question for the thesis: how much F1 does the model Zavala actually built lose
at INT8 and INT4? The rows have the same format as the grid's, under scores_reference/.

    python eval_reference.py --config grid_squad_es.json --path 
"""
import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import quantize
import run_pilot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    out = os.path.join(cfg["out_dir"], "scores_reference")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(cfg["out_dir"], "cache"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.path)
    feats = run_pilot.load_features(cfg, args.tag, tok, os.path.join(cfg["out_dir"], "cache"))
    model = AutoModelForQuestionAnswering.from_pretrained(args.path).to(device)
    fp32 = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    rows = []
    for quant, bits in [("none", "32")] + [(q, b) for q in cfg["quantizers"] for b in cfg["bits"]]:
        model.load_state_dict(fp32)
        if quant == "rtn":
            quantize.quantize_rtn(model, bits)
        elif quant == "gptq":
            quantize.quantize_gptq(model, bits, feats["calib"], device)
        for split in ("dev", "test"):
            for r in run_pilot.evaluate(model, feats["examples"][split], feats[split], device, cfg["eval_batch_size"]):
                rows.append({"model": args.tag, "quant": quant, "bits": bits, "split": split, **r})
        print(quant, bits, round(pd.DataFrame(rows).query("quant == @quant and bits == @bits and split == 'test'").f1.mean(), 2))
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{args.tag}.csv.gz"), index=False)


if __name__ == "__main__":
    main()

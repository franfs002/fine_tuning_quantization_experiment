"""Find which public pretrained model a fine-tuned checkpoint started from.

Fine-tuning moves the weights only a little, so the true base is the candidate whose weights are
closest (relative difference of a few percent); any other pretrained model differs by ~100% or more.

    python identify_base.py --finetuned \
        --candidates BSC-LT/roberta-base-biomedical-clinical-es BSC-LT/roberta-base-biomedical-es \
                     bertin-project/bertin-roberta-base-spanish
"""
import argparse
import glob
import json
import os

import torch


def _state_dict(folder):
    st = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if st:
        from safetensors.torch import load_file
        sd = {}
        for f in st:
            sd.update(load_file(f))
    else:
        sd = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    return {k.split(".", 1)[1] if k.startswith(("roberta.", "bert.")) else k: v.float() for k, v in sd.items()}


def describe(folder):
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    sd = _state_dict(folder)
    print(f"{folder}\n  architecture: {cfg.get('architectures')} | vocab {cfg.get('vocab_size')} | "
          f"layers {cfg.get('num_hidden_layers')} | hidden {cfg.get('hidden_size')}\n"
          f"  QA head present (already fine-tuned for QA): {any(k.startswith('qa_outputs') for k in sd)}")
    return sd


KEYS = ["embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.11.output.dense.weight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    ft = describe(args.finetuned)
    print("\nRelative difference to each candidate (smaller = closer; a true base is usually < 0.1):")
    for c in args.candidates:
        try:
            path = c if os.path.isdir(c) else snapshot_download(c, allow_patterns=["*.safetensors", "pytorch_model.bin", "config.json"])
            base = _state_dict(path)
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: could not load ({e.__class__.__name__})")
            continue
        diffs = []
        for k in KEYS:
            if k in ft and k in base and ft[k].shape == base[k].shape:
                diffs.append(((ft[k] - base[k]).norm() / base[k].norm()).item())
            else:
                diffs.append(float("nan"))
        print(f"  {c}: " + "  ".join(f"{k.split('.')[0]}..{k.split('.')[-2]}={d:.3f}" for k, d in zip(KEYS, diffs)))


if __name__ == "__main__":
    main()

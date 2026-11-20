"""Weight-only post-training quantization of the encoder's linear layers.

Two quantizers share one integer grid (symmetric, per output channel, or per group of
input columns), so the only difference between them is how weights are rounded

Weights are quantized and de-quantized ("fake quantization"), so evaluation runs in FP32 and
any accuracy change comes from the weights alone. Embeddings and the QA head stay in FP32.

Bit-width specs: "8", "6", "4" (per output channel) or "4g128" (groups of 128 input columns).
"""
import torch
import torch.nn as nn


def parse_bits(spec):
    spec = str(spec)
    if "g" in spec:
        b, g = spec.split("g")
        return int(b), int(g)
    return int(spec), 0


def _scale(w, qmax):
    """Per-row symmetric scale for the given weight slice (rows = output channels)."""
    return (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)


def _round(w, scale, qmax):
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def rtn(W, bits, group=0):
    W = W.float()
    qmax = 2 ** (bits - 1) - 1
    if group <= 0:
        return _round(W, _scale(W, qmax), qmax)
    Q = torch.empty_like(W)
    for i in range(0, W.shape[1], group):
        w = W[:, i:i + group]
        Q[:, i:i + group] = _round(w, _scale(w, qmax), qmax)
    return Q


def gptq(W, H, bits, group=0, blocksize=128, percdamp=0.01):
    """GPTQ for one weight matrix W [out, in] given H [in, in]; returns the quantized W."""
    W = W.clone().float()
    H = H.clone().float()
    n = W.shape[1]
    qmax = 2 ** (bits - 1) - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    scale = _scale(W, qmax) if group <= 0 else None
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if group > 0 and col % group == 0:
                # scale for the new group, from the weights as already updated by GPTQ
                cur = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :group]
                scale = _scale(cur, qmax)
            w = W1[:, i]
            q = _round(w.unsqueeze(1), scale, qmax).squeeze(1)
            Q1[:, i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def encoder_layers(model):
    return model.base_model.encoder.layer


def _linears(layer):
    return {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def quantize_rtn(model, spec):
    bits, group = parse_bits(spec)
    for layer in encoder_layers(model):
        for m in _linears(layer).values():
            m.weight.data = rtn(m.weight.data, bits, group).to(m.weight.dtype)


class _Stop(Exception):
    pass


@torch.no_grad()
def quantize_gptq(model, spec, calib, device, batch_size=16):
    """Sequential GPTQ: layer l is calibrated on inputs produced by the already-quantized layers < l.

    `calib` is a dict of tensors (input_ids, attention_mask[, token_type_ids]) of calibration windows.
    Padding tokens are excluded from H.
    """
    bits, group = parse_bits(spec)
    model.eval()
    n = calib["input_ids"].shape[0]
    for layer in encoder_layers(model):
        lins = _linears(layer)
        H = {k: torch.zeros(m.in_features, m.in_features, device=device) for k, m in lins.items()}
        count = {k: 0 for k in lins}
        state = {}

        def make_hook(key):
            def hook(mod, inp, out):
                x = inp[0].float()
                mask = state["mask"].bool()
                x = x[mask] if x.dim() == 3 else x
                x = x.reshape(-1, x.shape[-1])
                m = x.shape[0]
                H[key].mul_(count[key] / (count[key] + m))
                count[key] += m
                x = x * (2.0 / count[key]) ** 0.5
                H[key].add_(x.t() @ x)
            return hook

        def stop(*_):
            raise _Stop()  # layers above this one are not needed yet

        handles = [m.register_forward_hook(make_hook(k)) for k, m in lins.items()]
        handles.append(layer.register_forward_hook(stop))
        for i in range(0, n, batch_size):
            batch = {k: v[i:i + batch_size].to(device) for k, v in calib.items()}
            state["mask"] = batch["attention_mask"]
            try:
                model(**batch)
            except _Stop:
                pass
        for h in handles:
            h.remove()
        for k, m in lins.items():
            m.weight.data = gptq(m.weight.data, H[k], bits, group).to(m.weight.dtype)

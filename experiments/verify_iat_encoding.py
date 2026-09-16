"""U2 pre-flight: verify the `--interval_encoding` switch through the real loader path.

Asserts that (a) `sigmoid_log` reproduces the original upstream values bit-for-bit,
(b) `log10_decade` lands in the integer domain FixedCosineEmbed expects, and
(c) the resulting token embeddings are actually distinguishable -- the old encoding's
failure mode was 20 near-identical embeddings, so we measure that directly.

Run:  python experiments/verify_iat_encoding.py
"""
import json
import math
import os
import sys

import torch

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "NetMambaPlus", "src"))
sys.path.insert(0, os.path.join(_ROOT, "NetMambaPlus", "src", "models"))

from util.loader_data import ByteSizeIntervalDataset, encode_intervals  # noqa: E402
from models.common import FixedCosineEmbed  # noqa: E402

DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
SEQ_LEN = 20
EMBED_DIM = 256

checks = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def load_dataset(encoding):
    return ByteSizeIntervalDataset(
        os.path.join(DATA, "data-train.json"),
        num_packet=5, num_packet_byte=320, seq_len=SEQ_LEN,
        size_key="signed_sizes", interval_encoding=encoding,
    )


def raw_intervals():
    with open(os.path.join(DATA, "data-train.json")) as f:
        rows = json.load(f)
    out = []
    for it in rows:
        arr = [float(x) for x in it["intervals"].split(" ")][:SEQ_LEN]
        arr += [float("inf")] * (SEQ_LEN - len(arr))
        out.append(arr)
    return out


def main():
    print("=" * 74)
    print("1. encode_intervals() unit behaviour")
    print("=" * 74)
    # (a) sigmoid_log must be bit-identical to the original inline lambda
    probe = [0.0, 1e-7, 4e-6, 1.12e-4, 1.7e-3, 28.0, float("inf")]
    old_ref = [1 / (1 + 1 / (1 + x)) for x in probe]
    old_new = encode_intervals(probe, "sigmoid_log")
    check("sigmoid_log is bit-identical to the original formula",
          all(a == b for a, b in zip(old_ref, old_new)))

    new = encode_intervals(probe, "log10_decade")
    expect = {1e-6: 100.0, 1e-5: 200.0, 1e-4: 300.0, 1e-3: 400.0, 1.0: 700.0}
    got = encode_intervals(list(expect), "log10_decade")
    check("log10_decade decade anchors (1us->100 ... 1s->700)",
          all(abs(g - e) < 1e-9 for g, e in zip(got, expect.values())),
          f"{[round(g, 2) for g in got]}")
    check("log10_decade maps pad(inf) to the 1000 sentinel", new[-1] == 1000.0)
    check("log10_decade maps exact-zero IAT to the 0 floor", new[0] == 0.0)
    check("log10_decade is monotonic on finite input",
          all(new[i] <= new[i + 1] for i in range(len(new) - 2)),
          f"{[round(v, 2) for v in new[:-1]]}")

    print()
    print("=" * 74)
    print("2. through the real loader (data-train.json, signed_sizes)")
    print("=" * 74)
    raws = raw_intervals()
    tensors = {}
    for enc in ("sigmoid_log", "log10_decade"):
        ds = load_dataset(enc)
        _, sizes, ivs = ds[0][0], ds[0][1], ds[0][2]
        allv = torch.stack([ds[i][2] for i in range(len(ds))])
        tensors[enc] = allv
        print(f"  {enc:14s} n={len(ds)}  min={allv.min():.6f}  max={allv.max():.6f}  "
              f"std={allv.std():.6f}")
        # loader output must equal a direct re-encode of the raw field
        ref = torch.tensor([encode_intervals(r, enc) for r in raws], dtype=allv.dtype)
        check(f"{enc}: loader output == direct encode_intervals()",
              torch.allclose(allv, ref, atol=0, rtol=0))

    check("sizes still shifted into [0, 2*MTU] (unchanged by this patch)",
          bool(sizes.min() >= 0 and sizes.max() <= 3000),
          f"[{sizes.min().item():.0f}, {sizes.max().item():.0f}]")

    print()
    print("=" * 74)
    print("3. post-embedding distinguishability (the actual failure mode)")
    print("=" * 74)
    embed = FixedCosineEmbed(EMBED_DIM)
    with torch.no_grad():
        for enc in ("sigmoid_log", "log10_decade"):
            v = tensors[enc]
            # pad sentinel differs per encoding; compare only real (non-pad) tokens
            pad = 1.0 if enc == "sigmoid_log" else 1000.0
            emb = embed(v)  # [N, L, D]
            # mean pairwise cosine similarity between the 20 tokens of a flow
            sims, spreads = [], []
            for i in range(min(64, v.shape[0])):
                mask = v[i] != pad
                if mask.sum() < 2:
                    continue
                e = torch.nn.functional.normalize(emb[i][mask], dim=-1)
                s = e @ e.T
                off = s[~torch.eye(s.shape[0], dtype=torch.bool)]
                sims.append(off.mean().item())
                spreads.append((emb[i][mask].max(0).values - emb[i][mask].min(0).values).mean().item())
            ms = sum(sims) / len(sims)
            sp = sum(spreads) / len(spreads)
            print(f"  {enc:14s} mean intra-flow token cosine-sim = {ms:.6f}   "
                  f"mean per-dim spread = {sp:.6f}")
            if enc == "sigmoid_log":
                check("sigmoid_log tokens are near-collapsed (cos-sim > 0.99)", ms > 0.99,
                      f"cos-sim={ms:.6f} -- this is the bug being fixed")
            else:
                check("log10_decade tokens are separated (cos-sim < 0.9)", ms < 0.9,
                      f"cos-sim={ms:.6f}")

    print()
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print("=" * 74)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""[2026-10-16] 论文 III-A "Their handshakes match Direct's (1250 B Initial, 8-byte client
connection ID, no Retry)" 的出处: 11 条改标流与两个采集轮次的握手字段对照。

只读已提交的特征 (data/processed/quic_vs_masque_bi_feat/data.json), 字段解析与
experiments/analyze_v3_single_feature.py 的 v1 部分相同 (2026-10-12 的原始出处
outputs/v3_kfold/single_feature_diag.json["v1"], 数字一致):
  first_frame      首包帧长 (UDP 载荷 + 42 B; 1292 = Chromium 的 1250 B Initial, 1242 = WARP 的 1200 B)
  client_dcid_len  首包 (客户端 Initial, 长包头) 的 DCID 长度 (QUIC 载荷第 5 字节)
  retry_first5     前 5 包里是否有下行的长包头 Retry (type bits == 3)
特征 "data" 每包 = 80 B 头区 + 240 B 载荷, QUIC 首字节 = 载荷第 0 字节。

    python experiments/analyze_handshake_fields.py   # -> outputs/handshake_fields/summary.json
"""
import json
import os
from collections import Counter

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat", "data.json")
NOISE = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
OUT = os.path.join(_ROOT, "outputs", "handshake_fields", "summary.json")


def fields(it):
    s = [int(x) for x in it["signed_sizes"].split()][:5]
    dcid, retry = -1, 0
    for i, pkt in enumerate(it["data"][:5]):
        pl = [int(x) for x in pkt.split()][80:]
        if not pl or not (pl[0] & 0x80):
            continue
        if i == 0:
            dcid = pl[5]
        if ((pl[0] >> 4) & 0x3) == 3 and s[i] < 0:
            retry = 1
    return {"first_frame": s[0], "client_dcid_len": dcid, "retry_first5": retry}


def tally(rows):
    F = [fields(r) for r in rows]
    return {"n": len(rows), **{k: Counter(f[k] for f in F).most_common() for k in F[0]}}


def main():
    with open(DATA, encoding="utf-8") as f:
        items = json.load(f)
    with open(NOISE, encoding="utf-8") as f:
        noise = set(json.load(f)["label_noise_flows"])
    base = lambda it: os.path.basename(it["pcap_file"].replace("\\", "/"))  # noqa: E731
    groups = {
        "WARP_off (capture run)": [it for it in items if it["label"] == 0],
        "WARP_on, peer in 162.159 (MASQUE)": [it for it in items if it["label"] == 1 and base(it) not in noise],
        "WARP_on, relabelled Direct (the 11)": [it for it in items if base(it) in noise],
    }
    res = {k: tally(v) for k, v in groups.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""[2026-10-15 ICOIN r3 校对 S11] 改标判据的穷举核对: 按采集轮次 (capture-run label) 统计对端是否在 WARP ingress 前缀 162.159。

论文 III-A 原写 "11 of the 270 WARP-on flows fall in the disconnect window" —— 断线窗口是推断出的机制。
可观测的事实是: WARP-on 的 270 条里有多少条对端在 162.159, WARP-off 的 234 条里有没有。
对端 = pcap 文件名里两个端点中不在 192.168.0.0/16 的那个 (文件名格式见 split_quic_vs_masque_bi.py)。
同时与 outputs/e3_kfold/error_analysis_e3.json 的 label_noise_flows (11 条) 交叉核对。

    python -X utf8 experiments/check_peer_prefix.py
输出 outputs/peer_prefix_check.json
"""

import json
import os
import re
from collections import Counter

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat", "data.json")
NOISE = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
OUT = os.path.join(_ROOT, "outputs", "peer_prefix_check.json")
PREFIX = "162.159."

# ..._<date>_<time>_<a.b.c.d>_<port>_<a.b.c.d>_<port>[.pcap]
PAT = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)(?:\.pcap)?$")


def peer(pcap_file):
    m = PAT.search(os.path.basename(pcap_file.replace("\\", "/")))
    assert m, pcap_file
    g = m.groups()
    a, b = ".".join(g[0:4]), ".".join(g[5:9])
    ends = [x for x in (a, b) if not x.startswith("192.168.")]
    assert len(ends) == 1, (pcap_file, a, b)
    return ends[0]


def _base(it):
    return os.path.basename(it["pcap_file"].replace("\\", "/"))


def main():
    with open(DATA, encoding="utf-8") as f:
        items = json.load(f)
    with open(NOISE, encoding="utf-8") as f:
        noise = json.load(f)["label_noise_flows"]
    noise_keys = {os.path.basename(str(x.get("pcap_file", x) if isinstance(x, dict) else x).replace("\\", "/"))
                  for x in noise}

    res = {}
    outside_on = []
    for run, lab in (("WARP_off", 0), ("WARP_on", 1)):
        sub = [it for it in items if it["label"] == lab]
        in_pfx = [it for it in sub if peer(it["pcap_file"]).startswith(PREFIX)]
        res[run] = {"n": len(sub), "peer_in_162.159": len(in_pfx), "peer_outside": len(sub) - len(in_pfx),
                    "distinct_peers_in_162.159": sorted(Counter(peer(i["pcap_file"]) for i in in_pfx))}
        if lab == 1:
            outside_on = [os.path.basename(it["pcap_file"].replace("\\", "/"))
                          for it in sub if not peer(it["pcap_file"]).startswith(PREFIX)]
    res["WARP_on_outside_equals_label_noise_flows"] = set(outside_on) == noise_keys
    res["n_label_noise_flows"] = len(noise_keys)
    res["WARP_on_outside_files"] = sorted(outside_on)
    res["note"] = ("labels in data.json are capture-run labels (before relabeling); "
                   "peer = endpoint outside 192.168.0.0/16 in the pcap file name")

    # [2026-10-16] 论文 VI-B 的 "103 distinct peers in 35 /16 prefixes": 改标后 (11 条 -> Direct) 按类统计
    relab = {}
    for cls, keep in (("Direct", lambda it: it["label"] == 0 or _base(it) in noise_keys),
                      ("MASQUE", lambda it: it["label"] == 1 and _base(it) not in noise_keys)):
        peers = [peer(it["pcap_file"]) for it in items if keep(it)]
        relab[cls] = {"n": len(peers), "distinct_peers": len(set(peers)),
                      "distinct_slash16": len({".".join(p.split(".")[:2]) for p in peers}),
                      "peer_in_162.159": sum(p.startswith(PREFIX) for p in peers)}
    res["relabelled"] = relab
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "WARP_on_outside_files"}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()

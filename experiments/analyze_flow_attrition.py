# -*- coding: utf-8 -*-
"""[2026-09-08] E6-G: 量化">= 50 包"过滤造成的流损耗, 以及它是否与类别相关。

## 动机

采集是 100 域名 x 5 轮 x 2 类, 每类理应约 500 条会话, 但数据集只有
Direct 234 / MASQUE 270 —— 也就是【一半以上的流被丢弃了, 且两类丢弃比例不同】。
论文完全没有报这个损耗率。

这构成一个未讨论的选择效应: 过滤条件是"UDP/443 且包数 >= 50", 而 MASQUE 因为
隧道分片, 同样的网页负载会产生更多的包, 天然更容易过线。于是留在数据集里的
Direct 是"包比较多的那部分 Direct", 两类的可比性被这一步悄悄改变了。

审稿人问"你们丢了多少、丢得偏不偏"时, 必须有数字可答。

顺带产出稳态段实验 (E6-C) 需要的数字: 有多少条流的包数 >= 100, 即可以在丢弃
前 50 包之后仍然凑满一个 50 包的窗口。

    wsl -e bash -lic 'cd /mnt/d/zemi/2026/dataset; python experiments/analyze_flow_attrition.py'
"""
import json
import os
import subprocess
from collections import Counter

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "data", "raw")
OUT = os.path.join(_ROOT, "outputs", "e6_attrition")
TSHARK = "/mnt/c/Program Files/Wireshark/tshark.exe"

MIN_PACKETS = 50     # split_quic_vs_masque_bi.py 的过滤阈值
STEADY_NEED = 100    # 稳态段实验需要的最小包数 (丢前 50 + 用后 50)


def win_path(p):
    """tshark 是 Windows 可执行文件, 读不了 /mnt/d/... 形式的 WSL 路径。"""
    p = os.path.abspath(p)
    if p.startswith("/mnt/") and len(p) > 6:
        return p[5].upper() + ":\\" + p[7:].replace("/", "\\")
    return p


def conversations(pcap):
    """返回 UDP/443 每条双向会话的包数列表。

    用 udp.stream 而不是 conv,udp 的表格: tshark 的字节列带单位 (kB/MB), 列宽随
    数据变化, 正则解析很脆; udp.stream 本身就是 tshark 给双向会话编的号,
    与切流脚本的规范化五元组同粒度, 逐包计数即可。
    """
    p = subprocess.run(
        [TSHARK, "-r", win_path(pcap), "-Y", "udp.port==443",
         "-T", "fields", "-e", "udp.stream"],
        capture_output=True, text=True)
    c = Counter(s for s in p.stdout.split() if s.isdigit())
    if not c:
        raise SystemExit(f"tshark 无输出 ({pcap}): {p.stderr.strip()[:300]}")
    if p.returncode != 0:
        # 10 个 pcap 都是抓包被中止时在包中间截断的, tshark 会报 "cut short in the
        # middle of a packet" 并返回非 0, 但截断点之前的包已全部解析完毕。
        # 影响仅限每个文件最后那一个不完整的包, 对会话计数无实质影响。
        print(f"    (注: {os.path.basename(pcap)} 末尾截断, 已解析 {sum(c.values())} 个 UDP/443 包)")
    return list(c.values())


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".pcap"))
    per_class = {}

    print(f"{'pcap':<46}{'UDP/443会话':>12}{'>=50包':>9}{'>=100包':>9}{'留存%':>8}")
    print("-" * 90)
    for f in files:
        cls = "Direct" if f.startswith("class0") else "MASQUE"
        c443 = conversations(os.path.join(RAW, f))
        ge50 = [n for n in c443 if n >= MIN_PACKETS]
        ge100 = [n for n in c443 if n >= STEADY_NEED]
        keep = len(ge50) / len(c443) * 100 if c443 else 0.0
        print(f"{f:<46}{len(c443):>12}{len(ge50):>9}{len(ge100):>9}{keep:>7.1f}%")

        d = per_class.setdefault(cls, {"udp": 0, "udp443": 0, "ge50": 0, "ge100": 0,
                                       "files": 0, "pkts443": []})
        d["udp"] += len(c443)
        d["udp443"] += len(c443)
        d["ge50"] += len(ge50)
        d["ge100"] += len(ge100)
        d["files"] += 1
        d["pkts443"] += c443

    print("-" * 90)
    result = {"min_packets_filter": MIN_PACKETS, "steady_need": STEADY_NEED, "classes": {}}
    for cls, d in per_class.items():
        keep = d["ge50"] / d["udp443"] * 100
        keep100 = d["ge100"] / d["udp443"] * 100
        pk = sorted(d["pkts443"])
        med = pk[len(pk) // 2]
        print(f"{cls:<46}{d['udp']:>9}{d['udp443']:>9}{d['ge50']:>9}{d['ge100']:>9}{keep:>7.1f}%")
        result["classes"][cls] = {
            "udp_conversations": d["udp"], "udp443_conversations": d["udp443"],
            "ge50": d["ge50"], "ge100": d["ge100"],
            "retention_ge50_pct": round(keep, 2),
            "retention_ge100_pct": round(keep100, 2),
            "median_packets_udp443": med,
        }

    a, b = result["classes"].get("Direct"), result["classes"].get("MASQUE")
    if a and b:
        # 过滤是否与类别相关: 2x2 Fisher 精确检验
        try:
            from scipy.stats import fisher_exact
            tbl = [[a["ge50"], a["udp443_conversations"] - a["ge50"]],
                   [b["ge50"], b["udp443_conversations"] - b["ge50"]]]
            odds, p = fisher_exact(tbl)
            result["fisher_retention_vs_class"] = {"table": tbl, "odds_ratio": round(float(odds), 3),
                                                   "p": float(p)}
            print(f"\n留存率是否与类别相关 (Fisher 精确检验): "
                  f"Direct {a['retention_ge50_pct']:.1f}% vs MASQUE {b['retention_ge50_pct']:.1f}%, "
                  f"OR={odds:.2f}, p={p:.3g}")
        except ImportError:
            pass
        print(f"UDP/443 会话包数中位数: Direct {a['median_packets_udp443']} / "
              f"MASQUE {b['median_packets_udp443']}")
        print(f"\n稳态段实验 (E6-C) 可用流数 (>= {STEADY_NEED} 包): "
              f"Direct {a['ge100']} / MASQUE {b['ge100']}")

    path = os.path.join(OUT, "summary_attrition.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()

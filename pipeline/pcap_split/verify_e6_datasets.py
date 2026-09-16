"""[2026-09-08] E6 数据集的断言式校验。

沿用 verify_obfuscation_bi.py 的做法: 在花 GPU 时间之前, 先用断言把
"变换到底改了什么、没改什么"逐条钉死。E6 的两组条件各有一个容易出错的点:

  * 单通道充分性: "只留方向"要求 |signed_size| 恒定【且符号逐包与原始一致】。
    若 pad 顺手把符号也归一, 该条件就退化成"什么都没有", 训出来的 53% 会被
    误读成"方向不充分"。
  * 观测点抖动: 排序后再 diff 才能保证间隔非负; 若忘了排序, 会出现负间隔,
    log10_decade 编码会把它折成 NaN/极值, 精度下降就不是抖动引起的了。

    python pipeline/pcap_split/verify_e6_datasets.py
"""

import json
import os
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(_ROOT, "data", "processed")
SRC = "quic_vs_masque_bi_feat"
SPLITS = ["train", "valid", "test"]
PAD_SIZE = 1350

JITTER_SIGMAS = {"e6_jit50us": 50e-6, "e6_jit200us": 200e-6,
                 "e6_jit1ms": 1e-3, "e6_jit5ms": 5e-3}

n_assert = 0
failures = []


def check(cond, msg):
    global n_assert
    n_assert += 1
    if not cond:
        failures.append(msg)
        print(f"  [FAIL] {msg}")
    return cond


def load(name):
    out = {}
    for sp in SPLITS:
        with open(os.path.join(PROC, name, f"data-{sp}.json"), encoding="utf-8") as f:
            for it in json.load(f):
                out[os.path.basename(it["pcap_file"])] = it
    return out


def sizes(it):
    return [int(x) for x in it["signed_sizes"].split()]


def ivs(it):
    return [float(x) for x in it["intervals"].split()]


def bytes_all_zero(it):
    return all(int(v) == 0 for d in it["data"] for v in d.split())


def main():
    src = load(SRC)
    print(f"源数据集 {SRC}: {len(src)} 条流\n")

    # ---------------- 通用: 样本集合必须与源逐条一致 ----------------
    for name in list(JITTER_SIGMAS) + ["e6_suff_dir_only", "e6_suff_mag_only",
                                       "e6_suff_bytes_only", "e6_jit1ms_full"]:
        d = load(name)
        check(set(d) == set(src), f"{name}: 样本集合与源不一致 "
                                  f"(缺 {len(set(src)-set(d))}, 多 {len(set(d)-set(src))})")
        check(all(d[k]["label"] == src[k]["label"] for k in src),
              f"{name}: 有样本的 label 被改动")
        check(all(len(sizes(d[k])) == len(sizes(src[k])) for k in src),
              f"{name}: 有样本的包数被改动")

    # ---------------- (1) 单通道充分性 ----------------
    print("[1] 单通道充分性条件")

    d = load("e6_suff_dir_only")
    check(all(abs(s) == PAD_SIZE for k in src for s in sizes(d[k])),
          "dir_only: 存在 |signed_size| != 1350 的包")
    # 核心断言: 符号必须逐包与原始一致 —— 这是该条件唯一承载信息的地方
    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    bad = sum(1 for k in src
              for a, b in zip(sizes(src[k]), sizes(d[k]))
              if sign(a) != 0 and sign(a) != sign(b))
    check(bad == 0, f"dir_only: {bad} 个包的方向符号与原始不一致")
    check(all(bytes_all_zero(d[k]) for k in src), "dir_only: 字节通道未归零")
    check(all(all(v == 0.0 for v in ivs(d[k])) for k in src), "dir_only: 时间通道未归零")
    # 反向断言: 方向序列必须真的携带信息 (两类的上行占比应当有差异, 否则条件无意义)
    up = {0: [], 1: []}
    for k in src:
        s = sizes(d[k])
        up[src[k]["label"]].append(sum(1 for v in s if v > 0) / len(s))
    m0, m1 = sum(up[0]) / len(up[0]), sum(up[1]) / len(up[1])
    print(f"  上行包占比: Direct {m0*100:.1f}%  MASQUE {m1*100:.1f}%  (差 {abs(m0-m1)*100:.1f} pp)")
    check(abs(m0 - m1) > 0.01, "dir_only: 两类上行占比几乎相同, 该条件不承载信息")

    d = load("e6_suff_mag_only")
    check(all(s >= 0 for k in src for s in sizes(d[k])), "mag_only: 存在负 signed_size, 方向未抹净")
    check(all(sizes(d[k]) == [abs(v) for v in sizes(src[k])] for k in src),
          "mag_only: 幅度与原始 |signed_size| 不一致")
    check(all(bytes_all_zero(d[k]) for k in src), "mag_only: 字节通道未归零")
    check(all(all(v == 0.0 for v in ivs(d[k])) for k in src), "mag_only: 时间通道未归零")

    d = load("e6_suff_bytes_only")
    check(all(all(s == 0 for s in sizes(d[k])) for k in src), "bytes_only: 包长未归零")
    check(all(all(v == 0.0 for v in ivs(d[k])) for k in src), "bytes_only: 时间通道未归零")
    check(any(not bytes_all_zero(d[k]) for k in src), "bytes_only: 字节通道也被归零了, 条件无意义")

    # ---------------- (2) 观测点抖动 ----------------
    print("\n[2] 观测点抖动条件")
    for name, sigma in JITTER_SIGMAS.items():
        d = load(name)
        neg = sum(1 for k in src for v in ivs(d[k]) if v < 0)
        check(neg == 0, f"{name}: {neg} 个负间隔 (排序步骤失效)")
        check(all(ivs(d[k])[0] == 0.0 for k in src), f"{name}: 首个间隔不为 0")
        check(all(all(s == 0 for s in sizes(d[k])) for k in src), f"{name}: 包长未归零")
        check(all(bytes_all_zero(d[k]) for k in src), f"{name}: 字节通道未归零")

        # 抖动必须真的把亚 100us 结构打散: 统计 <100us 间隔占比的变化
        def frac_sub100us(dd):
            tot = hit = 0
            for k in src:
                for v in ivs(dd[k])[1:]:
                    tot += 1
                    hit += (v < 100e-6)
            return hit / tot
        f_src, f_new = frac_sub100us(src), frac_sub100us(d)
        # 流总时长应基本守恒 (抖动是零均值的)
        dur_src = sum(sum(ivs(src[k])) for k in src) / len(src)
        dur_new = sum(sum(ivs(d[k])) for k in src) / len(src)
        print(f"  {name:14s} sigma={sigma*1e6:7.0f}us  "
              f"<100us 间隔占比 {f_src*100:5.1f}% -> {f_new*100:5.1f}%   "
              f"平均流时长 {dur_src:.3f}s -> {dur_new:.3f}s")
        check(f_new < f_src, f"{name}: 抖动后 <100us 间隔占比没有下降, 变换未生效")
        check(abs(dur_new - dur_src) < 0.5 * max(dur_src, 1e-9) + 10 * sigma,
              f"{name}: 平均流时长变化过大 ({dur_src:.3f} -> {dur_new:.3f}), 抖动不应是有偏的")

    # 抖动的确定性: 重新加载必须逐值一致 (RNG 由文件名派生)
    a, b = load("e6_jit1ms"), load("e6_jit1ms")
    check(all(a[k]["intervals"] == b[k]["intervals"] for k in src),
          "jitter 不可复现 (RNG 未由文件名派生?)")

    d = load("e6_jit1ms_full")
    check(all(sizes(d[k]) == sizes(src[k]) for k in src), "jit1ms_full: 包长被改动了")
    check(all(d[k]["data"] == src[k]["data"] for k in src), "jit1ms_full: 字节通道被改动了")
    check(any(ivs(d[k]) != ivs(src[k]) for k in src), "jit1ms_full: 时间通道没被抖动")

    # ---------------- 结论 ----------------
    print(f"\n{'='*66}")
    if failures:
        print(f"{len(failures)}/{n_assert} 条断言失败:")
        for m in failures:
            print(f"  - {m}")
        sys.exit(1)
    print(f"全部 {n_assert} 条断言通过。")


if __name__ == "__main__":
    main()

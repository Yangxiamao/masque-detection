"""
[2026-07-20] 校验 simulate_obfuscation_bi.py 的输出。

对应 NEXT_adversarial_training_handoff.md Step 2 ("别跳过")。
逐包核对: padding 后符号与原符号一致、绝对值均为 1350、intervals 未被误改、
data 未被误改; 并额外验证会话级零泄漏在对抗训练集里依然成立。

任何一项失败即 assert 抛错, 不做"看起来对"的目视判断。
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from split_data import session_key  # noqa: E402

PAD = 1350
SPLITS = ["train", "valid", "test"]


def base_dir():
    return _ROOT


PROC = os.path.join(base_dir(), "data", "processed")
SRC = os.path.join(PROC, "quic_vs_masque_bi_feat")


def load(name, split):
    with open(os.path.join(PROC, name, f"data-{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def ints(s):
    return [int(x) for x in s.split()]


def floats(s):
    return [float(x) for x in s.split()]


ok = []


def check(label, cond, detail=""):
    assert cond, f"FAIL: {label} {detail}"
    ok.append(label)
    print(f"  [OK] {label}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. 保号 padding
print("\n=== 1. bi_t2_pad: 保号 padding 逐包核对 (test, 全 51 条) ===")
src_test = load("quic_vs_masque_bi_feat", "test")
pad_test = load("bi_t2_pad", "test")

check("条数一致", len(src_test) == len(pad_test) == 51, f"{len(pad_test)} 条")

n_pkt = 0
for a, b in zip(src_test, pad_test):
    check_id = os.path.basename(a["pcap_file"])
    assert a["label"] == b["label"] and a["pcap_file"] == b["pcap_file"], f"顺序错位 {check_id}"
    sa, sb = ints(a["signed_sizes"]), ints(b["signed_sizes"])
    assert len(sa) == len(sb), f"包数变了 {check_id}"
    for i, (x, y) in enumerate(zip(sa, sb)):
        assert (x >= 0) == (y >= 0), f"符号翻转 {check_id} pkt{i}: {x} -> {y}"
        assert abs(y) == PAD, f"绝对值不是 {PAD}: {check_id} pkt{i} -> {y}"
        n_pkt += 1
    assert a["intervals"] == b["intervals"], f"intervals 被误改 {check_id}"
    assert a["data"] == b["data"], f"data 被误改 {check_id}"
    assert a["num_packet"] == b["num_packet"] and a["name"] == b["name"]

check("符号全部保持 + 绝对值全为 1350", True, f"共核对 {n_pkt} 个包")
check("intervals 未被改动", True)
check("data 未被改动", True)

# 抽一条流打印首尾，人工可读的旁证
a, b = src_test[0], pad_test[0]
print(f"\n  样例流 {os.path.basename(a['pcap_file'])[:60]} (label={a['label']})")
print(f"    原始 signed_sizes[:10] = {ints(a['signed_sizes'])[:10]}")
print(f"    padded signed_sizes[:10] = {ints(b['signed_sizes'])[:10]}")
neg = [i for i, v in enumerate(ints(a["signed_sizes"])) if v < 0][:5]
print(f"    下行包位置 {neg} 在 padded 中的值 = {[ints(b['signed_sizes'])[i] for i in neg]}")

# ---------------------------------------------------------------- 2. 双向比例
print("\n=== 2. 方向分布在 padding 前后不变 (方向信息确实被保住) ===")
for tag, ds in [("clean", src_test), ("padded", pad_test)]:
    up = sum(1 for it in ds for v in ints(it["signed_sizes"]) if v > 0)
    dn = sum(1 for it in ds for v in ints(it["signed_sizes"]) if v < 0)
    print(f"    {tag:7s}: 上行 {up}  下行 {dn}")
u1 = sum(1 for it in src_test for v in ints(it["signed_sizes"]) if v > 0)
u2 = sum(1 for it in pad_test for v in ints(it["signed_sizes"]) if v > 0)
check("上/下行包数完全一致", u1 == u2, f"{u1} == {u2}")

# ---------------------------------------------------------------- 3. masque_only
print("\n=== 3. bi_t2_pad_masque_only: 只有 label==1 被 padding ===")
mo_test = load("bi_t2_pad_masque_only", "test")
n_d, n_m = 0, 0
for a, b in zip(src_test, mo_test):
    if a["label"] == 0:
        assert a["signed_sizes"] == b["signed_sizes"], "Direct 不该被改"
        n_d += 1
    else:
        assert all(abs(v) == PAD for v in ints(b["signed_sizes"])), "MASQUE 该被 pad"
        n_m += 1
check("Direct 保持原样 / MASQUE 全部 padded", True, f"Direct {n_d} 条未改, MASQUE {n_m} 条已 pad")

# ---------------------------------------------------------------- 4. strict
print("\n=== 4. bi_t2_pad_strict: 字节通道确已归零 ===")
st_test = load("bi_t2_pad_strict", "test")
allz = all(int(v) == 0 for it in st_test for d in it["data"] for v in d.split())
same_pad = all(ints(x["signed_sizes"]) == ints(y["signed_sizes"])
               for x, y in zip(pad_test, st_test))
check("data 全零", allz)
check("signed_sizes 与 T2 相同", same_pad)
check("intervals 未动", all(x["intervals"] == y["intervals"] for x, y in zip(src_test, st_test)))

# ---------------------------------------------------------------- 5. shaping
print("\n=== 5. bi_t3_pad_shape / bi_t_shape: intervals 变换 ===")
t3 = load("bi_t3_pad_shape", "test")
sh = load("bi_t_shape", "test")
for name, ds in [("T3", t3), ("shape-only", sh)]:
    for it in ds:
        iv = floats(it["intervals"])
        assert iv[0] == 0.0, f"{name} 首包间隔应为 0"
        assert all(abs(v - 0.005) < 1e-9 for v in iv[1:]), f"{name} 非首包应为 0.005"
check("T3 与 shape-only 的 intervals 均为 CBR 0.005s (首包 0)", True)
check("T3 的 signed_sizes 已 padding",
      all(abs(v) == PAD for it in t3 for v in ints(it["signed_sizes"])))
check("shape-only 的 signed_sizes 未动",
      all(x["signed_sizes"] == y["signed_sizes"] for x, y in zip(src_test, sh)))

# ---------------------------------------------------------------- 6. 归因消融集
print("\n=== 6. 归因消融评测集 ===")
noint = load("bi_t2_pad_noint", "test")
absd = load("bi_t2_pad_absdir", "test")
check("bi_t2_pad_noint: intervals 全零",
      all(float(v) == 0.0 for it in noint for v in it["intervals"].split()))
check("bi_t2_pad_noint: signed_sizes 仍为保号 padding",
      all(abs(v) == PAD for it in noint for v in ints(it["signed_sizes"])))
check("bi_t2_pad_absdir: 方向已抹除 (全为 +1350)",
      all(v == PAD for it in absd for v in ints(it["signed_sizes"])))
check("bi_t2_pad_absdir: intervals 未动",
      all(x["intervals"] == y["intervals"] for x, y in zip(src_test, absd)))

# ---------------------------------------------------------------- 7. No-Sizes 上界
print("\n=== 7. bi_nosizes: 全 split 包长归零 ===")
for split in SPLITS:
    ns = load("bi_nosizes", split)
    assert all(v == 0 for it in ns for v in ints(it["signed_sizes"])), f"{split} 未归零"
    src = load("quic_vs_masque_bi_feat", split)
    assert all(x["intervals"] == y["intervals"] for x, y in zip(src, ns)), f"{split} intervals 被动"
    assert all(x["data"] == y["data"] for x, y in zip(src, ns)), f"{split} data 被动"
check("train/valid/test 的 signed_sizes 全为 0, 且 intervals/data 未动", True)

# ---------------------------------------------------------------- 8. 对抗训练集
print("\n=== 8. bi_adv: 对抗训练集构成 ===")
adv_tr = load("bi_adv", "train")
adv_va = load("bi_adv", "valid")
adv_te = load("bi_adv", "test")
src_tr = load("quic_vs_masque_bi_feat", "train")
src_va = load("quic_vs_masque_bi_feat", "valid")

check("train = 806 (403 clean + 403 padded)", len(adv_tr) == 806, f"{len(adv_tr)}")
check("valid = 100 (50 clean + 50 padded)", len(adv_va) == 100, f"{len(adv_va)}")
check("test = 51 且保持干净 (T1)",
      len(adv_te) == 51 and all(x["signed_sizes"] == y["signed_sizes"]
                                for x, y in zip(src_test, adv_te)))

half = len(src_tr)
check("train 前半为原样 clean",
      all(a["signed_sizes"] == b["signed_sizes"] for a, b in zip(src_tr, adv_tr[:half])))
check("train 后半为保号 padded",
      all(abs(v) == PAD for it in adv_tr[half:] for v in ints(it["signed_sizes"])))
check("train 后半符号与对应 clean 一致",
      all((x >= 0) == (y >= 0)
          for a, b in zip(src_tr, adv_tr[half:])
          for x, y in zip(ints(a["signed_sizes"]), ints(b["signed_sizes"]))))
check("variant 标签正确",
      [it["variant"] for it in adv_tr[:half]] == ["clean"] * half
      and [it["variant"] for it in adv_tr[half:]] == ["pad"] * half)

# padding 与 label 统计独立 —— §4.2② 的核心
from collections import Counter  # noqa: E402

# 仓库根:优先取 MASQUE_ROOT 环境变量,否则由本文件位置推导。
# 如此 git worktree 中的副本会各自指向自己的目录,输出互不覆盖。
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ct = Counter((it["label"], it["variant"]) for it in adv_tr)
print(f"\n    (label, variant) 分布: {dict(ct)}")
check("padding 与 label 统计独立 (两类的 clean/padded 数量各自相等)",
      ct[(0, "clean")] == ct[(0, "pad")] and ct[(1, "clean")] == ct[(1, "pad")],
      f"Direct {ct[(0,'clean')]}/{ct[(0,'pad')]}, Masque {ct[(1,'clean')]}/{ct[(1,'pad')]}")

# ---------------------------------------------------------------- 9. 会话级零泄漏
print("\n=== 9. 会话级零泄漏 (padding 在切分之后做, 不得引入泄漏) ===")
for ds_name in ["quic_vs_masque_bi_feat", "bi_adv", "bi_adv_masque_only", "bi_nosizes"]:
    sets = {}
    for split in SPLITS:
        sets[split] = {session_key(it) for it in load(ds_name, split)}
    tr_va = sets["train"] & sets["valid"]
    tr_te = sets["train"] & sets["test"]
    va_te = sets["valid"] & sets["test"]
    assert not (tr_va or tr_te or va_te), \
        f"{ds_name} 会话泄漏! train∩valid={len(tr_va)} train∩test={len(tr_te)} valid∩test={len(va_te)}"
    total = len(sets["train"]) + len(sets["valid"]) + len(sets["test"])
    print(f"    {ds_name:24s} 会话数 train/valid/test = "
          f"{len(sets['train'])}/{len(sets['valid'])}/{len(sets['test'])} (共 {total}), 交集全空")
check("四个数据集的 train/valid/test 会话交集均为空", True)

print(f"\n{'='*66}")
print(f"全部 {len(ok)} 项校验通过。")
print(f"{'='*66}")

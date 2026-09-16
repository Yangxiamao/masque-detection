"""
[2026-07-20] 双向管道 (signed_sizes) 的混淆 / 消融数据集生成器。

替代旧的 simulate_obfuscation.py —— 旧脚本是针对【旧单向管道】写的, 有三个致命问题:
  1. 读 item["sizes"] (无符号), 双向管道的字段名是 signed_sizes
  2. `if item["label"] == 0: return item` —— 只给 MASQUE 加 padding。
     对抗训练必须两类都加, 否则模型立刻学到新捷径"包长恒定 = MASQUE"
  3. 源目录写死 quic_vs_masque_feat, 且只处理 data-test.json

核心设计要点 (见 NEXT_adversarial_training_handoff.md §4.2):
  * padding 必须【保号】: 正 -> +1350, 负 -> -1350。
    若变成无符号, 方向信息会被 padding 顺手一起抹掉, 就成了"同时隐藏包长和方向",
    两个变量混在一起无法归因。
  * padding 必须在【会话级切分之后】做, 故本脚本直接对已切好的
    data-train/valid/test.json 分别施加变换, 绝不重新切分。
  * 输出目录总是写全 data-train/valid/test.json + metadata.json 四件套,
    因为 fine-tune.py 即使跑 --eval 也会加载 train/valid (见 fine-tune.py L79-84)。

已知局限 (2026-07-20 实测确认, 结论中必须标注):
  data 字节通道里【明文残留包长】—— 全部包为 IPv4, wire_len - ip_total_len 恒为 14,
  即 bytes[2:4] = IP total length, bytes[24:26] = UDP length; 且 payload 区的零填充
  长度本身也泄漏包长 (157B 包只有 115/240 非零, 满 MTU 包 240/240)。
  因此"只改 signed_sizes 的 padding"并未真正剥夺全部长度线索。
  -> 故额外提供 nobytes 变换, 构造 T2-strict (padded + 字节通道全零) 作为夹逼下界。
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys

# 仓库根:优先取 MASQUE_ROOT 环境变量,否则由本文件位置推导。
# 如此 git worktree 中的副本会各自指向自己的目录,输出互不覆盖。
_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAD_SIZE = 1350          # MTU padding 目标包长
SHAPE_INTERVAL = "0.005000"  # CBR shaping 的固定间隔 (沿用旧脚本定义)
SPLITS = ["train", "valid", "test"]


# --------------------------------------------------------------------------
# 变换原语。每个都返回新对象, 绝不原地改。
# --------------------------------------------------------------------------

def tf_pad(item):
    """MTU padding, 保号。正(上行) -> +1350, 负(下行) -> -1350。"""
    new = copy.deepcopy(item)
    new["signed_sizes"] = " ".join(
        str(PAD_SIZE) if int(s) >= 0 else str(-PAD_SIZE)
        for s in item["signed_sizes"].split()
    )
    return new


def tf_shape(item):
    """CBR interval shaping。全部置 0.005s, 第 1 个保持 0.0。"""
    new = copy.deepcopy(item)
    ivs = item["intervals"].split()
    new_ivs = [SHAPE_INTERVAL] * len(ivs)
    if new_ivs:
        new_ivs[0] = "0.0"
    new["intervals"] = " ".join(new_ivs)
    return new


def tf_nosizes(item):
    """消融: 包长信息全部归零 (loader 会 shift 成常数 MTU, 等价于无信息)。"""
    new = copy.deepcopy(item)
    n = len(item["signed_sizes"].split())
    new["signed_sizes"] = " ".join(["0"] * n)
    return new


def tf_nobytes(item):
    """消融: 字节通道全部归零。用于切断 IP/UDP 头长度字段的泄漏。"""
    new = copy.deepcopy(item)
    new["data"] = [" ".join(["0"] * len(d.split())) for d in item["data"]]
    return new


def tf_nointervals(item):
    """消融: 时间间隔全部归零。"""
    new = copy.deepcopy(item)
    n = len(item["intervals"].split())
    new["intervals"] = " ".join(["0.0"] * n)
    return new


def tf_absdir(item):
    """消融: 抹除方向, 只留包长绝对值 (回答'回升是否靠方向')。"""
    new = copy.deepcopy(item)
    new["signed_sizes"] = " ".join(
        str(abs(int(s))) for s in item["signed_sizes"].split()
    )
    return new


# --------------------------------------------------------------------------
# [2026-09-08 E6] 观测点抖动 (jitter)。
#
# 动机: E3 的 timing-only 条件 (93.85%) 里, RF 的头部重要度全部落在 <100 us 的
# 间隔占比上。但本项目在【客户端本机】用 tshark 抓包, 2-3 us 量级的包间隔几乎
# 只能来自 NIC/内核侧的 GRO/中断合并, 而不是网络可观测的结构。论文的威胁模型
# 却写着"运营商侧的中间盒", 那个观测点看不到这些微秒级结构。
#
# 故本变换模拟观测点从客户端移到网络中途所引入的路径抖动:
#   t      = cumsum(intervals)                # 还原到达时刻
#   t'     = t + N(0, sigma)   逐包独立        # 路径抖动 (可致乱序)
#   t''    = sort(t') - min(t')                # 乱序在观测点表现为重排
#   新 intervals = diff(t''), 首个置 0
# 排序保证 diff 非负 —— 观测点看到的只能是单调时刻序列。
#
# 随机数按 pcap 文件名派生, 故同一条流在任何条件/任何折里得到同一份抖动,
# 结果可复现且与折分配无关。
# --------------------------------------------------------------------------

JITTER_SIGMAS = {          # 变换名后缀 -> sigma (秒)
    "jitter50us": 50e-6,
    "jitter200us": 200e-6,
    "jitter1ms": 1e-3,
    "jitter5ms": 5e-3,
}


def _flow_rng(item, salt):
    """由流文件名派生的确定性 RNG (str 的内置 hash 有进程级盐, 不能用)。"""
    key = f"{os.path.basename(item['pcap_file'])}|{salt}".encode()
    return random.Random(int(hashlib.md5(key).hexdigest()[:8], 16))


def make_tf_jitter(sigma, salt):
    def tf_jitter(item):
        new = copy.deepcopy(item)
        ivs = [float(x) for x in item["intervals"].split()]
        if not ivs:
            return new
        rng = _flow_rng(item, salt)
        t, acc = [], 0.0
        for x in ivs:
            acc += x
            t.append(acc)
        t = sorted(v + rng.gauss(0.0, sigma) for v in t)
        base = t[0]
        out = [0.0] + [t[i] - t[i - 1] for i in range(1, len(t))]
        del base
        new["intervals"] = " ".join(f"{v:.6f}" for v in out)
        return new
    return tf_jitter


# --------------------------------------------------------------------------
# [2026-09-14 v2 采集计划预备测试 / L8] 时间戳分辨率变换。
#
# v1 的 intervals 是 1 us 量化的 (wrpcap 写经典 pcap + round(6)), 而原始 pcapng 是
# 100 ns。log10_decade 编码把 0 映到 0、1 us 映到 100, 于是 [0, 0.5 us) 的间隔
# (占 19%) 被整体压到编码值 0 —— 需要检验 timing 的判别力是否来自这个量化台阶。
#   dequant1us : 到达时刻 + U(-0.5, +0.5) us 后重排再差分 (去量化抖动, 计划 §6 的做法);
#   quant10us  : 到达时刻舍入到 10 us 网格后差分 (更粗的量化, 看衰减方向)。
# 与 make_tf_jitter 同样按文件名派生 RNG, 逐条可复现。输出保留 7 位小数 (100 ns)。
# --------------------------------------------------------------------------

def _resample_intervals(item, transform_ts, decimals=7):
    new = copy.deepcopy(item)
    ivs = [float(x) for x in item["intervals"].split()]
    if not ivs:
        return new
    t, acc = [], 0.0
    for x in ivs:
        acc += x
        t.append(acc)
    t = sorted(transform_ts(t))
    out = [0.0] + [t[i] - t[i - 1] for i in range(1, len(t))]
    new["intervals"] = " ".join(f"{v:.{decimals}f}" for v in out)
    return new


def tf_dequant1us(item):
    rng = _flow_rng(item, "dequant1us")
    return _resample_intervals(item, lambda t: [v + rng.uniform(-0.5e-6, 0.5e-6) for v in t])


def tf_quant10us(item):
    return _resample_intervals(item, lambda t: [round(v * 1e5) / 1e5 for v in t])


TRANSFORMS = {
    "pad": tf_pad,
    "shape": tf_shape,
    "nosizes": tf_nosizes,
    "nobytes": tf_nobytes,
    "nointervals": tf_nointervals,
    "absdir": tf_absdir,
    "dequant1us": tf_dequant1us,
    "quant10us": tf_quant10us,
}
TRANSFORMS.update({name: make_tf_jitter(s, name) for name, s in JITTER_SIGMAS.items()})

# 防御类变换受 --targets 约束 (现实中由某一方部署);
# 消融类变换是对模型输入的遮蔽, 恒作用于全部样本。
DEFENSE_TRANSFORMS = {"pad", "shape"}


# --------------------------------------------------------------------------

def apply_transforms(item, transform_names, targets):
    """按顺序施加变换。targets 仅约束防御类变换。"""
    out = item
    for name in transform_names:
        if name in DEFENSE_TRANSFORMS and targets == "masque_only" and item["label"] != 1:
            continue
        out = TRANSFORMS[name](out)
    return out


def build_split(src_items, transform_names, targets, mix_clean, tag):
    """返回变换后的 split。mix_clean=True 时 = 干净版 + 变换版 拼接。"""
    transformed = []
    for it in src_items:
        new = apply_transforms(it, transform_names, targets)
        new = copy.deepcopy(new) if new is it else new
        new["variant"] = tag
        transformed.append(new)

    if not mix_clean:
        return transformed

    clean = []
    for it in src_items:
        c = copy.deepcopy(it)
        c["variant"] = "clean"
        clean.append(c)
    return clean + transformed


def generate(src_dir, out_dir, transform_names, targets, apply_to, mix_clean, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    tag = "+".join(transform_names) if transform_names else "clean"

    counts = {}
    for split in SPLITS:
        src_path = os.path.join(src_dir, f"data-{split}.json")
        with open(src_path, "r", encoding="utf-8") as f:
            src_items = json.load(f)

        if split in apply_to:
            items = build_split(src_items, transform_names, targets, mix_clean, tag)
        else:
            items = []
            for it in src_items:
                c = copy.deepcopy(it)
                c["variant"] = "clean"
                items.append(c)

        with open(os.path.join(out_dir, f"data-{split}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        counts[split] = len(items)

    with open(os.path.join(src_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        applied = ",".join(sorted(apply_to)) if apply_to else "-"
        print(f"  {os.path.basename(out_dir):26s} tf={tag:16s} targets={targets:11s} "
              f"apply_to={applied:18s} mix={int(mix_clean)}  "
              f"train/valid/test = {counts['train']}/{counts['valid']}/{counts['test']}")
    return counts


# --------------------------------------------------------------------------
# 实验矩阵。一次生成全部所需数据集, 保证可复现。
# --------------------------------------------------------------------------

MATRIX = [
    # --- 评测集: 全部来自同一 test split 的 51 条, 训练/验证保持干净 ---
    dict(name="bi_t2_pad",             tf=["pad"],            targets="both",
         apply_to=["test"], mix=False,
         note="T2 主指标: 两类都 padding, 保号"),
    dict(name="bi_t2_pad_strict",      tf=["pad", "nobytes"], targets="both",
         apply_to=["test"], mix=False,
         note="T2-strict: 再切断字节通道, 保证零长度泄漏 (夹逼下界)"),
    dict(name="bi_t3_pad_shape",       tf=["pad", "shape"],   targets="both",
         apply_to=["test"], mix=False,
         note="T3: padding + shaping 同时施加"),
    dict(name="bi_t_shape",            tf=["shape"],          targets="both",
         apply_to=["test"], mix=False,
         note="仅 shaping (旧实验此项 100%, 双向管道重算)"),
    dict(name="bi_t2_pad_masque_only", tf=["pad"],            targets="masque_only",
         apply_to=["test"], mix=False,
         note="旧实验定义 (只 pad MASQUE) 的双向重算, 与旧 41.76% 最可比"),

    # --- 归因消融用评测集 (在对抗训练模型上跑) ---
    dict(name="bi_t2_pad_noint",       tf=["pad", "nointervals"], targets="both",
         apply_to=["test"], mix=False,
         note="归因: T2 再关掉时间间隔"),
    dict(name="bi_t2_pad_absdir",      tf=["pad", "absdir"],      targets="both",
         apply_to=["test"], mix=False,
         note="归因: T2 再抹掉方向"),

    # --- [2026-09-06 E5] T3 的归因评测集 (见 docs/01_Project_Management/E5_design.md §2.2) ---
    dict(name="bi_t3_pad_shape_noint",  tf=["pad", "shape", "nointervals"], targets="both",
         apply_to=["test"], mix=False,
         note="归因: T3 再关时间 -> 与 T3 同分则说明模型没在用时间"),
    dict(name="bi_t3_pad_shape_absdir", tf=["pad", "shape", "absdir"],      targets="both",
         apply_to=["test"], mix=False,
         note="归因: T3 再抹方向 -> B4 三通道联合防御的天花板"),

    # --- 对抗训练集: train/valid 混入 padded 版, test 保持干净 (T1) ---
    dict(name="bi_adv",                tf=["pad"],            targets="both",
         apply_to=["train", "valid"], mix=True,
         note="B1 主实验: 403 clean + 403 padded = 806"),
    dict(name="bi_adv_masque_only",    tf=["pad"],            targets="masque_only",
         apply_to=["train", "valid"], mix=True,
         note="附加对照: 只 pad MASQUE 的对抗训练"),
    # [2026-09-06 E5] shaping 对抗训练的主实验数据集 (8/18 ゼミ承诺的实验②)。
    # 与 bi_adv 严格同构, 唯一差别是变换从 ["pad"] 变成 ["pad","shape"]。
    dict(name="bi_adv_shape",          tf=["pad", "shape"],   targets="both",
         apply_to=["train", "valid"], mix=True,
         note="E5 主实验: 403 clean + 403 pad+shape = 806"),

    # --- 能力上界: 训练与测试都完全看不到包长 ---
    dict(name="bi_nosizes",            tf=["nosizes"],        targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="No-Sizes 上界: 全 split 包长归零"),
]


# --------------------------------------------------------------------------
# 特征消融矩阵 (A1 重跑)。方法论沿用 2026-06-29 的旧消融实验:
# 变换施加于【全部 split】, 每组单独重训一个模型 —— 即"模型从头就看不到该特征"。
#
# 注意 bi_nosizes 已在 MATRIX 中定义 (对抗训练实验的能力上界), 此处直接复用,
# 不重复生成。
# --------------------------------------------------------------------------

ABLATION_MATRIX = [
    dict(name="bi_abl_no_bytes",         tf=["nobytes"],     targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="遮蔽载荷字节 (对标旧 No-Bytes = 100%)"),
    dict(name="bi_abl_no_intervals",     tf=["nointervals"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="遮蔽时间间隔 (对标旧 No-Intervals = 100%)"),
    dict(name="bi_abl_no_direction",     tf=["absdir"],      targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="【双向管道新增】只抹方向, 保留绝对包长 -> 量化方向本身的贡献"),
    dict(name="bi_abl_no_sizes_nobytes", tf=["nosizes", "nobytes"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="遮蔽包长 + 载荷 -> 控制字节通道里残留的 IP/UDP 长度字段泄漏"),
]


# --------------------------------------------------------------------------
# [2026-09-08 E6] 审稿意见驱动的补充条件。同 ABLATION_MATRIX 的方法论:
# 变换施加于【全部 split】, 每组从预训练 ckpt 重新微调 —— 模型从头看不到被遮蔽的通道。
#
# 两组:
#  (1) 充分性补全。论文声称 length / direction / timing 三通道各自充分, 但 E3 的
#      Table I 里 No-Sizes 同时抹掉幅度与方向 (方向是 signed size 的符号), 表中
#      没有任何一行能把两者分开; direction 充分性此前只有一个"修复前编码 + 51 条
#      单切分 + 100%"的弱证据。这里给三个通道各配一个 5 折 CV 的单通道条件。
#  (2) 观测点抖动。见 make_tf_jitter 的注释: timing 的判别力集中在 <100 us,
#      而那个尺度在客户端本机之外不可观测。
# --------------------------------------------------------------------------

E6_MATRIX = [
    # --- (1) 单通道充分性 (全 split 施加) ---
    dict(name="e6_suff_dir_only",   tf=["pad", "nobytes", "nointervals"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="仅方向: |包长| 恒为 1350 但保号, 字节与时间全关"),
    dict(name="e6_suff_mag_only",   tf=["absdir", "nobytes", "nointervals"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="仅包长幅度: 抹方向, 字节与时间全关"),
    dict(name="e6_suff_bytes_only", tf=["nosizes", "nointervals"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="仅报头字节: 包长与时间全关"),

    # --- (2) timing-only 在观测点抖动下 (基底 = nosizes+nobytes, 与 E3 同构) ---
    dict(name="e6_jit50us",  tf=["nosizes", "nobytes", "jitter50us"],  targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="timing-only + sigma=50us 路径抖动"),
    dict(name="e6_jit200us", tf=["nosizes", "nobytes", "jitter200us"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="timing-only + sigma=200us"),
    dict(name="e6_jit1ms",   tf=["nosizes", "nobytes", "jitter1ms"],   targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="timing-only + sigma=1ms (典型广域抖动量级)"),
    dict(name="e6_jit5ms",   tf=["nosizes", "nobytes", "jitter5ms"],   targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="timing-only + sigma=5ms"),

    # --- 全特征模型在抖动下: 回答"主结论是否依赖本机时间戳" ---
    dict(name="e6_jit1ms_full", tf=["jitter1ms"], targets="both",
         apply_to=["train", "valid", "test"], mix=False,
         note="全特征 + sigma=1ms: 主指标对观测点抖动的敏感性"),
]


def default_base_dir():
    return _ROOT


def main():
    p = argparse.ArgumentParser(description="双向管道混淆/消融数据集生成器")
    p.add_argument("--base-dir", default=default_base_dir())
    p.add_argument("--src", default="quic_vs_masque_bi_feat",
                   help="源数据集目录名 (data/processed/ 下)")
    p.add_argument("--out", default=None, help="单次生成: 输出目录名")
    p.add_argument("--transforms", default="", help=f"逗号分隔, 可选: {','.join(TRANSFORMS)}")
    p.add_argument("--targets", default="both", choices=["both", "masque_only"])
    p.add_argument("--apply-to", default="test", help="逗号分隔的 split, 或 none")
    p.add_argument("--mix-clean", action="store_true", help="变换 split 中同时保留干净版")
    p.add_argument("--all", action="store_true", help="按内置 MATRIX 一次生成全部混淆数据集")
    p.add_argument("--ablation", action="store_true", help="按 ABLATION_MATRIX 生成消融数据集")
    p.add_argument("--e6", action="store_true", help="按 E6_MATRIX 生成补充条件数据集")
    args = p.parse_args()

    processed = os.path.join(args.base_dir, "data", "processed")
    src_dir = os.path.join(processed, args.src)
    assert os.path.isdir(src_dir), f"源目录不存在: {src_dir}"

    if args.all or args.ablation or args.e6:
        matrix = ((MATRIX if args.all else [])
                  + (ABLATION_MATRIX if args.ablation else [])
                  + (E6_MATRIX if args.e6 else []))
        print(f"源数据集: {src_dir}")
        print(f"生成 {len(matrix)} 个数据集 ->  {processed}\n")
        for spec in matrix:
            generate(src_dir, os.path.join(processed, spec["name"]),
                     spec["tf"], spec["targets"], set(spec["apply_to"]), spec["mix"])
            print(f"  {'':26s} └─ {spec['note']}")
        print("\n完成。")
        return

    assert args.out, "非 --all 模式必须指定 --out"
    tf_names = [t for t in args.transforms.split(",") if t]
    for t in tf_names:
        assert t in TRANSFORMS, f"未知变换: {t} (可选 {list(TRANSFORMS)})"
    apply_to = set() if args.apply_to == "none" else set(args.apply_to.split(","))
    for s in apply_to:
        assert s in SPLITS, f"未知 split: {s}"

    generate(src_dir, os.path.join(processed, args.out),
             tf_names, args.targets, apply_to, args.mix_clean)


if __name__ == "__main__":
    main()

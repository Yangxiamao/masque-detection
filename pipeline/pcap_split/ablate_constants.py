"""[2026-08-18] E1 常量消融 —— 把三个"部署配置常量"从特征里抹掉。

背景 (见 docs/01_Project_Management/NEXT_four_experiments_handoff.md §1.2):
2026-08-18 的报头几何分析发现 Direct / MASQUE 的可分性主要落在三个**与协议结构无关
的配置常量**上 —— 最大包 1200 vs 1250、最小包 43 vs 24 (|DCID| 20 vs 0)、
是否发 Retry。三者都不是 MASQUE RFC 强制的，改 Cloudflare 的部署配置即可全部失效。
本脚本生成"看不到这些常量"的数据集，用于重训 NetMamba+ 并给出各常量的贡献。

变换 (施加于 train/valid/test 全部 split，每组单独重训一个模型)
--------------------------------------------------------------------------
C1  MTU 归一化   每条流自身最大包 -> 1292 (Direct 的满包尺度)，保号
C2  CID 归一化   按方向把该方向的最小包平移到 63，整方向等量平移 (CID 长度差
                 本就是每包一个常数偏移，故平移是物理正确的建模)
C3  去 Retry     丢掉真实的 Retry 帧 (严谨版，帧序号来自
                 experiments/analyze_header_geometry.py --dump-retry-index)
C123  三者同时   **主指标**

⚠️ C123 的施加顺序是 C3 -> C2 -> C1，不是交接文档写的 C1 -> C2 -> C3。
   理由: C2 是整方向平移，会把 C1 刚归一好的 1292 再拽下去 (Masque -21 / Direct -3)，
   于是最大包重新可分 (1271 vs 1289)，C1 白做。先平移再钳位才能让两类的
   最大包同时落在 1292、最小包同时落在 63。已在 --verify 中实测确认。

所有变换均为 **label-free**: 只用每条流自身的统计量，不读 label 字段。
所有变换均 **保号**: 正=上行 负=下行，不动方向信息 (否则等于顺带做了 No-Direction 消融)。

C3 不走 JSON 改写而是**从 pcap 重新抽取**，复用 dataset_mm_bi_common.get_chunk_feature
本身，这样字节通道 (data)、包长、到达间隔三者天然自洽 —— 若只从 JSON 删元素，
`data` 只存了前 5 个包，删掉第 3 个包后凑不出新的第 5 个包的字节。
代价是必须在有 scapy 的 WSL MAMBA 环境下跑。

    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; \
        python pipeline/pcap_split/ablate_constants.py --verify'   # 自检，不写数据
    wsl -e bash -lic 'conda activate MAMBA; cd /mnt/d/zemi/2026/dataset; \
        python pipeline/pcap_split/ablate_constants.py'            # 生成 5 个数据集
"""

import argparse
import copy
import json
import os
import shutil
import sys

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(_ROOT, "NetMambaPlus", "dataset_scripts"))

SRC_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi_feat")
FLOW_DIR = os.path.join(_ROOT, "data", "processed", "quic_vs_masque_bi")
RETRY_INDEX = os.path.join(_ROOT, "outputs", "header_geometry", "retry_index.json")
OUT_BASE = os.path.join(_ROOT, "data", "processed")
SPLITS = ("train", "valid", "test")

# 与 feat_quic_vs_masque_bi.py 严格一致
BYTE_PACKET_NUM = 5
HEADER_BYTES = 80
PAYLOAD_BYTES = 240

MTU_TARGET = 1292   # Direct 满包 (UDP 载荷 1250 + 42)
FLOOR_TARGET = 63   # Direct 侧观测到的最小包地板


# --------------------------------------------------------------------------
# 长度变换 (纯 JSON, 无外部依赖)
# --------------------------------------------------------------------------

def _parse(sizes):
    return [int(x) for x in sizes.split()]


def _fmt(vals):
    return " ".join(str(v) for v in vals)


def c1_mtu(vals):
    """把处于该流自身最大值的包统一钳到 MTU_TARGET。0 (padding) 不动。"""
    nz = [abs(v) for v in vals if v != 0]
    if not nz:
        return list(vals)
    m = max(nz)
    return [(MTU_TARGET if abs(v) == m else abs(v)) * (1 if v > 0 else -1)
            if v != 0 else 0 for v in vals]


def c2_cid(vals):
    """按方向把该方向的最小包平移到 FLOOR_TARGET，整方向等量平移。"""
    out = list(vals)
    for sign in (1, -1):
        idx = [i for i, v in enumerate(vals) if v != 0 and (v > 0) == (sign > 0)]
        if not idx:
            continue
        delta = min(abs(vals[i]) for i in idx) - FLOOR_TARGET
        for i in idx:
            out[i] = sign * (abs(vals[i]) - delta)
    return out


# --------------------------------------------------------------------------
# C3: 丢掉真实 Retry 帧后，从 pcap 重新抽取整条流的特征
# --------------------------------------------------------------------------

def _local_pcap(item):
    """JSON 里的 pcap_file 是 Windows 绝对路径，按 name + basename 重建本地路径。"""
    return os.path.join(FLOW_DIR, item["name"],
                        os.path.basename(item["pcap_file"].replace("\\", "/")))


def _reextract(item, drop_idx, debias_funcs):
    """按原管道的代码路径重抽特征；drop_idx 为要剔除的帧序号 (pcap 记录序)。"""
    import scapy.all as scapy
    from dataset_mm_bi_common import get_chunk_feature, merge_1d_array, merge_2d_array

    packets = scapy.rdpcap(_local_pcap(item))
    for fn in debias_funcs:
        packets = fn(packets)
    assert 0 not in drop_idx, "不能丢第 0 帧: 方向基准 (first_src_ip) 依赖它"
    keep = scapy.PacketList([p for i, p in enumerate(packets) if i not in drop_idx])
    bytes_list, signed, inter = get_chunk_feature(
        keep, BYTE_PACKET_NUM, HEADER_BYTES, PAYLOAD_BYTES, merge_packet_bytes=True)
    new = copy.deepcopy(item)
    new["data"] = merge_2d_array(bytes_list)
    new["signed_sizes"] = merge_1d_array(signed)
    new["intervals"] = merge_1d_array(inter)
    new["num_packet"] = len(keep)
    return new


def _debias():
    from dataset_json_common import relative_zero_ip, zero_port
    return [relative_zero_ip, zero_port]


def load_retry_index():
    with open(RETRY_INDEX, encoding="utf-8") as f:
        return json.load(f)


def no_bytes(item):
    new = copy.deepcopy(item)
    new["data"] = [" ".join(["0"] * len(d.split())) for d in new["data"]]
    return new


# --------------------------------------------------------------------------
# 数据集构建
# --------------------------------------------------------------------------

def build(kind, retry_idx_map, debias_funcs):
    """返回 {split: [item, ...]}。kind in c1/c2/c3/c123/c123_nobytes。"""
    out = {}
    for split in SPLITS:
        with open(os.path.join(SRC_DIR, f"data-{split}.json"), encoding="utf-8") as f:
            items = json.load(f)
        new_items = []
        for it in items:
            base = os.path.basename(it["pcap_file"].replace("\\", "/"))
            drop = retry_idx_map[base]["retry_idx"]

            if kind in ("c3", "c123", "c123_nobytes"):
                cur = _reextract(it, set(drop), debias_funcs) if drop else copy.deepcopy(it)
            else:
                cur = copy.deepcopy(it)

            if kind != "c3":
                vals = _parse(cur["signed_sizes"])
                if kind == "c1":
                    vals = c1_mtu(vals)
                elif kind == "c2":
                    vals = c2_cid(vals)
                else:                      # c123 / c123_nobytes: 先平移再钳位
                    vals = c1_mtu(c2_cid(vals))
                cur["signed_sizes"] = _fmt(vals)

            if kind == "c123_nobytes":
                cur = no_bytes(cur)
            new_items.append(cur)
        out[split] = new_items
        print(f"  {kind:<14} {split:<6} {len(new_items)} flows")
    return out


def save(kind, data):
    out_dir = os.path.join(OUT_BASE, f"bi_const_{kind}")
    os.makedirs(out_dir, exist_ok=True)
    for split, items in data.items():
        with open(os.path.join(out_dir, f"data-{split}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
    shutil.copy(os.path.join(SRC_DIR, "metadata.json"),
                os.path.join(out_dir, "metadata.json"))
    print(f"  -> {out_dir}")


# --------------------------------------------------------------------------
# Step 2 自检
# --------------------------------------------------------------------------

def verify(retry_idx_map, debias_funcs, n_show=3):
    with open(os.path.join(SRC_DIR, "data-train.json"), encoding="utf-8") as f:
        train = json.load(f)

    print("=" * 78)
    print("[1] 帧序号对齐: |signed_sizes[i]| 是否等于 pcap 第 i 帧的长度")
    print("=" * 78)
    bad = 0
    for it in train:
        base = os.path.basename(it["pcap_file"].replace("\\", "/"))
        rec = retry_idx_map[base]
        vals = [abs(v) for v in _parse(it["signed_sizes"])]
        if rec["n_frames"] != it["num_packet"] or rec["frame_lens"][:len(vals)] != vals:
            bad += 1
    print(f"  {len(train) - bad}/{len(train)} 条 train 流逐帧对齐, 不一致 {bad} 条")
    assert bad == 0, "帧序号未对齐, C3 的 retry_idx 不可用"

    print()
    print("=" * 78)
    print("[2] 重抽取恒等性: drop=[] 时能否逐字节复现原 JSON")
    print("=" * 78)
    for it in train[:5]:
        r = _reextract(it, set(), debias_funcs)
        same = (r["data"] == it["data"] and r["signed_sizes"] == it["signed_sizes"]
                and r["intervals"] == it["intervals"] and r["num_packet"] == it["num_packet"])
        print(f"  {'OK ' if same else 'FAIL'} {os.path.basename(it['pcap_file'])[:58]}")
        assert same, "重抽取与原管道不一致, C3 不可信"

    print()
    print("=" * 78)
    print(f"[3] 变换前后逐流对照 (各类 {n_show} 条)")
    print("=" * 78)
    for label_name in ("Direct", "Masque"):
        shown = 0
        for it in train:
            if it["name"] != label_name:
                continue
            base = os.path.basename(it["pcap_file"].replace("\\", "/"))
            drop = retry_idx_map[base]["retry_idx"]
            v0 = _parse(it["signed_sizes"])
            c3 = _reextract(it, set(drop), debias_funcs) if drop else it
            v3 = _parse(c3["signed_sizes"])
            v123 = c1_mtu(c2_cid(v3))
            print(f"\n  --- {label_name}  {base[:56]}")
            print(f"      retry_idx = {drop}")
            print(f"      orig  [:12] {v0[:12]}")
            print(f"      C1    [:12] {c1_mtu(v0)[:12]}")
            print(f"      C2    [:12] {c2_cid(v0)[:12]}")
            print(f"      C3    [:12] {v3[:12]}")
            print(f"      C123  [:12] {v123[:12]}")
            for tag, v in (("orig", v0), ("C1", c1_mtu(v0)), ("C2", c2_cid(v0)),
                           ("C3", v3), ("C123", v123)):
                nz = [abs(x) for x in v if x != 0]
                up = [abs(x) for x in v if x > 0]
                dn = [abs(x) for x in v if x < 0]
                print(f"      {tag:<5} max={max(nz):<5} min={min(nz):<4} "
                      f"min_up={min(up) if up else '-':<5} min_dn={min(dn) if dn else '-':<5} "
                      f"n={len(nz)} n_pos={len(up)} n_neg={len(dn)} zeros={len(v)-len(nz)}")
            shown += 1
            if shown >= n_show:
                break

    print()
    print("=" * 78)
    print("[4] 全 train 集分布: 变换是否真的抹掉了三个常量")
    print("=" * 78)
    import collections
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in train:
        base = os.path.basename(it["pcap_file"].replace("\\", "/"))
        drop = retry_idx_map[base]["retry_idx"]
        v0 = _parse(it["signed_sizes"])
        v3 = _parse(_reextract(it, set(drop), debias_funcs)["signed_sizes"]) if drop else v0
        for tag, v in (("orig", v0), ("C1", c1_mtu(v0)), ("C2", c2_cid(v0)),
                       ("C3", v3), ("C123", c1_mtu(c2_cid(v3)))):
            nz = [abs(x) for x in v if x != 0]
            rows[tag][it["name"]].append((max(nz), min(nz)))

    print(f"  {'变换':<6}{'类别':<8}{'最大包众数(占比)':>26}{'最小包众数(占比)':>26}")
    print("  " + "-" * 66)
    for tag in ("orig", "C1", "C2", "C3", "C123"):
        for cls in ("Direct", "Masque"):
            mx = collections.Counter(a for a, _ in rows[tag][cls])
            mn = collections.Counter(b for _, b in rows[tag][cls])
            n = len(rows[tag][cls])
            (a, ca), (b, cb) = mx.most_common(1)[0], mn.most_common(1)[0]
            print(f"  {tag:<6}{cls:<8}{f'{a} ({100*ca/n:.1f}%)':>26}"
                  f"{f'{b} ({100*cb/n:.1f}%)':>26}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只做自检, 不写数据集")
    ap.add_argument("--only", type=str, default=None, help="只建指定 kind (逗号分隔)")
    args = ap.parse_args()

    retry_idx_map = load_retry_index()
    debias_funcs = _debias()

    if args.verify:
        verify(retry_idx_map, debias_funcs)
        return

    kinds = ["c1", "c2", "c3", "c123", "c123_nobytes"]
    if args.only:
        kinds = [k.strip() for k in args.only.split(",")]
    for kind in kinds:
        print(f"\n构建 bi_const_{kind} ...")
        save(kind, build(kind, retry_idx_map, debias_funcs))


if __name__ == "__main__":
    main()

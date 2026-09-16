"""[2026-08-23] E4: 模型是在认「隧道」还是在认「对端是不是 Cloudflare」?

回答什么 (NEXT_four_experiments_handoff.md §7):
论文最容易被问到的一句话是「MASQUE 全经 Cloudflare、Direct 是多站点, 你的模型
只是在认对端」。E4 用两个纯离线设计定量削弱这个替代解释。

  E4-A  留出全部 49 条 Direct->Cloudflare 流 (train 里一条都没有), 看它们是否仍被判为 Direct。
        ★主指标 = 这批流的 Direct recall。
        配一个**同规模同流程的对照** nocf_holdout (留出 49 条随机非 CF Direct,
        CF 流全部留在 train), 用来把「Cloudflare」与「换一批测试流」分开。
        两组 test 集不同 -> **不是配对**, 用 Fisher exact, 不能用 McNemar。

  E4-B  按 Direct 侧对端 /16 做 leave-one-group-out (7 个 >=8 条流的组)。
        非对称设计: 只对 Direct 侧分组, MASQUE 侧随机 —— MASQUE 96% 的对端都是
        162.159, 对称分组会让一折拿走几乎全部 MASQUE (E3_kfold_design.md §1.3)。
        7 组里有 2 组 (104.18 / 170.114) 是 Cloudflare, 直接给出组内对比。

⚠️ 前提: 去偏用 [relative_zero_ip, zero_port], 真实 IP 已被抹成 0.0.0.0/0.0.0.1。
   **模型读不到对端 IP**。E4 检验的是「是否学到与对端强相关的站点级指纹」,
   不是「是否直接读了 IP」。

⚠️ 措辞纪律 (§7.6): 结论只能写「已定量削弱」, **不可写「交络因子已排除」**。
   Cloudflare CDN 边缘 (104.18) 与 WARP 入口 (162.159) 是不同服务、行为不同。

    python pipeline/split_data_e4.py --all        # 先造数据集
    python experiments/run_e4_cloudflare.py       # 9 次训练 + 汇总
    python experiments/run_e4_cloudflare.py --summary
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from run_kfold_e3 import wilson_ci, prf, holm  # 复用统计工具, 不复制实现

DATA_ROOT = os.path.join(_ROOT, "data", "processed", "e4_groups")
OUT_ROOT = os.path.join(_ROOT, "outputs", "e4_cloudflare")

# 超参: 与 E1/E3 (§3.4) 严格一致, 单变量只有数据集构造方式
BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 20
HEADER_BYTE_NUM = 80
PAYLOAD_BYTE_NUM = 240
STRIDE_LEN = 4
BATCH_SIZE = 16
EPOCHS = 30
SEED = 0
INTERVAL_ENCODING = "log10_decade"   # ⚠️ 必须显式传, 默认 sigmoid_log 有缺陷

CF_GROUPS = {"104.16", "104.17", "104.18", "104.19", "172.64", "170.114"}
MAJORITY_BASELINE = 270 / 504


def runs_from_manifests():
    """从 pipeline/split_data_e4.py 落盘的清单读运行列表, 保证与实际造出的数据集一致。"""
    p = os.path.join(OUT_ROOT, "e4_datasets.json")
    if not os.path.exists(p):
        sys.exit(f"找不到 {p} —— 先跑 python pipeline/split_data_e4.py --all")
    with open(p, encoding="utf-8") as f:
        mans = json.load(f)
    return [(m["mode"].replace(":", "_").replace(".", "_"), m) for m in mans]


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #

def train(name, epochs, cuda_idx=0):
    data_dir = os.path.join(DATA_ROOT, name)
    output_dir = os.path.join(OUT_ROOT, name)
    ckpt_path = os.path.join(_ROOT, "NetMambaPlus", "dataset", "checkpoints",
                             "pre-trained checkpoints", "fuse3_mamba.pth")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        num_class = len(json.load(f)["name_to_idx"])

    print(f"\n{'='*74}\n训练 E4 条件: {name}\n  data   = {data_dir}\n"
          f"  output = {output_dir}\n{'='*74}", flush=True)

    cmd = f"""
python -u "{os.path.normpath(os.path.join(_ROOT, "NetMambaPlus", "src", "fine-tune.py"))}" \\
    --num_packet {BYTE_PACKET_NUM} \\
    --num_packet_byte {HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM} \\
    --seq_len {SEQ_PACKET_NUM} \\
    --stride_size {STRIDE_LEN} \\
    --batch_size {BATCH_SIZE} \\
    --epochs {epochs} \\
    --seed {SEED} \\
    --nb_classes {num_class} \\
    --finetune "{ckpt_path}" \\
    --data_path "{data_dir}" \\
    --output_dir "{output_dir}" \\
    --log_dir "{output_dir}" \\
    --model fuse3_mamba_classifier \\
    --dataset_type byte_size_interval \\
    --size_key signed_sizes \\
    --interval_encoding {INTERVAL_ENCODING} \\
    --no_amp
"""
    env = os.environ.copy()
    if sys.platform != "win32":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, env=env,
                          cwd=os.path.join(_ROOT, "NetMambaPlus", "src", "scripts"))
    if proc.returncode != 0:
        sys.exit(f"!!! 训练失败: {name}")
    print(f"--- {name} 用时 {time.time()-t0:.1f}s", flush=True)


def dump(name):
    """逐样本预测落盘 —— E4 的主指标是「test 里 Direct 那部分的 recall」,
    光有混淆矩阵也够算, 但逐流预测能进一步看错在哪个 /16 组。"""
    d = os.path.join(DATA_ROOT, name)
    o = os.path.join(OUT_ROOT, name)
    proc = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "experiments", "eval_dump_preds.py"),
         "--data_dir", d, "--ckpt", os.path.join(o, "checkpoint-best.pth"),
         "--out", os.path.join(o, "preds.json"), "--seed", str(SEED),
         "--verify", os.path.join(o, "test_stats.json")])
    return proc.returncode == 0


def drop_ckpt(name, keep):
    p = os.path.join(OUT_ROOT, name, "checkpoint-best.pth")
    if not keep and os.path.exists(p):
        os.remove(p)   # 22.5 MB/次


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #

def fisher(a1, n1, a2, n2):
    """两个**独立**测试集上的正确率比较 (E4-A 的两组 test 集不同, 不能用 McNemar)。"""
    from scipy.stats import fisher_exact
    table = [[a1, n1 - a1], [a2, n2 - a2]]
    _odds, p = fisher_exact(table)
    return {"table": table, "p": float(p)}


def analyze(name, man, peer):
    """由 preds.json 算总体 acc / Direct recall / Masque recall, 并按对端 /16 拆错误。"""
    p = os.path.join(OUT_ROOT, name, "preds.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        preds = json.load(f)

    cm = [[0, 0], [0, 0]]
    for k, v in preds.items():
        cm[v["label"]][v["pred"]] += 1
    n = len(preds)
    correct = cm[0][0] + cm[1][1]

    d_n, d_ok = cm[0][0] + cm[0][1], cm[0][0]
    m_n, m_ok = cm[1][0] + cm[1][1], cm[1][1]
    wp, wr, wf = prf(cm)

    # Direct 侧按对端 /16 拆
    by16 = collections.defaultdict(lambda: [0, 0])   # g -> [n, ok]
    for k, v in preds.items():
        if v["label"] != 0:
            continue
        g = peer[k]["peer16"]
        by16[g][0] += 1
        by16[g][1] += (v["pred"] == 0)

    return {
        "name": name, "mode": man["mode"], "desc": man["desc"],
        "n_train": man["n_train"], "n_valid": man["n_valid"], "n_test": n,
        "acc": correct / n, "acc_ci95": list(wilson_ci(correct, n)),
        "cm": cm, "weighted_f1": wf, "weighted_pre": wp, "weighted_rec": wr,
        "direct_n": d_n, "direct_correct": d_ok, "direct_recall": d_ok / d_n if d_n else None,
        "direct_recall_ci95": list(wilson_ci(d_ok, d_n)) if d_n else None,
        "masque_n": m_n, "masque_correct": m_ok, "masque_recall": m_ok / m_n if m_n else None,
        "direct_by16": {g: {"n": v[0], "correct": v[1], "recall": v[1] / v[0]}
                        for g, v in sorted(by16.items(), key=lambda x: -x[1][0])},
    }


def e3_reference():
    """E3 的 full 条件 OOF 混淆矩阵 -> 常规随机切分下的 Direct recall, 作参照。"""
    p = os.path.join(_ROOT, "outputs", "e3_kfold", "summary_e3_kfold.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    c = s.get("conditions", {}).get("full")
    if not c:
        return None
    cm = c["cm"]
    return {"acc": c["acc"], "direct_n": cm[0][0] + cm[0][1], "direct_correct": cm[0][0],
            "direct_recall": cm[0][0] / (cm[0][0] + cm[0][1])}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑指定条件 (逗号分隔的目录名)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--summary", action="store_true", help="跳过训练, 只汇总")
    ap.add_argument("--keep-ckpt", dest="keep_ckpt", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    runs = runs_from_manifests()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs if r[0] in want]

    with open(os.path.join(OUT_ROOT, "peer_groups.json"), encoding="utf-8") as f:
        peer = json.load(f)

    # ---------------- 训练 ----------------
    if not args.summary:
        t_all = time.time()
        for i, (name, _man) in enumerate(runs, 1):
            o = os.path.join(OUT_ROOT, name)
            have = os.path.exists(os.path.join(o, "test_stats.json"))
            have_preds = os.path.exists(os.path.join(o, "preds.json"))
            if have and have_preds and not args.force:
                print(f"[{i}/{len(runs)}] 跳过已完成: {name}")
                continue
            if not (have and os.path.exists(os.path.join(o, "checkpoint-best.pth"))) or args.force:
                print(f"[{i}/{len(runs)}] 训练 {name}")
                train(name, args.epochs)
            if not dump(name):
                print(f"!!! dump 失败: {name}")
            drop_ckpt(name, keep=args.keep_ckpt)
        print(f"\n全部训练用时 {time.time()-t_all:.1f}s")

    # ---------------- 汇总 ----------------
    res = {}
    for name, man in runs:
        r = analyze(name, man, peer)
        if r is None:
            print(f"(跳过未完成: {name})")
            continue
        res[name] = r

    ref = e3_reference()

    # E4-A: cf_holdout vs nocf_holdout (独立测试集 -> Fisher exact)
    tests = []
    if "cf_holdout" in res and "nocf_holdout" in res:
        a, b = res["cf_holdout"], res["nocf_holdout"]
        t = fisher(a["direct_correct"], a["direct_n"], b["direct_correct"], b["direct_n"])
        t.update({"what": "E4-A: Direct recall, CF 留出 vs 非 CF 留出对照",
                  "A": "cf_holdout", "B": "nocf_holdout",
                  "recall_A": a["direct_recall"], "recall_B": b["direct_recall"]})
        tests.append(t)

    # E4-B: CF 组 vs 非 CF 组的 leave-one-group-out Direct recall (合并计数)
    go = {k: v for k, v in res.items() if v["mode"].startswith("groupout:")}
    if go:
        cf_n = cf_ok = nc_n = nc_ok = 0
        for v in go.values():
            g = v["mode"].split(":", 1)[1]
            if g in CF_GROUPS:
                cf_n += v["direct_n"]; cf_ok += v["direct_correct"]
            else:
                nc_n += v["direct_n"]; nc_ok += v["direct_correct"]
        if cf_n and nc_n:
            t = fisher(cf_ok, cf_n, nc_ok, nc_n)
            t.update({"what": "E4-B: 留出组的 Direct recall, Cloudflare 组 vs 非 Cloudflare 组",
                      "A": "CF groups", "B": "non-CF groups",
                      "recall_A": cf_ok / cf_n, "recall_B": nc_ok / nc_n})
            tests.append(t)
        res_pool = {"cf_n": cf_n, "cf_correct": cf_ok, "nc_n": nc_n, "nc_correct": nc_ok,
                    "pooled_n": cf_n + nc_n, "pooled_correct": cf_ok + nc_ok,
                    "pooled_recall": (cf_ok + nc_ok) / (cf_n + nc_n),
                    "pooled_ci95": list(wilson_ci(cf_ok + nc_ok, cf_n + nc_n))}
    else:
        res_pool = None

    for t, adj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = adj

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {
            "num_packet": BYTE_PACKET_NUM, "num_packet_byte": HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM,
            "seq_len": SEQ_PACKET_NUM, "stride_size": STRIDE_LEN, "batch_size": BATCH_SIZE,
            "epochs": args.epochs, "seed": SEED, "model": "fuse3_mamba_classifier",
            "dataset_type": "byte_size_interval", "size_key": "signed_sizes",
            "interval_encoding": INTERVAL_ENCODING,
        },
        "conditions": res,
        "groupout_pooled": res_pool,
        "tests": tests,
        "e3_full_reference": ref,
        "majority_baseline": MAJORITY_BASELINE,
        "caveats": [
            "去偏 [relative_zero_ip, zero_port] 已抹掉真实 IP -> 模型读不到对端 IP 本身",
            "只能写『交络因子已定量削弱』, 不可写『已排除』(handoff §7.6)",
            "CDN 边缘 104.18 与 WARP 入口 162.159 同属 Cloudflare 但是不同服务、行为不同",
            "MASQUE 侧 test 从 259 条 162.159 流里随机抽, 11 条标签噪声流一律留在 train",
            "单次采集 2026-06-16, 单一部署 WARP 2026.6.880.0",
        ],
    }
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = os.path.join(OUT_ROOT, "summary_e4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------------- 打印 ----------------
    W = 104
    print(f"\n\n{'='*W}")
    print("E4-A  留出 Direct->Cloudflare (主指标 = 留出的 Direct 流仍被判为 Direct 的比例)")
    print(f"{'='*W}")
    print(f"{'条件':<30}{'test':>6}{'总 Acc':>10}{'Direct recall':>16}{'95% CI':>20}"
          f"{'Masque recall':>16}")
    print("-" * W)
    for k in ("cf_holdout", "nocf_holdout"):
        if k not in res:
            continue
        r = res[k]
        ci = f"[{r['direct_recall_ci95'][0]*100:.1f}, {r['direct_recall_ci95'][1]*100:.1f}]"
        print(f"{r['desc'][:28]:<30}{r['n_test']:>6}{r['acc']*100:>9.2f}%"
              f"{r['direct_correct']}/{r['direct_n']} = {r['direct_recall']*100:>6.2f}%"
              f"{ci:>20}{r['masque_correct']}/{r['masque_n']} = {r['masque_recall']*100:>6.2f}%")
    if ref:
        print("-" * W)
        print(f"{'[参照] E3 full 条件 5 折 OOF':<30}{'504':>6}{ref['acc']*100:>9.2f}%"
              f"{ref['direct_correct']}/{ref['direct_n']} = {ref['direct_recall']*100:>6.2f}%")

    if go:
        print(f"\n{'='*W}")
        print("E4-B  leave-one-group-out (按 Direct 侧对端 /16; MASQUE 侧随机, 非对称设计)")
        print(f"{'='*W}")
        print(f"{'留出的 /16':<14}{'CF':>4}{'Direct n':>10}{'Direct recall':>16}{'95% CI':>20}"
              f"{'总 Acc':>10}{'Masque recall':>16}")
        print("-" * W)
        for name, r in sorted(go.items(), key=lambda x: -x[1]["direct_n"]):
            g = r["mode"].split(":", 1)[1]
            ci = f"[{r['direct_recall_ci95'][0]*100:.1f}, {r['direct_recall_ci95'][1]*100:.1f}]"
            print(f"{g:<14}{'CF' if g in CF_GROUPS else '':>4}{r['direct_n']:>10}"
                  f"{r['direct_correct']}/{r['direct_n']} = {r['direct_recall']*100:>6.2f}%"
                  f"{ci:>20}{r['acc']*100:>9.2f}%"
                  f"{r['masque_correct']}/{r['masque_n']} = {r['masque_recall']*100:>6.2f}%")
        if res_pool:
            print("-" * W)
            ci = res_pool["pooled_ci95"]
            print(f"{'合并 7 组':<18}{res_pool['pooled_n']:>10}"
                  f"{res_pool['pooled_correct']}/{res_pool['pooled_n']} = "
                  f"{res_pool['pooled_recall']*100:>6.2f}%"
                  f"{f'[{ci[0]*100:.1f}, {ci[1]*100:.1f}]':>20}")
            print(f"{'  其中 Cloudflare 组':<20}{res_pool['cf_n']:>8}"
                  f"{res_pool['cf_correct']}/{res_pool['cf_n']} = "
                  f"{100*res_pool['cf_correct']/res_pool['cf_n']:>6.2f}%")
            print(f"{'  其中非 Cloudflare 组':<19}{res_pool['nc_n']:>8}"
                  f"{res_pool['nc_correct']}/{res_pool['nc_n']} = "
                  f"{100*res_pool['nc_correct']/res_pool['nc_n']:>6.2f}%")

    if tests:
        print(f"\n{'='*W}")
        print("显著性检验 (两组 test 集不同 -> 非配对, 用 Fisher exact; Holm 校正)")
        print(f"{'='*W}")
        for t in tests:
            verdict = "有显著差异" if t["p_holm"] < 0.05 else "无显著差异"
            print(f"  {t['what']}")
            print(f"    {t['A']} {t['recall_A']*100:.2f}%  vs  {t['B']} {t['recall_B']*100:.2f}%"
                  f"   2x2={t['table']}   p={t['p']:.4f}  p(Holm)={t['p_holm']:.4f}  -> {verdict}")
        print("  注: 不显著 ≠ 等价 —— 本实验样本量小, 检出力有限。")

    print(f"\n措辞纪律 (handoff §7.6): 只能写「交络因子已定量削弱」, 不可写「已排除」。")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()

"""[2026-10-12 ICOIN r2 #3] 经典模型 / 分析脚本的改标开关。

环境变量 MASQUE_RELABEL=1 时:
  relabel_items(items)  把 11 条噪声流 (outputs/e3_kfold/error_analysis_e3.json 的
                        label_noise_flows, 判据 = 对端不在 WARP 162.159 前缀) 的
                        label 改为 0、name 改为 Direct; 其余逐字不变
  is_noise(basename)    供按文件名分类的脚本 (如 analyze_header_geometry) 使用
  out_dir(path)         输出目录加 "_relabel" 后缀, 绝不覆盖原标签结果
未设置时三者都是恒等变换, 原脚本行为不变。
"""

import json
import os

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELABEL = os.environ.get("MASQUE_RELABEL") == "1"
_ERR = os.path.join(_ROOT, "outputs", "e3_kfold", "error_analysis_e3.json")
_noise = None


def noise_set():
    global _noise
    if _noise is None:
        with open(_ERR, encoding="utf-8") as f:
            _noise = set(json.load(f)["label_noise_flows"])
        assert len(_noise) == 11, len(_noise)
    return _noise


def _base(p):
    return os.path.basename(p.replace("\\", "/"))


def is_noise(basename):
    return RELABEL and _base(basename) in noise_set()


def relabel_items(items):
    if not RELABEL:
        return items
    n = 0
    for it in items:
        if _base(it["pcap_file"]) in noise_set():
            assert it["label"] == 1, it["pcap_file"]
            it["label"] = 0
            it["name"] = "Direct"
            n += 1
    print(f"[relabel] {n} 条记录改为 Direct")
    return items


def out_dir(path):
    return path.rstrip("/\\") + "_relabel" if RELABEL else path

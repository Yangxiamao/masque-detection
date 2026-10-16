"""[2026-10-12 ICOIN r2 §7.5] NetMamba+ 选模规则开关: 改标版全部实验按 acc_loss 重跑。

见 docs/01_Planning/Handoff_Relabel_SelRule_Rerun_2026-10-12.md §4-§5。

环境变量 MASQUE_SELRULE=acc_loss 时:
  train_entry()   训练入口换成 experiments/finetune_selrule.py (fine-tune.py 的透传包装:
                  训练轨迹、RNG、checkpoint-best.pth、test_stats.json 都不变,
                  另存 ckpt-acc_loss.pth / ckpt-min_loss.pth / ckpt-last.pth 与 selrule.json)
  ckpt_name()     主结果用的 checkpoint = ckpt-acc_loss.pth
  out_dir(path)   输出目录加 "_sel" 后缀, 绝不覆盖原规则结果
  all_ckpts()     一次训练产生的全部 checkpoint (删除用)
  dump_ckpts()    dump 需要的 checkpoint (判断"训完未 dump"用)
  pred_files()    一折完成的标志文件
  dump_pair()     先 dump 原规则 ckpt 到 preds_acc_first.json 并与 test_stats.json 核对,
                  再 dump ckpt-acc_loss.pth 到 preds.json (主结果)
未设置时: fine-tune.py / checkpoint-best.pth / 原路径 / [checkpoint-best.pth] / [preds.json],
各脚本行为逐字不变。

规则在看结果之前已固定为 acc_loss: 验证 acc 最高, 打平时取 valid loss 最低。
其他取值直接报错 —— min_loss / last 只在 run_selrule_diag.py 里作补充, 不得换用。
"""

import json
import os
import shutil

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RULE = os.environ.get("MASQUE_SELRULE") or None
if _RULE not in (None, "acc_loss"):
    raise SystemExit(f"MASQUE_SELRULE={_RULE!r}: 只允许 acc_loss (预先指定, 不得事后更换)")
SELRULE = _RULE == "acc_loss"
RULE_DESC = "acc_loss: valid acc 最高, 打平取 valid loss 最低 (experiments/finetune_selrule.py)"

ACC_FIRST = "checkpoint-best.pth"   # fine-tune.py 原规则: valid acc 第一次达到最高
ACC_LOSS = "ckpt-acc_loss.pth"
_SEL_CKPTS = [ACC_FIRST, ACC_LOSS, "ckpt-min_loss.pth", "ckpt-last.pth"]


def train_entry():
    if SELRULE:
        return os.path.join(_ROOT, "experiments", "finetune_selrule.py")
    return os.path.join(_ROOT, "NetMambaPlus", "src", "fine-tune.py")


def ckpt_name():
    return ACC_LOSS if SELRULE else ACC_FIRST


def out_dir(path):
    return path.rstrip("/\\") + "_sel" if SELRULE else path


def all_ckpts():
    return list(_SEL_CKPTS) if SELRULE else [ACC_FIRST]


def dump_ckpts():
    return [ACC_FIRST, ACC_LOSS] if SELRULE else [ACC_FIRST]


def pred_files():
    return ["preds.json", "preds_acc_first.json"] if SELRULE else ["preds.json"]


def clear_run_dir(path):
    """fine-tune.py 的 log.txt 是追加写入, 重训前清空。只允许清 *_sel 目录下的东西。"""
    assert SELRULE and "_sel" in os.path.abspath(path).replace("\\", "/"), path
    if os.path.isdir(path):
        shutil.rmtree(path)


def dump_pair(data_dir, run_dir, seed=0, primary_out="preds.json"):
    """sel 模式的逐样本 dump (与 run_selrule_diag.dump_fold 同法, 进程内调用)。

    1) checkpoint-best.pth -> preds_acc_first.json, 核对 acc == test_stats.json
       (test_stats.json 是 fine-tune.py 用原规则 ckpt 写的, 所以只能拿它核对这一份)
    2) ckpt-acc_loss.pth -> primary_out (主结果); primary_out=None 时跳过
    失败打印 '!!!' 并返回 False, 不抛异常 (与原 dump_fold 的退路一致)。
    """
    from eval_dump_preds import dump
    try:
        acc = dump(data_dir, os.path.join(run_dir, ACC_FIRST),
                   os.path.join(run_dir, "preds_acc_first.json"), seed=seed)
        with open(os.path.join(run_dir, "test_stats.json"), encoding="utf-8") as f:
            ref = json.load(f)["acc"]
        if abs(acc - ref) > 1e-9:
            print(f"!!! verify 不一致: acc_first dump {acc:.10f} vs test_stats {ref:.10f}  ({run_dir})",
                  flush=True)
            return False
        print(f"核对通过: acc_first dump == test_stats == {ref*100:.4f}%", flush=True)
        if primary_out:
            dump(data_dir, os.path.join(run_dir, ACC_LOSS), os.path.join(run_dir, primary_out), seed=seed)
    except (Exception, SystemExit) as e:   # dump() 用 SystemExit 报顺序/标签错位
        print(f"!!! dump 失败 {run_dir}: {e!r}", flush=True)
        return False
    return True

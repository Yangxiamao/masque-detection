"""[2026-10-12 ICOIN r2 §7.5] 选模规则诊断: fine-tune.py 的透传包装器。

fine-tune.py 在验证集准确率"第一次"达到最高的 epoch 保存 checkpoint-best.pth (严格 >)。
改标后验证集 (每折 41 条) 在 epoch 2-3 就到 100%, 之后 valid loss 还在从 ~0.33
降到 ~0.04, 怀疑因此选中了欠训练模型 (进展说明 ICOIN_R2_Progress_2026-10-12.md §7.5)。

⚠️ 不改 NetMambaPlus/ 的任何文件 (submodule)。做法: 把 fine-tune.py 当模块载入,
把它的模块全局 evaluate 换成一个透传包装 —— 每次验证照原样返回统计量, 只是额外按
其他规则另存 checkpoint。训练轨迹、RNG 消耗、checkpoint-best.pth 都与直接运行
fine-tune.py 完全相同, 所以原规则的结果应能逐条复现 (run_selrule_diag.py 会核对)。

--output_dir 下的 checkpoint:
  checkpoint-best.pth  原规则 acc_first: valid acc 第一次达到最高 (fine-tune.py 自己保存)
  ckpt-acc_loss.pth    valid acc 最高, 打平时取 valid loss 最低   <- 预先指定的主规则
  ckpt-min_loss.pth    valid loss 最低 (补充)
  ckpt-last.pth        最后一个 epoch, 不做选择 (补充)
以及 selrule.json: 逐 epoch 的 valid acc / loss 与各规则选中的 epoch。

命令行参数与 fine-tune.py 完全相同。
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

_ROOT = os.environ.get("MASQUE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "NetMambaPlus", "src")
sys.path.insert(0, _SRC)

_spec = importlib.util.spec_from_file_location("netmamba_finetune", os.path.join(_SRC, "fine-tune.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

import torch  # noqa: E402


def make_evaluate(args):
    orig_evaluate = ft.evaluate
    n_train_epochs = args.epochs - args.start_epoch
    state = {"calls": 0, "hist": [], "acc_loss": None, "min_loss": None}

    def save(model, epoch, name):
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(args.output_dir, f"ckpt-{name}.pth"))

    def evaluate(data_loader, model, device, a):
        stats = orig_evaluate(data_loader, model, device, a)
        state["calls"] += 1
        if state["calls"] > n_train_epochs:
            return stats   # 训练结束后对 data-test 的那次评估: 原样透传
        epoch = args.start_epoch + state["calls"] - 1
        acc, loss = float(stats["acc"]), float(stats["loss"])
        state["hist"].append({"epoch": epoch, "valid_acc": acc, "valid_loss": loss})

        b = state["acc_loss"]
        if b is None or acc > b["acc"] or (acc == b["acc"] and loss < b["loss"]):
            state["acc_loss"] = {"epoch": epoch, "acc": acc, "loss": loss}
            save(model, epoch, "acc_loss")
        m = state["min_loss"]
        if m is None or loss < m["loss"]:
            state["min_loss"] = {"epoch": epoch, "acc": acc, "loss": loss}
            save(model, epoch, "min_loss")
        if state["calls"] == n_train_epochs:
            save(model, epoch, "last")
        return stats

    evaluate.state = state
    return evaluate


def selected_epochs(state):
    hist = state["hist"]
    best = max(h["valid_acc"] for h in hist)
    first = next(h["epoch"] for h in hist if h["valid_acc"] == best)
    return {"acc_first": first,
            "acc_loss": state["acc_loss"]["epoch"],
            "min_loss": state["min_loss"]["epoch"],
            "last": hist[-1]["epoch"]}


if __name__ == "__main__":
    args = ft.get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ev = make_evaluate(args)
    ft.evaluate = ev
    ft.main(args)
    with open(os.path.join(args.output_dir, "selrule.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_epoch": selected_epochs(ev.state), "history": ev.state["hist"]}, f, indent=2)

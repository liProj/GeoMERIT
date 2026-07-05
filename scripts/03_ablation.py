from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--features_config", default="configs/features.yaml")
    parser.add_argument("--penalty", default="configs/penalty_matrix.csv")
    parser.add_argument("--out", default="runs/ablation")
    parser.add_argument("--baselines", default="rfe_xgb_repro,xgb_bare,lgb_bare,cat_bare")
    parser.add_argument("--toggles", default="no_strat,no_bayes_decode,no_viterbi,no_ensemble,no_coarse2fine,no_tail_expert,no_logit_adjust,no_classbalanced_w,argmax_only,tau_sweep")
    parser.add_argument("--folds", type=int, default=5, help="Folds for ablation speed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = _resolve(args.out, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(_resolve(args.config, project_root).read_text(encoding="utf-8"))

    results = []

    # ---------- Baselines ----------
    for baseline in args.baselines.split(","):
        baseline = baseline.strip()
        if not baseline:
            continue
        print(f"\n{'='*60}\nBaseline: {baseline}\n{'='*60}")
        exp_dir = out_dir / f"baseline_{baseline}"
        cfg = copy.deepcopy(base_config)
        components = "flat"
        decode_order = "bayes_only"
        tau = 0.0
        beta = 0.0

        if baseline == "rfe_xgb_repro":
            cfg["models"]["lightgbm"]["enabled"] = False
            cfg["models"]["catboost"]["enabled"] = False
            cfg["models"]["xgboost"]["enabled"] = True
            cfg["weights"]["rho"] = 0.0  # disable effective number; fall back to uniform-ish
            cfg["weights"]["boundary_weight"] = 1.0
            cfg["weights"]["interior_weight"] = 1.0
        elif baseline == "xgb_bare":
            cfg["models"]["lightgbm"]["enabled"] = False
            cfg["models"]["catboost"]["enabled"] = False
            cfg["models"]["xgboost"]["enabled"] = True
        elif baseline == "lgb_bare":
            cfg["models"]["lightgbm"]["enabled"] = True
            cfg["models"]["catboost"]["enabled"] = False
            cfg["models"]["xgboost"]["enabled"] = False
        elif baseline == "cat_bare":
            cfg["models"]["lightgbm"]["enabled"] = False
            cfg["models"]["catboost"]["enabled"] = True
            cfg["models"]["xgboost"]["enabled"] = False
        else:
            print(f"  Unknown baseline {baseline}, skipping.")
            continue

        report = _run_experiment(args, project_root, exp_dir, cfg, components, decode_order, tau, beta)
        if report:
            results.append({"experiment": baseline, **report})

    # ---------- Full GeoMERIT ----------
    print(f"\n{'='*60}\nFull GeoMERIT\n{'='*60}")
    exp_dir = out_dir / "full_geomerit"
    report = _run_experiment(
        args, project_root, exp_dir, base_config,
        components="flat,coarse2fine,tail_experts",
        decode_order="bayes_then_viterbi",
        tau=None, beta=None,
    )
    if report:
        results.append({"experiment": "full_geomerit", **report})

    # ---------- Toggles ----------
    for toggle in args.toggles.split(","):
        toggle = toggle.strip()
        if not toggle:
            continue
        print(f"\n{'='*60}\nToggle: {toggle}\n{'='*60}")
        exp_dir = out_dir / f"toggle_{toggle}"
        cfg = copy.deepcopy(base_config)
        components = "flat,coarse2fine,tail_experts"
        decode_order = "bayes_then_viterbi"
        tau = None
        beta = None

        if toggle == "no_strat":
            # Disable stratigraphy in feature engineering => rebuild dataset with modified features config
            # For speed, we just skip this in ablation by noting it; full implementation would rebuild.
            print("  NOTE: no_strat requires rebuilding features. Skipping in quick ablation.")
            continue
        elif toggle == "no_bayes_decode":
            decode_order = "viterbi_then_bayes"
        elif toggle == "no_viterbi":
            beta = 0.0
        elif toggle == "no_ensemble":
            # Keep only LightGBM
            cfg["models"]["xgboost"]["enabled"] = False
            cfg["models"]["catboost"]["enabled"] = False
        elif toggle == "no_coarse2fine":
            components = "flat,tail_experts"
        elif toggle == "no_tail_expert":
            components = "flat,coarse2fine"
        elif toggle == "no_logit_adjust":
            tau = 0.0
        elif toggle == "no_classbalanced_w":
            cfg["weights"]["rho"] = 0.0
            cfg["weights"]["boundary_weight"] = 1.0
            cfg["weights"]["interior_weight"] = 1.0
        elif toggle == "argmax_only":
            decode_order = "bayes_only"
            tau = 0.0
            beta = 0.0
        elif toggle == "tau_sweep":
            # Special: just run decode with full model but sweep tau
            _tau_sweep(args, project_root, exp_dir, base_config)
            continue
        else:
            print(f"  Unknown toggle {toggle}, skipping.")
            continue

        report = _run_experiment(args, project_root, exp_dir, cfg, components, decode_order, tau, beta)
        if report:
            results.append({"experiment": toggle, **report})

    # ---------- Summary ----------
    if results:
        df = pd.DataFrame(results)
        summary_path = out_dir / "ablation_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}\nAblation Summary\n{'='*60}")
        print(df.to_string(index=False))
        print(f"\nWrote {summary_path}")


def _run_experiment(args, project_root, exp_dir, cfg, components, decode_order, tau, beta):
    train_dir = exp_dir / "train"
    decode_dir = exp_dir / "decode"
    train_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)

    # Write temp config
    temp_cfg_path = exp_dir / "model_temp.yaml"
    with temp_cfg_path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)

    # Train
    cmd = [
        sys.executable, str(project_root / "scripts" / "01_train.py"),
        "--table", args.table,
        "--split", "groupkfold",
        "--folds", str(args.folds),
        "--config", str(temp_cfg_path),
        "--penalty", args.penalty,
        "--out", str(train_dir),
        "--components", components,
    ]
    print("  Train:", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=False, text=True)
    if ret.returncode != 0:
        print("  Training failed.")
        return None

    # Decode
    cmd = [
        sys.executable, str(project_root / "scripts" / "02_predict_decode.py"),
        "--run", str(train_dir),
        "--eval", "oof",
        "--penalty", args.penalty,
        "--config", str(temp_cfg_path),
        "--decode_order", decode_order,
        "--out", str(decode_dir / "decode_report.json"),
    ]
    if tau is not None:
        cmd += ["--tau", str(tau)]
    if beta is not None:
        cmd += ["--beta", str(beta)]
    print("  Decode:", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=False, text=True)
    if ret.returncode != 0:
        print("  Decoding failed.")
        return None

    report_path = decode_dir / "decode_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Flatten
    flat = {
        "weighted_f1": report.get("weighted_f1"),
        "macro_f1": report.get("macro_f1"),
        "boundary_f1": report.get("boundary_f1"),
        "penalty": report.get("penalty"),
        "tail_mean_f1": report.get("tail_mean_f1"),
        "tau": report.get("tau"),
        "beta": report.get("beta"),
        "decode_order": report.get("decode_order"),
    }
    return flat


def _tau_sweep(args, project_root, exp_dir, cfg):
    """Run full model once, then decode with multiple tau values."""
    train_dir = exp_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg_path = exp_dir / "model_temp.yaml"
    with temp_cfg_path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)

    cmd = [
        sys.executable, str(project_root / "scripts" / "01_train.py"),
        "--table", args.table,
        "--split", "groupkfold",
        "--folds", str(args.folds),
        "--config", str(temp_cfg_path),
        "--penalty", args.penalty,
        "--out", str(train_dir),
        "--components", "flat,coarse2fine,tail_experts",
    ]
    print("  Train:", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=False, text=True)
    if ret.returncode != 0:
        print("  Training failed.")
        return

    results = []
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        decode_dir = exp_dir / f"decode_tau{tau}"
        decode_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(project_root / "scripts" / "02_predict_decode.py"),
            "--run", str(train_dir),
            "--eval", "oof",
            "--penalty", args.penalty,
            "--config", str(temp_cfg_path),
            "--tau", str(tau),
            "--beta", "0.0",
            "--out", str(decode_dir / "decode_report.json"),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode == 0:
            report = json.loads((decode_dir / "decode_report.json").read_text(encoding="utf-8"))
            results.append({
                "tau": tau,
                "weighted_f1": report.get("weighted_f1"),
                "macro_f1": report.get("macro_f1"),
                "boundary_f1": report.get("boundary_f1"),
                "penalty": report.get("penalty"),
                "tail_mean_f1": report.get("tail_mean_f1"),
            })
    if results:
        df = pd.DataFrame(results)
        summary = exp_dir / "tau_sweep.csv"
        df.to_csv(summary, index=False)
        print("  Tau sweep results:")
        print(df.to_string(index=False))
        print(f"  Wrote {summary}")


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()

"""CLI エントリポイント.

最終課題(Advanced).ipynb のセクションをサブコマンドに対応させた:

  check      : 実行環境チェック (Section 1)
  setup      : apt + LeRobot + LIBERO-plus セットアップ (Sections 2-5)
  data       : Spatial 10 task x 5 ep の学習データ選抜 (Section 7)
  download   : 事前学習済み SmolVLA の DL (Section 8.1)
  train      : lerobot-train で LoRA 追加学習 (Section 8.2)
  merge      : LoRA マージ + 対照ベースライン作成 (Sections 8.3-8.4)
  eval       : 追加学習前後の Spatial 評価と比較表 (Sections 8.5-8.6)
  package    : マージ済みモデルの zip 化 (Section 8.7)
  video      : rollout 動画の録画 (Section 8.8)
  adv-eval   : LIBERO-plus 4 suite 広域評価 (Section 9)
  submit     : 提出用 submission.json 生成 (Advanced ルート)
  all        : setup -> ... -> adv-eval を順に実行

Basic ルート (PyTorch 自作 SmolVLA / 最終課題(Basic).ipynb 相当):

  basic-data  : 学習データ選抜 + Dataset 読み込みの確認
  basic-train : 自作 SmolVLA (SmolVLM2 凍結 + Flow Matching) を学習
  basic-eval  : LiberoEnv で rollout 評価 + mp4 出力
  basic-submit: Basic ルートの submission.json 生成
  basic-all   : basic-train -> basic-eval -> basic-submit を順に実行

各コマンドは notebook と同じ前提 (GPU / Python 3.12) で動く。
WORKDIR 等は環境変数 SMOLVLA_WORKDIR / SMOLVLA_CACHE_DIR で変更できる。
GPU の選択は --gpu N (サブコマンドの前) or 環境変数 CUDA_VISIBLE_DEVICES。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
from pathlib import Path

from . import config


def _cmd_check(args: argparse.Namespace) -> None:
    config.check_runtime(require_gpu=not args.no_gpu)
    print(f"WORKDIR   : {config.WORKDIR}")
    print(f"CACHE_DIR : {config.CACHE_DIR}")
    print(f"mixed precision: {config.mixed_precision()}")


def _cmd_setup(args: argparse.Namespace) -> None:
    from .setup_lerobot import install_lerobot
    from .setup_libero import setup_libero_plus
    from .setup_system import install_system_packages

    if not args.skip_apt:
        install_system_packages()
    install_lerobot()
    setup_libero_plus()


def _cmd_data(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    from .data_select import select_episode_indices

    select_episode_indices()


def _cmd_download(args: argparse.Namespace) -> None:
    from .train import download_base_model

    download_base_model()


def _cmd_train(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    from .data_select import select_episode_indices
    from .train import download_base_model, train_lora

    episode_indices = select_episode_indices()
    base_model_local = download_base_model()
    train_lora(episode_indices, base_model_local)


def _cmd_merge(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    from .train import download_base_model
    from .merge import build_baseline, merge_lora

    base_model_local = download_base_model()
    merge_lora(base_model_local)
    build_baseline(base_model_local)


def _cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import compare_spatial, run_evaluation

    base_info = run_evaluation(
        config.BASELINE_MODEL_DIR, config.BASE_EVAL_DIR, "Base model"
    )
    finetuned_info = run_evaluation(
        config.MERGED_MODEL_DIR, config.FINETUNED_EVAL_DIR, "Spatial LoRA"
    )
    compare_spatial(base_info, finetuned_info)


def _cmd_package(args: argparse.Namespace) -> None:
    from .package import package_merged_model

    package_merged_model()


def _cmd_video(args: argparse.Namespace) -> None:
    from .videos import record_rollout_videos

    record_rollout_videos()


def _cmd_adv_eval(args: argparse.Namespace) -> None:
    from .evaluate import compare_advanced, run_advanced_eval
    from .preflight import preflight

    preflight()

    print("=== Advanced eval: baseline ===")
    adv_base = run_advanced_eval(
        config.BASELINE_MODEL_DIR,
        config.ADVANCED_OUT_DIR / "baseline",
        "Base",
    )

    print("\n=== Advanced eval: fine-tuned ===")
    adv_finetuned = run_advanced_eval(
        config.MERGED_MODEL_DIR,
        config.ADVANCED_OUT_DIR / "finetuned",
        "LoRA",
    )

    compare_advanced(adv_base, adv_finetuned)


def _cmd_submit(args: argparse.Namespace) -> None:
    from .submission import write_submission_advanced

    write_submission_advanced(methods_notes=args.notes)


def _cmd_basic_data(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    from .basic_data import load_train_dataset
    from .data_select import select_episode_indices

    episode_indices = select_episode_indices()
    load_train_dataset(episode_indices, config.PY_CHUNK_SIZE)


def _build_basic_smolvla():
    """basic-train / basic-eval 共通: VLM + 自作 SmolVLA を構築する."""
    from .basic_model import build_smolvla, build_vlm_encoder

    vlm_encoder = build_vlm_encoder()
    smolvla = build_smolvla(vlm_encoder)

    trainable = sum(
        p.numel() for p in smolvla.parameters() if p.requires_grad
    )
    frozen = sum(
        p.numel() for p in smolvla.parameters() if not p.requires_grad
    )
    print(
        f"SmolVLA — trainable: {trainable / 1e6:.2f}M  "
        f"frozen: {frozen / 1e6:.2f}M"
    )
    return smolvla


def _cmd_basic_train(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    config.seed_everything(config.SEED)
    from .basic_data import (
        compute_action_stats,
        compute_state_stats,
        load_train_dataset,
        make_normalizers,
    )
    from .basic_train import train_pytorch_smolvla
    from .data_select import select_train_validation_episode_indices
    from .experiment import create_experiment_run
    from .wandb_logger import init_wandb

    batch_size = args.batch_size or config.TRAIN_BATCH_SIZE_PY

    train_episode_indices, validation_episode_indices = (
        select_train_validation_episode_indices(
            train_episodes_per_task=args.train_episodes_per_task,
            validation_episodes_per_task=args.validation_episodes_per_task,
            seed=args.validation_seed,
        )
    )
    dataset = load_train_dataset(train_episode_indices, config.PY_CHUNK_SIZE)
    validation_dataset = load_train_dataset(
        validation_episode_indices, config.PY_CHUNK_SIZE
    )
    if args.epochs is not None:
        effective_batch = batch_size * args.gradient_accumulation_steps
        train_steps = math.ceil(len(dataset) * args.epochs / effective_batch)
    else:
        train_steps = args.steps or config.TRAIN_STEPS_PY
    action_mean, action_std = compute_action_stats(
        dataset, config.PY_ACTION_DIM
    )
    normalize_actions, _denormalize = make_normalizers(action_mean, action_std)

    smolvla = _build_basic_smolvla()

    # Stage 2 (CHANGES 2026-10-29): state も z-score 正規化。
    # stats は model buffer に入り ckpt 経由で rollout へ伝播する。
    state_mean, state_std = compute_state_stats(
        dataset, config.PY_STATE_DIM
    )
    smolvla.set_state_stats(state_mean, state_std)

    # C1: VLM 特徴の事前計算キャッシュ (--no-vlm-cache で無効化)。
    # 凍結 VLM の forward を学習から完全に除外する。
    # precompute はデータ全件ぶんの VLM forward が必要なため、学習 step 数が
    # データ件数を超えないと元が取れない -> should_use_cache() が自動判定。
    if config.PY_USE_VLM_CACHE and not args.no_vlm_cache:
        from .basic_vlm_cache import (
            CachedVlmDataset,
            cache_dir_for,
            ensure_vlm_cache_capacity,
            is_cache_valid,
            precompute_vlm_features,
            should_use_cache,
        )

        cache_dir = cache_dir_for(train_episode_indices)
        if args.vlm_cache_only:
            # 明示要求 = 自動判定を無視して cache を作成する
            # (将来の長め学習 / hyp スイープに備える用途)
            cache_dir = precompute_vlm_features(
                dataset,
                smolvla.vlm_encoder,
                train_episode_indices,
                force=args.force_vlm_cache,
            )
            validation_cache_dir = precompute_vlm_features(
                validation_dataset,
                smolvla.vlm_encoder,
                validation_episode_indices,
                force=args.force_vlm_cache,
            )
            print(
                "VLM cache 準備完了: "
                f"train={cache_dir}, validation={validation_cache_dir}"
            )
            print("(--vlm-cache-only のため学習は行いません)")
            return

        use_cache, reason = should_use_cache(
            dataset_count=len(dataset),
            train_steps=train_steps,
            batch_size=batch_size * args.gradient_accumulation_steps,
            cache_dir=cache_dir,
        )
        print(f"[C1] VLM cache: {'使用' if use_cache else '不使用'} — {reason}")

        if use_cache:
            cache_capacity = ensure_vlm_cache_capacity(
                len(dataset) + len(validation_dataset)
            )
            cache_dir = precompute_vlm_features(
                dataset,
                smolvla.vlm_encoder,
                train_episode_indices,
                force=args.force_vlm_cache,
            )
            dataset = CachedVlmDataset(dataset, cache_dir)
            validation_cache_dir = precompute_vlm_features(
                validation_dataset,
                smolvla.vlm_encoder,
                validation_episode_indices,
                force=args.force_vlm_cache,
            )
            validation_dataset = CachedVlmDataset(
                validation_dataset, validation_cache_dir
            )
        else:
            cache_capacity = None
    else:
        cache_capacity = None

    run_config = {
        "train_steps": train_steps,
        "requested_epochs": args.epochs,
        "effective_epochs": (
            train_steps
            * batch_size
            * args.gradient_accumulation_steps
            / len(dataset)
        ),
        "batch_size": batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": config.TRAIN_LR_PY,
        "warmup_steps": config.TRAIN_WARMUP_PY,
        "checkpoint_freq": args.checkpoint_freq,
        "dataset_samples": len(dataset),
        "train_episodes_per_task": args.train_episodes_per_task,
        "validation_episodes_per_task": args.validation_episodes_per_task,
        "train_episodes": train_episode_indices,
        "validation_episodes": validation_episode_indices,
        "validation_samples": len(validation_dataset),
        "validation_freq": args.validation_freq,
        "validation_max_samples": args.validation_max_samples,
        "validation_seed": args.validation_seed,
        "vlm_cache_capacity": cache_capacity,
        "chunk_size": config.PY_CHUNK_SIZE,
        "flow_solver": config.PY_EVAL_FLOW_SOLVER,
        "receding_steps": config.PY_EVAL_RECEDING_STEPS,
    }
    experiment = create_experiment_run(
        config.WORKDIR,
        stage=args.stage,
        name=args.run_name,
        config_values=run_config,
    )
    experiment.update_manifest(vlm_cache_capacity=cache_capacity)
    print(f"[experiment] {experiment.root}", flush=True)
    wandb_logger = init_wandb(
        experiment,
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config_values=run_config,
    )
    try:
        ckpt_path = train_pytorch_smolvla(
            smolvla,
            dataset,
            normalize_actions,
            action_mean,
            action_std,
            output_dir=experiment.checkpoints,
            plot_dir=experiment.plots,
            metrics_path=experiment.metrics / "train_history.jsonl",
            train_steps=train_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            checkpoint_freq=args.checkpoint_freq,
            resume_path=args.resume,
            wandb_logger=wandb_logger,
            validation_dataset=validation_dataset,
            validation_metrics_path=(
                experiment.metrics / "validation_history.jsonl"
            ),
            validation_freq=args.validation_freq,
            validation_max_samples=args.validation_max_samples,
            validation_seed=args.validation_seed,
        )
        experiment.update_manifest(
            status="completed",
            checkpoint=str(ckpt_path),
            wandb_url=wandb_logger.url,
        )
        args.trained_checkpoint = ckpt_path
    except Exception as error:
        experiment.update_manifest(status="failed", error=repr(error))
        wandb_logger.set_summary(status="failed", error=repr(error))
        raise
    finally:
        wandb_logger.finish()


def _cmd_basic_eval(args: argparse.Namespace) -> None:
    config.ensure_lerobot_on_path()
    config.seed_everything(config.EVAL_SEED)
    # basic_rollout は in-process で libero を import するため fork 側を
    # sys.path に載せる必要がある (Advanced ルートは subprocess_env で解決)
    config.ensure_libero_on_path()
    from .basic_data import make_normalizers
    from .basic_model import build_smolvla, build_vlm_encoder
    from .basic_rollout import evaluate_pytorch_smolvla
    from .experiment import create_experiment_run
    from .wandb_logger import init_wandb

    ckpt_arg = getattr(args, "checkpoint", None)
    trained_checkpoint = getattr(args, "trained_checkpoint", None)
    ckpt_path = (
        ckpt_arg
        or trained_checkpoint
        or config.PY_CHECKPOINT_DIR / "smolvla_ckpt.pt"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint が見つかりません: {ckpt_path}. "
            "先に basic-train を実行してください。"
        )

    import torch

    vlm_encoder = build_vlm_encoder()
    smolvla = build_smolvla(vlm_encoder)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    smolvla.flow_head.load_state_dict(ckpt["flow_head_state_dict"])
    projector_state = ckpt["vlm_projector_state_dict"]
    if projector_state:  # nn.Identity の場合 {} のため
        smolvla.vlm_projector.load_state_dict(projector_state)
    smolvla.state_proj.load_state_dict(ckpt["state_proj_state_dict"])

    # Stage 2: ckpt に state stats があれば復元 (旧 ckpt は identity のまま)
    if "state_mean" in ckpt and "state_std" in ckpt:
        smolvla.set_state_stats(ckpt["state_mean"], ckpt["state_std"])
        print("[state] ckpt の state 正規化 stats を復元しました", flush=True)
    else:
        print("[state] ckpt に state stats なし (無正規化で評価)", flush=True)

    action_mean = ckpt["action_mean"].to("cuda", dtype=torch.float32)
    action_std = ckpt["action_std"].to("cuda", dtype=torch.float32)
    _normalize, denormalize_actions = make_normalizers(action_mean, action_std)

    task_ids = list(getattr(args, "task_ids", config.PY_EVAL_TASK_IDS))
    episodes_per_task = getattr(
        args, "episodes_per_task", config.PY_EVAL_EPISODES_PER_TASK
    )
    env_seed = getattr(args, "env_seed", config.EVAL_SEED)
    policy_seed = getattr(args, "policy_seed", config.PY_EVAL_POLICY_SEED)
    stage = getattr(args, "stage", "stage1-baseline")
    run_name = getattr(args, "run_name", "formal-baseline")
    if trained_checkpoint is not None:
        stage = f"{stage}-eval"
        run_name = f"{run_name}-eval"
    checkpoint_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    eval_config = {
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "task_ids": task_ids,
        "episodes_per_task": episodes_per_task,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "max_steps": config.PY_EVAL_MAX_STEPS,
        "num_flow_steps": config.PY_EVAL_NUM_FLOW_STEPS,
        "flow_solver": config.PY_EVAL_FLOW_SOLVER,
        "receding_steps": config.PY_EVAL_RECEDING_STEPS,
    }
    experiment = create_experiment_run(
        config.WORKDIR,
        stage=stage,
        name=run_name,
        config_values=eval_config,
    )
    print(f"[experiment] {experiment.root}", flush=True)
    wandb_logger = init_wandb(
        experiment,
        mode=getattr(args, "wandb_mode", "auto"),
        project=getattr(args, "wandb_project", config.WANDB_PROJECT_PY),
        entity=getattr(args, "wandb_entity", None),
        config_values=eval_config,
    )
    results_path = experiment.metrics / "eval_results.json"
    try:
        results = evaluate_pytorch_smolvla(
            smolvla,
            denormalize_actions,
            task_ids=task_ids,
            episodes_per_task=episodes_per_task,
            env_seed=env_seed,
            policy_seed=policy_seed,
            results_path=results_path,
            video_dir=experiment.videos,
            wandb_logger=wandb_logger,
        )
        shutil.copy2(results_path, config.PY_EVAL_RESULTS_PATH)
        cache_path = config.CACHE_DIR / config.PY_EVAL_RESULTS_PATH.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results_path, cache_path)
        experiment.update_manifest(
            status="completed",
            checkpoint=str(ckpt_path.resolve()),
            checkpoint_sha256=checkpoint_sha256,
            results=str(results_path),
            overall_success=results["overall_success"],
            wandb_url=wandb_logger.url,
        )
        args.evaluation_results_path = results_path
    except Exception as error:
        experiment.update_manifest(status="failed", error=repr(error))
        wandb_logger.set_summary(status="failed", error=repr(error))
        raise
    finally:
        wandb_logger.finish()


def _cmd_basic_submit(args: argparse.Namespace) -> None:
    from .submission import write_submission_basic

    write_submission_basic(methods_notes=args.notes)


def _cmd_basic_all(args: argparse.Namespace) -> None:
    _cmd_basic_train(args)
    # --vlm-cache-only は cache 作成のみで学習以降を行わない
    if getattr(args, "vlm_cache_only", False):
        return
    _cmd_basic_eval(args)
    _cmd_basic_submit(args)


def _cmd_all(args: argparse.Namespace) -> None:
    # notebook の上から順の実行に対応させる
    _cmd_setup(args)
    _cmd_train(args)
    _cmd_merge(args)
    _cmd_eval(args)
    _cmd_package(args)
    _cmd_adv_eval(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smolvla-adv",
        description=(
            "SmolVLA x LIBERO-plus Spatial LoRA fine-tuning "
            "(最終課題 Advanced の CLI 版)"
        ),
    )
    # グローバルオプション: サブコマンドの前に書く (例: smolvla-adv --gpu 1 train)
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="N",
        help=(
            "使用するGPU番号 (例: --gpu 1)。CUDA_VISIBLE_DEVICES に設定され、"
            "lerobot-train / lerobot-eval の子プロセスにも伝播する。"
            "複数指定は不可 (先頭のみが有効なため)"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="実行環境チェック (Section 1)")
    p.add_argument("--no-gpu", action="store_true", help="GPU 必須チェックを省略")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("setup", help="apt + LeRobot + LIBERO-plus (Sections 2-5)")
    p.add_argument(
        "--skip-apt",
        action="store_true",
        help="システムパッケージ導入をスキップ (コンテナで事前導入済みの場合)",
    )
    p.set_defaults(func=_cmd_setup)

    p = sub.add_parser("data", help="学習データ選抜の確認 (Section 7)")
    p.set_defaults(func=_cmd_data)

    p = sub.add_parser("download", help="事前学習済み SmolVLA の DL (8.1)")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("train", help="LoRA 追加学習 (8.2)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("merge", help="LoRA マージ + ベースライン (8.3-8.4)")
    p.set_defaults(func=_cmd_merge)

    p = sub.add_parser("eval", help="Spatial 評価と比較表 (8.5-8.6)")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("package", help="マージ済みモデルの zip 化 (8.7)")
    p.set_defaults(func=_cmd_package)

    p = sub.add_parser("video", help="rollout 動画の録画 (8.8)")
    p.set_defaults(func=_cmd_video)

    p = sub.add_parser("adv-eval", help="4 suite 広域評価 (Section 9)")
    p.set_defaults(func=_cmd_adv_eval)

    p = sub.add_parser(
        "submit", help="提出用 submission.json 生成 (Advanced ルート)"
    )
    p.add_argument(
        "--notes", default="(自由記述: どんな工夫を試したか等)",
        help="methods_notes に書き込む内容",
    )
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("basic-data", help="Basic: 学習データ選抜の確認")
    p.set_defaults(func=_cmd_basic_data)

    p = sub.add_parser(
        "basic-train", help="Basic: 自作 SmolVLA (SmolVLM2 凍結) を学習"
    )
    _add_basic_train_arguments(p)
    p.set_defaults(func=_cmd_basic_train)

    p = sub.add_parser(
        "basic-eval", help="Basic: LiberoEnv で rollout 評価 + mp4"
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="評価するcheckpoint (省略時はworkdir/py_train/smolvla_ckpt.pt)",
    )
    _add_basic_eval_arguments(p, include_tracking=True)
    p.set_defaults(func=_cmd_basic_eval)

    p = sub.add_parser(
        "basic-submit", help="Basic: 提出用 submission.json 生成"
    )
    p.add_argument(
        "--notes", default="(自由記述: どんな工夫を試したか等)",
        help="methods_notes に書き込む内容",
    )
    p.set_defaults(func=_cmd_basic_submit)

    p = sub.add_parser(
        "basic-all", help="Basic: basic-train -> basic-eval -> basic-submit"
    )
    _add_basic_train_arguments(p)
    _add_basic_eval_arguments(p, include_tracking=False)
    p.add_argument(
        "--notes", default="(自由記述: どんな工夫を試したか等)",
        help="methods_notes に書き込む内容",
    )
    p.set_defaults(func=_cmd_basic_all)

    p = sub.add_parser("all", help="setup -> train -> merge -> eval -> package -> adv-eval")
    p.add_argument("--skip-apt", action="store_true")
    p.set_defaults(func=_cmd_all)

    return parser


def _add_basic_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-vlm-cache", action="store_true",
        help="C1 の VLM 特徴キャッシュを使わず従来通り毎 step VLM を実行",
    )
    parser.add_argument(
        "--force-vlm-cache", action="store_true",
        help="既存 cache を無視して再生成",
    )
    parser.add_argument(
        "--vlm-cache-only", action="store_true",
        help="cache 作成まで行い学習はスキップ (次実行以降が高速)",
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--steps", type=int, default=None,
        help="optimizer step数 (省略時はconfig.TRAIN_STEPS_PY)",
    )
    duration.add_argument(
        "--epochs", type=float, default=None,
        help="train sample数からoptimizer step数を算出するepoch数",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="micro batch size (省略時はconfig.TRAIN_BATCH_SIZE_PY)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="勾配蓄積micro batch数",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int,
        default=config.TRAIN_CHECKPOINT_FREQ_PY,
        help="定期checkpoint間隔 (0で無効)",
    )
    parser.add_argument(
        "--train-episodes-per-task", type=int,
        default=config.PY_TRAIN_EPISODES_PER_TASK,
        help="各taskでtrainへ使用するepisode数",
    )
    parser.add_argument(
        "--validation-episodes-per-task", type=int,
        default=config.PY_VALIDATION_EPISODES_PER_TASK,
        help="各taskでvalidationへ分離するepisode数",
    )
    parser.add_argument(
        "--validation-freq", type=int,
        default=config.TRAIN_VALIDATION_FREQ_PY,
        help="validation lossの計測間隔",
    )
    parser.add_argument(
        "--validation-max-samples", type=int,
        default=config.PY_VALIDATION_MAX_SAMPLES,
        help="validationで等間隔に評価する最大sample数",
    )
    parser.add_argument(
        "--validation-seed", type=int,
        default=config.PY_VALIDATION_SEED,
        help="episode分割と固定validation noise/timeのseed",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="optimizer/scheduler/RNGを含むcheckpointから再開",
    )
    parser.add_argument(
        "--stage", default="stage0", help="run IDに含めるstage名",
    )
    parser.add_argument(
        "--run-name", default="basic-train", help="日時run名のsuffix",
    )
    parser.add_argument(
        "--wandb-mode", choices=("auto", "online", "offline", "disabled"),
        default="auto", help="autoは認証済みならonline、未認証ならoffline",
    )
    parser.add_argument(
        "--wandb-project", default=config.WANDB_PROJECT_PY,
        help="W&B project名",
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity/team (任意)",
    )


def _add_basic_eval_arguments(
    parser: argparse.ArgumentParser, *, include_tracking: bool
) -> None:
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=config.PY_EVAL_TASK_IDS,
        help="評価するLIBERO-Spatial task ID一覧 (0-9)",
    )
    parser.add_argument(
        "--episodes-per-task", type=int,
        default=config.PY_EVAL_EPISODES_PER_TASK,
        help="各taskのrollout episode数",
    )
    parser.add_argument(
        "--env-seed", type=int, default=config.EVAL_SEED,
        help="環境初期化seedの基準値",
    )
    parser.add_argument(
        "--policy-seed", type=int, default=config.PY_EVAL_POLICY_SEED,
        help="Flow noise seedの基準値",
    )
    if not include_tracking:
        return
    parser.add_argument(
        "--stage", default="stage1-baseline", help="run IDに含めるstage名",
    )
    parser.add_argument(
        "--run-name", default="formal-baseline", help="日時run名のsuffix",
    )
    parser.add_argument(
        "--wandb-mode", choices=("auto", "online", "offline", "disabled"),
        default="auto", help="autoは認証済みならonline、未認証ならoffline",
    )
    parser.add_argument(
        "--wandb-project", default=config.WANDB_PROJECT_PY,
        help="W&B project名",
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity/team (任意)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --gpu N: torch / MuJoCo / 子プロセスより先に GPU を固定する。
    # CUDA_VISIBLE_DEVICES は torch が CUDA を初期化する前に設定する必要がある
    # ため、init_environment() より前でなければならない。
    if args.gpu is not None:
        if not args.gpu.isdigit():
            parser.error(f"--gpu は数値で指定してください (例: --gpu 1): {args.gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[--gpu] CUDA_VISIBLE_DEVICES={args.gpu} を設定")

    config.init_environment()

    # GPU を触るコマンドは起動時にチェック (check / data / basic-data 除く)
    if args.command not in {"check", "data", "basic-data"}:
        config.check_runtime(require_gpu=True)

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

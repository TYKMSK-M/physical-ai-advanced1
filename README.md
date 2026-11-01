# smolvla_adv — SmolVLA × LIBERO-plus Spatial 課題パイプライン CLI

最終課題 (Basic) / (Advanced) notebook をローカル実行可能な Python ソースへ移植した
自作パッケージ。データ選抜 → 学習 → LIBERO 評価 → rollout 動画 → 提出物生成までを
17 サブコマンドの CLI (`smolvla-adv`) で自動化する。

- パッケージ種別: **自作実装**（本リポジトリのオリジナルソース）
- 実行: `smolvla_adv/.venv/bin/smolvla-adv <subcommand>` または `python -m smolvla_adv.cli`
- 設定の唯一の入口: `config.py`（パス・revision・seed・ハイパラを一元定義）

## モジュール一覧

### 共通・環境系

| モジュール | 役割 |
|---|---|
| `config.py` | 実行パス・定数・学習/評価ハイパラの一元定義。`subprocess_env()` は子プロセス PATH 先頭へ現 Python 実行 dir を追加し `lerobot-train`/`lerobot-eval` を解決 |
| `cli.py` | CLI エントリポイント。notebook のセクションをサブコマンドへ対応させる |
| `compat.py` | Colab 依存（Drive mount / files.download / display）のローカルフォールバック |
| `proc.py` | 外部コマンド実行ヘルパ（失敗時末尾 6000 文字を例外に載せる） |
| `hf_fetch.py` | HuggingFace 取得の 429 指数バックオフ + キャッシュ優先ラッパ |
| `setup_system.py` | システムパッケージ導入（sudo 有無・非 Debian 系を自動判定） |
| `setup_lerobot.py` | LeRobot v0.6.x の editable install と互換性パッチ（CUDA ビルド整合 `_align_cuda_builds()` 含む） |
| `setup_libero.py` | LIBERO-plus 評価環境のセットアップと、`LEROBOT_MAX_RECORDED` による rollout 録画制御 patch の適用 |
| `experiment.py` | 日時付き実験ディレクトリと再現情報（revision・seed・コマンド）の管理 |
| `wandb_logger.py` | W&B を optional に扱う学習ログラッパ |

### Advanced ルート（LeRobot `lerobot-train` / `lerobot-eval` 経由）

| モジュール | 役割 |
|---|---|
| `data_select.py` | LIBERO-Spatial 10 タスクからの episode 選抜。`TRAIN_EPISODES_PER_TASK=None` で各 task 全件選抜（不均等 episode 数対応）、task 層化 episode 分割 |
| `train.py` | 事前学習済みモデルの DL と LoRA 追加学習（dataset meta 由来の feature 自動推論: front+wrist+state 8D → action 7D、`eval_split=0.2`、W&B）。ACT など他 policy の学習起動にも使用 |
| `merge.py` | LoRA adapter を `merge_and_unload` でベースへマージ + 対照ベースライン作成 |
| `evaluate.py` | `lerobot-eval` による Spatial / 4 suite 評価。`run_evaluation()` は task 数・episode 数・extra_env（録画本数等）を指定可能。実行前に出力 dir を再作成 |
| `videos.py` | 成功 rollout 動画の録画。`recording=false` + `LEROBOT_MAX_RECORDED>0` の組み合わせで MP4 を出力し、生成ファイルを列挙 |
| `package.py` | 学習済みモデルディレクトリの zip 化（提出用） |
| `preflight.py` | 評価入力（モデル dir の必須ファイル構成）の事前チェック |
| `submission.py` | 提出用 `submission.json` / `.md` の生成（hash・環境証拠の自動同梱） |

### Basic ルート（PyTorch 自作 SmolVLA — 教材実装）

| モジュール | 役割 |
|---|---|
| `basic_model.py` | 教材の教育用簡略 VLA（凍結 SmolVLM2-500M + Flow Matching Transformer head）の実装 |
| `basic_data.py` | LIBERO データセット読み込み（chunk 付き）と action z-score 正規化 |
| `basic_train.py` | 自作学習ループ（AdamW + warmup→cosine + grad clip） |
| `basic_vlm_cache.py` | C1: 凍結 VLM 特徴の事前計算キャッシュ。`should_use_cache()` が step×batch > N のときのみ自動 ON |
| `basic_rollout.py` | LiberoEnv で 1 episode 実行 → 成功可否 + frame 列、mp4 化 |

## サブコマンド

```
check        実行環境チェック (Section 1)
setup        apt + LeRobot + LIBERO-plus (Sections 2-5)
data         学習データ選抜の確認 (Section 7)
download     事前学習済み SmolVLA の DL (8.1)
train        LoRA 追加学習 / policy 学習 (8.2)
merge        LoRA マージ + ベースライン (8.3-8.4)
eval         Spatial 評価と比較表 (8.5-8.6)
package      モデルの zip 化 (8.7)
video        rollout 動画の録画 (8.8)
adv-eval     4 suite 広域評価 (Section 9)
submit       提出物生成
basic-data / basic-train / basic-eval / basic-submit / basic-all   Basic ルート
all          setup -> train -> merge -> eval -> package -> adv-eval
```

## 最終提出（ACT）での使用例

```bash
cd smolvla_adv
.venv/bin/python - <<'PY'
from smolvla_adv import config
config.init_environment(); config.ensure_lerobot_on_path()
from smolvla_adv.evaluate import run_evaluation

policy = config.WORKDIR / "model_tournament/act_spatial_full_20261101/checkpoints/020000/pretrained_model"
run_evaluation(policy, config.WORKDIR / "submission/act_success_videos/task_0",
               "ACT videos", task_ids=[0], n_episodes=2,
               extra_env={"LEROBOT_MAX_RECORDED": "2"})
PY
```

- 学習: `lerobot-train` を `train.py` 経由で起動（ACT from scratch: 20,000 steps / batch 32 / LR 1e-5）
- 評価: quick 10/10 → formal 48/50 (96%) → 別環境 seed 95/100 (95%)
- 提出物: `workdir/submission/`（ZIP・manifest・成功動画。SHA-256 は `act_submission.json` で連鎖検証）

## テスト

```bash
cd smolvla_adv && .venv/bin/python -m pytest tests/ -q   # 12 passed
```

`tests/`（パッケージ外 `smolvla_adv/tests/`）に選抜・設定・CLI・モデルの回帰テストがある。
特に `test_config.py` は subprocess PATH 契約（現 Python 実行 dir が先頭）を検証する。

## 依存

- Python 3.12 / PyTorch / LeRobot v0.6.x（`workdir/lerobot` の local checkout）
- LIBERO-plus（robosuite + MuJoCo、`workdir/LIBERO-plus`）
- 任意: Weights & Biases（`wandb_logger.py` 経由、未設定でも学習は走る）

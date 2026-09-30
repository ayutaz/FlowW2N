# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

FlowW2N（Whispered-to-Normal Speech Conversion via Flow-Matching）論文の再現実装プロジェクト。
ささやき音声を通常音声に変換するためのconditional flow matchingモデル。

- 論文: arXiv:2603.04296v1 [eess.AS] (Samsung Electronics)
- 詳細ドキュメント: `docs/` ディレクトリ参照

## アーキテクチャ概要

FlowW2Nは3つの主要コンポーネントで構成される:

1. **VAE (Oobleck encoder-decoder)**: 波形→潜在空間の圧縮・復元 (D=64, ~15.6Hz)。stable-audio-toolsベース。normal speechのみで学習。
2. **DiT (Diffusion Transformer)**: 24 transformer blocks。ガウスノイズz0~N(0,I)から条件付き速度場を学習。AdaLNでタイムステップ・話者埋め込みを注入、cross-attentionでコンテンツ特徴を注入。
3. **条件付けモジュール**: Whisper Base encoder (layer 5, 凍結) + ECAPA-TDNN話者エンコーダ (凍結)

多言語対応（英語+日本語）: language embedding を FlowW2NModel レベルで追加し、AdaLN で DiT に注入。

学習は合成ウィスパー-通常音声ペアのみ使用。推論はEuler積分10ステップ（2次精度のHeunソルバーも選択可能）。

## パッケージ管理

```bash
uv add <package>    # 依存追加
uv run python <script>  # スクリプト実行
```

## プロジェクト構成

```
src/floww2n/           # メインパッケージ
  models/              # モデル定義
    vae.py             # Oobleck VAE (~573 lines, ~40M params)
    dit.py             # DiffusionTransformer (~580 lines, ~205M params)
    floww2n.py         # FlowW2NModel: CFM loss + Euler sampling
    content_encoder.py # Whisper Base layer 5 wrapper (凍結)
    speaker_encoder.py # ECAPA-TDNN wrapper (凍結)
  training/            # 学習関連
    losses.py          # Multi-res STFT + Discriminator + KL loss
    dataset.py         # VAEDataset + DiTDataset + dit_collate_fn
    train_vae.py       # VAE 学習ループ (EMA, bf16, discriminator)
    train_dit.py       # DiT 学習ループ (CFM, EMA, bf16, warmup)
    checkpoint_utils.py # チェックポイントのステップ順ソート・ローテーション
  inference/           # 推論パイプライン
    pipeline.py        # FlowW2NPipeline: end-to-end推論
    sampler.py         # Euler sampler
  data/                # 合成ウィスパー生成
    whisper_synthesis.py  # 4手法 (LPC, glottal, formant, praat)
    preprocess.py      # データ前処理
  evaluation/          # 評価パイプライン
    metrics.py         # WER, UTMOS, DNSMOS, SpkSim
    evaluate.py        # 評価実行スクリプト
configs/               # 設定ファイル (vae.json, dit.json, data.json)
scripts/               # エントリポイントスクリプト
  train_vae.py, train_dit.py, cache_features.py, generate_whisper.py, inference.py, evaluate.py, preprocess_data.py, smoke_test_training.py
tests/                 # テスト (86 tests passing)
  test_vae.py (9), test_dit.py (14), test_losses.py (12), test_pipeline.py (21), test_dataset.py (9), test_whisper_synthesis.py (11), test_training.py (10)
```

## 設定ファイル

- `configs/vae.json` - VAE モデル・学習パラメータ (sr=16kHz, strides=[4,4,8,8], D=64)
- `configs/dit.json` - DiT モデル・Flow Matching パラメータ (embed_dim=768, depth=24)
- `configs/data.json` - データセット・合成ウィスパー・評価データセット設定

## ドキュメント構成

- `docs/01_overview.md` - 論文概要・背景・貢献
- `docs/02_method.md` - 手法・数式・アーキテクチャ詳細
- `docs/03_experiments_results.md` - 実験設定・結果・アブレーション
- `docs/04_implementation_guide.md` - 再現実装ガイド・ハイパーパラメータ・チェックリスト
- `docs/05_plan_model.md` - コアモデル（VAE + DiT）実装計画
- `docs/06_plan_data.md` - データパイプライン計画
- `docs/07_plan_conditioning_inference.md` - 条件付け・推論・評価パイプライン計画
- `docs/08_plan_overall.md` - 総合実装計画・スケジュール・リスク分析

## 主要な外部依存

| コンポーネント | ソース |
|---------------|--------|
| VAE + DiT ベース | stable-audio-tools (Stability-AI) |
| Content Encoder | OpenAI Whisper Base (layer 5) |
| Speaker Encoder | speechbrain/spkrec-ecapa-voxceleb |
| 学習データ | HiFi-TTS-2 (nvidia/hifitts-2) |
| 合成ウィスパー | LPC devoicing, glottal removal, formant mod, Praat |

## 重要な設計判断

- Paired flow matching（z0=whisper, z1=normal）は時間的ミスアライメントで失敗する → ガウスノイズ事前分布+外部条件付けで解決
- ASR特徴（Whisper）はdomain-invariantであり、合成ウィスパーで学習しても実ウィスパーに汎化可能
- レイヤー選択基準: `ℓ* = argmax[Invariance(ℓ) × CCA(ℓ)]`
- 多言語対応: FlowW2NModel レベルで language embedding を追加、global_cond = speaker_proj + language_emb で DiT に AdaLN 注入。num_languages ≤ 1 では挙動不変（後方互換）

## 開発スコープ

本リポジトリは再現実装コード（モデル定義、学習スクリプト、推論・評価パイプライン）の作成が対象。
データセットのダウンロード・前処理および学習の実行は別環境で行う。

## 実装状況

| マイルストーン | 状態 | 内容 |
|--------------|------|------|
| M0: 環境構築 | ✅ 完了 | pyproject.toml, ディレクトリ構造, 設定ファイル |
| M1: VAE | ✅ 完了 | Oobleck VAE, 損失関数, 学習スクリプト |
| M2: 条件付け | ✅ 完了 | Whisper/ECAPA encoder, 合成ウィスパー, キャッシュ |
| M3: DiT・CFM | ✅ 完了 | DiffusionTransformer, FlowW2NModel, 学習スクリプト |
| M4: 推論・評価 | ✅ 完了 | FlowW2NPipeline, 評価指標, 統合テスト |
| 品質監査 | ✅ 完了 | ruff lint/format, バグ修正, テスト追加, 最適化 |
| 多言語対応 | ✅ 完了 | 英語+日本語, language embedding, JVS/JSUT対応 |
| 最適化 | ✅ 完了 | torch.compile, CUDA最適化, Heunソルバー, 非同期チェックポイント |

## テスト実行

```bash
uv run pytest tests/ -v   # 全テスト実行 (86 tests)
uv run pytest tests/test_vae.py -v  # VAE テストのみ (9)
uv run pytest tests/test_dit.py -v  # DiT テストのみ (14)
uv run pytest tests/test_losses.py -v  # 損失関数テスト (12)
uv run pytest tests/test_pipeline.py -v  # パイプラインテスト (21)
uv run pytest tests/test_dataset.py -v  # データセットテスト (9)
uv run pytest tests/test_whisper_synthesis.py -v  # 合成ウィスパーテスト (11)
uv run pytest tests/test_training.py -v  # 学習テスト (10)
uv run ruff check src/ tests/ scripts/  # lint チェック
uv run ruff format --check src/ tests/ scripts/  # フォーマットチェック
```

## 学習実行

```bash
# VAE 学習
uv run python scripts/train_vae.py --config configs/vae.json --data-dir <path> --output-dir outputs/vae

# 合成ウィスパー生成
uv run python scripts/generate_whisper.py --data-dir <path> --output-dir <path> --num-workers 8

# 特徴量キャッシュ
uv run python scripts/cache_features.py --data-dir <path> --output-dir <cache_path> --vae-checkpoint <vae_ckpt> --batch-size 16

# DiT 学習
uv run python scripts/train_dit.py --config configs/dit.json --cache-dir <cache_path> --output-dir outputs/dit

# 推論
uv run python scripts/inference.py --vae-checkpoint <vae_ckpt> --dit-checkpoint <dit_ckpt> --input <wav_or_dir> --output-dir <out_dir> --solver euler|heun
```

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

学習は合成ウィスパー-通常音声ペアのみ使用。推論はEuler積分10ステップ。

## パッケージ管理

```bash
uv add <package>    # 依存追加
uv run python <script>  # スクリプト実行
```

## プロジェクト構成

```
src/floww2n/           # メインパッケージ
  models/              # VAE, DiT, content/speaker encoder, 統合モデル
  training/            # 学習スクリプト, 損失関数, データセット
  inference/           # 推論パイプライン, Euler サンプラー
  data/                # 前処理, 合成ウィスパー生成
  evaluation/          # 評価指標, 評価パイプライン
configs/               # 設定ファイル (vae.json, dit.json, data.json)
scripts/               # エントリポイントスクリプト
tests/                 # テスト
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

## 開発スコープ

本リポジトリは再現実装コード（モデル定義、学習スクリプト、推論・評価パイプライン）の作成が対象。
データセットのダウンロード・前処理および学習の実行は別環境で行う。

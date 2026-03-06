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

## ドキュメント構成

- `docs/01_overview.md` - 論文概要・背景・貢献
- `docs/02_method.md` - 手法・数式・アーキテクチャ詳細
- `docs/03_experiments_results.md` - 実験設定・結果・アブレーション
- `docs/04_implementation_guide.md` - 再現実装ガイド・ハイパーパラメータ・チェックリスト

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

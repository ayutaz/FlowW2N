# FlowW2N: 論文概要

## 1. 論文情報

| 項目 | 内容 |
|------|------|
| タイトル | FlowW2N: Whispered-to-Normal Speech Conversion via Flow-Matching |
| 著者 | Fabian Ritter-Gutierrez, Md Asif Jalal, Pablo Peso Parada, Karthikeyan Saravanan, Yusun Shul, Minseung Kim, Gun-Woo Lee, Han-Gil Moon |
| 所属 | Samsung Electronics R&D Institute UK (SRUK) / Samsung Electronics Mobile eXperience Business |
| arXiv ID | 2603.04296v1 [eess.AS] 4 Mar 2026 |

## 2. 研究の背景と動機

### ささやき音声から通常音声への変換（W2N）の課題

ささやき音声（whispered speech）は声帯振動を伴わないため、通常の発声と比較して以下の根本的な違いがある。

- **基本周波数（F0）の欠如**: ささやき音声には声帯振動に起因するピッチ情報が存在しない。
- **調和構造（harmonic structure）の欠如**: 通常音声に見られる倍音構造が失われている。

これらの特徴から、W2N変換は単なるノイズ除去（denoising）とは本質的に異なるタスクである。特に、ささやき音声と通常音声の間には **時間的ミスアライメント（temporal misalignment）** が存在し、同一話者が同一内容を発話しても、ささやきと通常発声ではフレームレベルでの対応関係が一致しない。この時間的ずれが、ペアデータの構築や学習を困難にしている。

さらに、実環境において **ささやき音声と通常音声のペアデータを大規模に収集することが極めて困難** であるという実用上の制約も存在する。

## 3. 主要な貢献

本論文の主要な貢献は以下の3点である。

### (i) Conditional Flow Matching による W2N 変換で SOTA を達成

Conditional flow matching を W2N タスクに適用し、CHAINS および wTIMIT データセットにおいて最先端（SOTA）の Word Error Rate（WER）を達成した。推論時にはわずか **10ステップ** のみで高品質な変換を実現しており、既存手法と比較して WER を **26--46% 相対削減** した。

### (ii) 合成データのみでの学習とドメイン不変条件付け

学習には **合成的に生成された時間整合済みのささやき-通常音声ペア（synthetic, time-aligned whisper-normal pairs）** のみを使用し、実際のペアデータを一切必要としない。高レベルの ASR 埋め込み（ASR embeddings）がドメイン不変（domain-invariant）な特徴を持つことを利用し、合成ささやき音声と実ささやき音声の間で高い不変性を示すことを確認した。これにより、学習時に実ささやき音声を一度も観測していないにもかかわらず、実ささやき音声への汎化を実現した。

### (iii) ASR レイヤー選択基準の提案

ASR モデルの各レイヤーにおけるドメイン間不変性を検証し、**コンテンツ情報量（content informativeness）** と **クロスドメイン不変性（cross-domain invariance）** のバランスを最適化するレイヤー選択基準を提案した。

## 4. 既存手法の問題点と本手法の位置づけ

### 既存手法の問題点

これまでの W2N 変換手法にはそれぞれ以下の課題があった。

| 手法カテゴリ | 代表例 | 問題点 |
|-------------|--------|--------|
| VAE ベース | -- | 出力の **過平滑化（over-smoothing）** により、自然さや明瞭性が低下する |
| GAN ベース | -- | **学習の不安定性** および生成結果における **アーティファクト** の発生 |
| SSL ベース | WESPER, DistillW2N | 変換後の WER が **入力ささやき音声よりも悪化** してしまう場合がある |

いずれの手法においても、変換後の音声の明瞭性（intelligibility）が十分に改善されないという共通の課題が存在していた。

### Flow Matching の直接適用における課題

Flow matching を W2N に単純に適用した場合、**phoneme boundary blur（音素境界のぼやけ）** 効果が発生することが確認されている。また、Dynamic Time Warping（DTW）を用いた時間整合においても、フレームレベルでの音韻一貫性（phonetic consistency）は保証できない。

### 本手法（FlowW2N）の位置づけ

FlowW2N は、上記の問題を以下のアプローチで解決する。

- **Conditional flow matching** を採用し、VAE の過平滑化や GAN の不安定性を回避しつつ、高品質な音声生成を実現する。
- **合成データのみで学習** することで、実ペアデータ収集の困難さを克服する。
- **ドメイン不変な ASR 特徴量** を条件付けに利用することで、合成データで学習したモデルが実ささやき音声にも汎化する。
- 推論時の計算コストが低く（10ステップ）、実用性が高い。

これらにより、FlowW2N は CHAINS および wTIMIT データセットにおいて既存手法を大幅に上回る明瞭性を達成し、W2N 変換タスクにおける新たな最先端手法として位置づけられる。

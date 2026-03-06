# FlowW2N: 総合実装計画

本ドキュメントでは、FlowW2N（Whispered-to-Normal Speech Conversion via Flow-Matching, arXiv:2603.04296v1）の再現実装コードの開発計画を策定する。フェーズ別開発スケジュール、計算リソース見積もり、論文未記載パラメータの推定、リスク分析、マイルストーンを包括的にまとめる。

**スコープ**: 本計画の対象は再現実装コード（モデル定義、学習スクリプト、推論・評価パイプライン）の作成のみ。データセットのダウンロード・前処理、および学習の実行は別環境で行う前提とする。

---

## 1. フェーズ別開発スケジュール

### 1.1 全体ガントチャート

```
Week  1    2    3    4
      |    |    |    |
Ph0   [==]                 環境構築・プロジェクト構造
Ph1   [======]             VAE モデル・学習コード
Ph2      [======]          条件付けモジュール・合成ウィスパーコード
Ph3         [========]     DiT・Flow Matching・推論・評価コード
```

### 1.2 Phase 0: 環境構築・プロジェクト構造 (Week 1 前半)

| タスク | 詳細 | 完了条件 |
|--------|------|---------|
| 依存ライブラリ定義 | pyproject.toml に全依存を定義 | `uv sync` で全パッケージインストール成功 |
| ディレクトリ構造 | src/floww2n/ 以下のパッケージ構築 | 全モジュールのインポート成功 |
| 設定ファイル | configs/ に VAE/DiT/data の JSON 作成 | 設定の読み込み・検証コード動作 |

### 1.3 Phase 1: VAE モデル・学習コード (Week 1-2)

| タスク | 詳細 | 完了条件 |
|--------|------|---------|
| Oobleck VAE 移植 | stable-audio-tools から必要コードを抽出・適応 | ダミー入力でフォワードパス成功 |
| VAE 設定調整 | sr=16kHz, strides=[4,4,8,8], D=64 | 出力 shape 検証: フレームレート ≈ 15.6Hz |
| 損失関数実装 | Multi-res STFT + Discriminator + KL | 損失計算がエラーなく動作 |
| VAE 学習スクリプト | train_vae.py（DataLoader, optimizer, ループ, チェックポイント保存） | ダミーデータで1 step 実行成功 |
| DataLoader 実装 | VAEDataset（固定長セグメント、リサンプリング対応） | 単体テスト通過 |

### 1.4 Phase 2: 条件付けモジュール・合成ウィスパーコード (Week 1-2)

| タスク | 詳細 | 完了条件 |
|--------|------|---------|
| Whisper Base ラッパー | HuggingFace Transformers, layer 5 出力取得 | ダミー入力で shape = (B, 1500, 512) 確認 |
| ECAPA-TDNN ラッパー | SpeechBrain spkrec-ecapa-voxceleb | ダミー入力で shape = (B, 192) 確認 |
| 合成ウィスパーモジュール | 4手法の実装 + WhisperSynthesizer 統合クラス | 単体テスト通過 |
| 特徴量キャッシュスクリプト | Whisper h, ECAPA e_spk, VAE z1 の事前計算スクリプト | コード実行可能（データなしでも構造確認） |

### 1.5 Phase 3: DiT・Flow Matching・推論・評価コード (Week 2-4)

| タスク | 詳細 | 完了条件 |
|--------|------|---------|
| DiT アーキテクチャ | 24 blocks, AdaLN, cross-attention | ダミー入力でフォワードパス成功、出力 shape 確認 |
| Flow Matching 学習ループ | CFM 目的関数、train_dit.py | ダミーデータで1 step 実行成功 |
| DiT DataLoader | DiTDataset + collate 関数（可変長対応） | 単体テスト通過 |
| 推論パイプライン | FlowW2NPipeline + Euler サンプラー | ランダム入力でエンドツーエンド推論成功 |
| 評価パイプライン | WER-N, WER-W, UTMOS, DNSMOS, SpkSim | 各指標の計算コードが動作（ダミーデータ可） |
| 統合テスト | 全コンポーネントの結合テスト | ランダム重みで推論→評価のパイプライン通過 |

---

## 2. 計算リソース見積もり

### 2.1 GPU VRAM 要件

| コンポーネント | モデルサイズ | 学習時 VRAM (推定) | 備考 |
|---------------|------------|-------------------|------|
| VAE (Oobleck) | ~10-20M params | 8-12 GB | batch 256 は勾配累積で対応 |
| DiT (embed_dim=768) | ~170M params | 16-24 GB | batch は実効16程度 |
| Whisper Base (凍結) | 74M params | ~2 GB | 推論のみ |
| ECAPA-TDNN (凍結) | ~6M params | ~0.5 GB | 推論のみ |
| DiT 学習合計 | -- | **20-30 GB** | 混合精度 (bf16) 使用時 |

### 2.2 GPU タイプ別の学習時間見積もり

| GPU | VRAM | VAE (80K steps) | DiT (300K steps) | 合計 |
|-----|------|-----------------|------------------|------|
| RTX 3090 | 24 GB | ~8-12時間 | ~3-5日 | ~4-6日 |
| RTX 4090 | 24 GB | ~5-8時間 | ~2-3日 | ~2-4日 |
| A100 40GB | 40 GB | ~4-6時間 | ~1.5-2.5日 | ~2-3日 |
| A100 80GB | 80 GB | ~3-5時間 | ~1-2日 | ~1-2日 |
| H100 | 80 GB | ~2-4時間 | ~0.5-1.5日 | ~1-2日 |

**注意**: これらは単一 GPU での推定値。マルチ GPU で線形に高速化可能（DDP）。

### 2.3 ストレージ要件

| データ | 推定サイズ |
|--------|-----------|
| HiFi-TTS-2 (元データ) | ~50-100 GB |
| 前処理済みデータ (16kHz) | ~30-60 GB |
| 合成ウィスパー (4手法分) | ~120-240 GB |
| キャッシュ (Whisper h, e_spk, z1) | ~50-100 GB |
| チェックポイント | ~5-10 GB |
| **合計** | **~250-500 GB** |

### 2.4 クラウド GPU コスト見積もり

| プロバイダ | GPU | 時間単価 (目安) | 総コスト (推定) |
|-----------|-----|----------------|----------------|
| Lambda Labs | A100 80GB | ~$1.50/h | $72-144 |
| Vast.ai | A100 40GB | ~$0.80/h | $48-96 |
| RunPod | RTX 4090 | ~$0.40/h | $38-76 |
| Google Colab Pro+ | A100 | 月額$49.99 | $50-100 |

---

## 3. 論文未記載パラメータの推定

論文で明示されていないが実装に必要なパラメータを、stable-audio-tools (SA2.0) の設定と論文のヒントから推定する。

### 3.1 DiT ハイパーパラメータ

| パラメータ | 論文記載 | 推定値 | 根拠 |
|-----------|---------|--------|------|
| embed_dim | 未記載 | **768** (初期値) | SA2.0は1536だが、FlowW2Nは小規模タスク。512-1024の範囲で探索 |
| num_heads | 未記載 | **12** | embed_dim=768の場合、head_dim=64が標準 |
| cond_token_dim | 未記載 | **512** | Whisper Base の d_model=512 |
| global_cond_dim | 未記載 | **768** | embed_dim と同一が SA2.0 の慣例 |
| 学習ステップ数 | 未記載 | **200K-500K** | SA2.0 は大規模だが、FlowW2N は小規模タスクのため少なめ |
| Optimizer | 未記載 | **AdamW** | SA2.0 準拠 |
| 学習率 | 未記載 | **1e-4 ~ 3e-4** | SA2.0: 1.5e-4、DiT原論文: 1e-4 |
| Weight decay | 未記載 | **0.01** | AdamW デフォルト |
| EMA decay | 未記載 | **0.9999** | SA2.0 準拠 |
| Warmup steps | 未記載 | **1000-5000** | 一般的な範囲 |
| 混合精度 | 未記載 | **bf16** | 現代的な学習では標準 |

### 3.2 VAE ハイパーパラメータ

| パラメータ | 論文記載 | 推定値 | 根拠 |
|-----------|---------|--------|------|
| channels | 未記載 | **128** | SA2.0 のデフォルト |
| c_mults | 未記載 | **[1, 2, 4, 8]** | SA2.0: [1,2,4,8,16] から1段削減（stride 1段少ないため） |
| Optimizer | 未記載 | **AdamW** | SA2.0 準拠 |
| 学習率 | 未記載 | **1.5e-4** | SA2.0 の設定 |
| Discriminator | 未記載 | **MultiScale** | SA2.0 準拠 |
| adv_loss_weight | 未記載 | **0.1** | SA2.0 の設定 |
| feat_loss_weight | 未記載 | **5.0** | SA2.0 の設定 |
| kl_weight | 未記載 | **1e-4** | SA2.0 の設定 |

### 3.3 学習データ関連

| パラメータ | 論文記載 | 推定値 | 根拠 |
|-----------|---------|--------|------|
| HiFi-TTS-2 サブセット | 未記載 | **全量使用** | 論文にサブセットの言及なし |
| セグメント長 (VAE) | 未記載 | **65,536 samples** | SA2.0: 65536 samples @ 44.1kHz |
| セグメント長 (DiT) | 未記載 | **可変長 (最大10秒)** | 推論時を考慮 |
| 合成ウィスパー生成タイミング | 未記載 | **事前生成** | 効率性を考慮 |

---

## 4. リスク分析

### 4.1 リスクマトリクス

| ID | リスク | 影響度 | 発生確率 | 対策 |
|----|--------|--------|---------|------|
| R1 | SR/stride 設定の誤り | 高 | 中 | 16kHz+[4,4,8,8] を第一候補、不十分なら 32kHz+[2,4,4,8,8] |
| R2 | embed_dim の不適切な選択 | 中 | 中 | 768 で開始、512/1024 でアブレーション |
| R3 | VAE の復元品質不足 | 高 | 低 | SA2.0 の設定を忠実に再現、損失関数のバランス調整 |
| R4 | 合成ウィスパーの品質不足 | 高 | 中 | 4手法の品質を個別検証、実ウィスパーとの Pearson 相関確認 |
| R5 | DiT の学習不安定 | 中 | 中 | 学習率ウォームアップ、勾配クリッピング、EMA |
| R6 | ドメインシフト（合成→実ウィスパー） | 高 | 中 | Whisper layer 5 の domain invariance を事前検証 |
| R7 | 計算リソース不足 | 中 | 低-中 | 勾配累積、混合精度、チェックポイント |
| R8 | 評価データセットの入手困難 | 中 | 中 | wTIMIT: LDC 経由、CHAINS: 公開データ |
| R9 | Python / ライブラリ互換性 | 低 | 中 | Python 3.11/3.12 固定、仮想環境で管理 |
| R10 | 合成ウィスパー手法の実装困難 | 中 | 中 | LPC + Praat を先行実装、残り2手法は後回し可 |

### 4.2 リスク軽減の優先順序

1. **R1 (SR/stride)**: Phase 1 の最初に確認。VAE のフレームレートが 15.6Hz に一致することを検証。
2. **R4 (合成ウィスパー品質)**: Phase 2 で合成結果を Whisper 特徴で検証。
3. **R6 (ドメインシフト)**: Phase 2 で CHAINS のトリプレットを使って Pearson 相関を計算。
4. **R3 (VAE品質)**: Phase 1 完了時に復元品質を確認。
5. **R5 (DiT学習)**: Phase 3 で段階的に学習を進め、中間チェックポイントで品質確認。

---

## 5. マイルストーンと完了基準

### M0: プロジェクト基盤完了

- [ ] pyproject.toml に全依存定義、`uv sync` 成功
- [ ] ディレクトリ構造・パッケージ構造の構築完了
- [ ] configs/ に VAE/DiT/data の設定ファイル作成

**完了基準**: `uv run python -c "import floww2n"` が成功。

### M1: VAE コード完成

- [ ] OobleckEncoder/Decoder の移植完了
- [ ] ダミー入力 (B, 65536) → VAE → 復元 (B, 65536) のフォワードパス成功
- [ ] フレームレート ≈ 15.6 Hz（潜在 shape = (B, 64, L)）確認
- [ ] 損失関数（Multi-res STFT + Discriminator + KL）のコード動作確認
- [ ] train_vae.py でダミーデータ 1 step 実行成功

**完了基準**: VAE の encode → decode → loss 計算が正常動作。

### M2: 条件付けモジュール・合成ウィスパーコード完成

- [ ] Whisper Base ラッパーでダミー入力から layer 5 出力取得成功
- [ ] ECAPA-TDNN ラッパーでダミー入力から話者埋め込み取得成功
- [ ] 合成ウィスパー 4 手法の実装完了、単体テスト通過
- [ ] 特徴量キャッシュスクリプトの実装完了

**完了基準**: 全条件付けモジュールが正しい shape のテンソルを出力。

### M3: DiT・学習コード完成

- [ ] DiT (24 blocks, AdaLN, cross-attention) のフォワードパス成功
- [ ] CFM 目的関数の実装完了
- [ ] train_dit.py でダミーデータ 1 step 実行成功
- [ ] DiTDataset + collate 関数の単体テスト通過

**完了基準**: DiT の入力→速度場出力→損失計算が正常動作。

### M4: 推論・評価パイプライン完成

- [ ] FlowW2NPipeline でランダム重みによるエンドツーエンド推論成功
- [ ] Euler サンプラー（N=10）の動作確認
- [ ] 評価指標（WER-N, WER-W, UTMOS, DNSMOS, SpkSim）の計算コード動作
- [ ] 統合テスト: 推論→評価のパイプライン全体が通過

**完了基準**: ダミーまたはランダム重みで推論→波形出力→評価スコア算出が一通り動作。全コードが学習環境にデプロイ可能な状態。

---

## 6. ディレクトリ構成（推奨）

```
FlowW2N/
├── CLAUDE.md
├── pyproject.toml
├── docs/                          # ドキュメント
│   ├── 01_overview.md
│   ├── 02_method.md
│   ├── 03_experiments_results.md
│   ├── 04_implementation_guide.md
│   ├── 05_plan_model.md
│   ├── 06_plan_data.md
│   ├── 07_plan_conditioning_inference.md
│   └── 08_plan_overall.md
├── configs/                       # 設定ファイル
│   ├── vae.json
│   ├── dit.json
│   └── data.json
├── src/
│   └── floww2n/
│       ├── __init__.py
│       ├── models/                # モデル定義
│       │   ├── __init__.py
│       │   ├── vae.py             # Oobleck VAE
│       │   ├── dit.py             # DiffusionTransformer
│       │   ├── content_encoder.py # Whisper Base wrapper
│       │   ├── speaker_encoder.py # ECAPA-TDNN wrapper
│       │   └── floww2n.py         # 統合モデル
│       ├── training/              # 学習関連
│       │   ├── __init__.py
│       │   ├── train_vae.py
│       │   ├── train_dit.py
│       │   ├── losses.py
│       │   └── dataset.py
│       ├── inference/             # 推論パイプライン
│       │   ├── __init__.py
│       │   ├── pipeline.py        # FlowW2NPipeline
│       │   └── sampler.py         # Euler sampler
│       ├── data/                  # データ処理
│       │   ├── __init__.py
│       │   ├── preprocess.py
│       │   └── whisper_synthesis.py
│       └── evaluation/            # 評価
│           ├── __init__.py
│           ├── metrics.py
│           └── evaluate.py
├── scripts/                       # 実行スクリプト
│   ├── preprocess_data.py
│   ├── generate_whisper.py
│   ├── cache_features.py
│   ├── train_vae.py
│   ├── train_dit.py
│   ├── inference.py
│   └── evaluate.py
└── tests/                         # テスト
    ├── test_vae.py
    ├── test_dit.py
    └── test_pipeline.py
```

---

## 7. Quick-Win 戦略

最小限の労力で動作するプロトタイプを早期に構築し、段階的に品質を向上させる方針。

### 7.1 最小実行可能コード (MVP)

1. **VAE**: SA2.0 のコードを最小限の修正で移植（stride/SR 変更のみ）
2. **DiT**: SA2.0 の DiffusionTransformer を CFM 目的関数に変更
3. **合成ウィスパー**: LPC devoicing 1手法のみで実装
4. **条件付け**: Whisper layer 5 + ECAPA-TDNN ラッパー
5. **推論**: FlowW2NPipeline + Euler サンプラー

### 7.2 段階的実装

| 段階 | 追加コード | 成果物 |
|------|---------|---------|
| MVP | VAE + DiT + 条件付け + 推論 | ダミーデータでエンドツーエンド動作 |
| +1 | 合成ウィスパー残り3手法 | 全4手法の WhisperSynthesizer |
| +2 | 学習スクリプト完成 | train_vae.py, train_dit.py が実行可能 |
| +3 | 評価パイプライン | 全5指標の計算コード |
| +4 | 設定ファイル・テスト | configs/, tests/ の整備 |

---

## 8. まとめ

1. **4フェーズ、約3-4週間** のコード実装スケジュール（学習・データ準備は別環境）
2. **論文未記載パラメータ** は SA2.0 と DiT 原論文から推定（embed_dim=768, lr=1.5e-4 等）
3. **最大リスク** は SR/stride 設定とライブラリ互換性 → ダミーデータでの早期検証で軽減
4. **Quick-Win 戦略** により、まず MVP コードを構築し、段階的にコンポーネントを追加
5. **成果物**: 学習環境にデプロイ可能な再現実装コード一式（モデル、学習スクリプト、推論・評価パイプライン）

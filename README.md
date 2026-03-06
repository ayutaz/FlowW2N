# FlowW2N

**Whispered-to-Normal Speech Conversion via Flow-Matching** の再現実装

> 論文: [FlowW2N: Whispered-to-Normal Speech Conversion via Flow-Matching](https://arxiv.org/abs/2603.04296) (arXiv:2603.04296v1, Samsung Electronics)

## Overview

FlowW2N は、ささやき音声（whispered speech）を通常音声（normal speech）に変換する conditional flow matching モデルです。合成ウィスパーデータのみで学習し、実ウィスパー音声にも汎化できることが特徴です。

### Key Features

- **Conditional Flow Matching**: ガウスノイズ z0 ~ N(0, I) から条件付き速度場を学習し、Euler 積分 10 ステップで高品質な変換を実現
- **合成データのみで学習**: 実ウィスパー音声のペアデータ不要。4手法の信号処理ベース合成ウィスパーで学習
- **Domain-Invariant 条件付け**: Whisper Base encoder の ASR 特徴量がドメイン不変であることを利用し、合成→実ウィスパーへの汎化を実現
- **SOTA 性能**: CHAINS / wTIMIT で既存手法比 WER 26-46% 相対改善

## Architecture

```
入力: ささやき音声 s_whisper (16kHz, mono)
  |
  +---> [Whisper Base encoder, layer 5, 凍結]
  |         |-> h: (B, 1500, 512)    コンテンツ特徴
  |
  +---> [ECAPA-TDNN, 凍結]
  |         |-> e_spk: (B, 192)      話者埋め込み
  |
  +---> z0 ~ N(0, I)                 (B, 64, L)
  |
  +---> [DiT: 24 Transformer blocks, Euler 積分 10 steps]
  |         - AdaLN: timestep + speaker embedding
  |         - Cross-Attention: Whisper content features
  |         |-> z1: (B, 64, L)
  |
  +---> [VAE Decoder (Oobleck), 凍結]
            |-> s_hat: (B, T')       変換された通常音声
```

| Component | Description | Params |
|-----------|-------------|--------|
| VAE (Oobleck) | 波形 <-> 潜在空間 (D=64, ~15.6Hz) | ~40M |
| DiT | 24 Transformer blocks, CFM velocity field | ~205M |
| Content Encoder | Whisper Base encoder, layer 5 | 74M (凍結) |
| Speaker Encoder | ECAPA-TDNN (speechbrain) | ~6M (凍結) |

## Setup

### Requirements

- Python >= 3.11
- CUDA 対応 GPU (学習時推奨)

### Installation

```bash
# uv を使用 (推奨)
uv sync

# または pip
pip install -e .

# 開発用依存 (pytest, ruff)
uv sync --group dev
```

## Usage

### Training

学習は 2 段階で行います。

#### Stage 1: VAE 学習

通常音声（HiFi-TTS-2）のみで VAE を学習します。

```bash
uv run python scripts/train_vae.py \
  --config configs/vae.json \
  --data-dir <path-to-hifitts2> \
  --output-dir outputs/vae
```

#### Feature Caching

DiT 学習の高速化のため、Whisper / ECAPA-TDNN / VAE の特徴量を事前キャッシュします。

```bash
# 合成ウィスパー生成
uv run python scripts/generate_whisper.py \
  --data-dir <path-to-hifitts2> \
  --output-dir <whisper-output>

# 特徴量キャッシュ
uv run python scripts/cache_features.py \
  --data-dir <path-to-data> \
  --output-dir <cache-path> \
  --vae-checkpoint <vae-ckpt>
```

#### Stage 2: DiT 学習

合成ウィスパー-通常音声ペアで DiT (Flow Matching) を学習します。

```bash
uv run python scripts/train_dit.py \
  --config configs/dit.json \
  --cache-dir <cache-path> \
  --output-dir outputs/dit
```

### Inference

```bash
uv run python scripts/inference.py \
  --input <whisper-audio-or-dir> \
  --output-dir outputs/converted \
  --vae-checkpoint outputs/vae/vae_ema_final.pt \
  --dit-checkpoint outputs/dit/dit_ema_final.pt \
  --num-steps 10
```

### Evaluation

```bash
uv run python scripts/evaluate.py \
  --input-dir outputs/converted \
  --reference-dir <reference-normal-audio> \
  --output results.json
```

#### Evaluation Metrics

| Metric | Tool | Direction |
|--------|------|-----------|
| WER-N | Whisper large-v3 | Lower is better |
| WER-W | Whisper tiny | Lower is better |
| UTMOS | SpeechMOS | Higher is better |
| DNSMOS | ONNX P.835 | Higher is better |
| SpkSim | ECAPA-TDNN cosine sim | Higher is better |

## Project Structure

```
FlowW2N/
├── configs/                        # Configuration files
│   ├── vae.json                    #   VAE model & training params
│   ├── dit.json                    #   DiT model & flow matching params
│   └── data.json                   #   Dataset & whisper synthesis params
├── src/floww2n/                    # Main package
│   ├── models/                     #   Model definitions
│   │   ├── vae.py                  #     Oobleck VAE (~40M params)
│   │   ├── dit.py                  #     DiffusionTransformer (~205M params)
│   │   ├── floww2n.py              #     FlowW2NModel (CFM loss + Euler sampling)
│   │   ├── content_encoder.py      #     Whisper Base layer 5 wrapper
│   │   └── speaker_encoder.py      #     ECAPA-TDNN wrapper
│   ├── training/                   #   Training utilities
│   │   ├── losses.py               #     Multi-res STFT + Discriminator + KL
│   │   ├── dataset.py              #     VAEDataset + DiTDataset + collate_fn
│   │   ├── train_vae.py            #     VAE training loop
│   │   └── train_dit.py            #     DiT training loop
│   ├── inference/                  #   Inference pipeline
│   │   ├── pipeline.py             #     FlowW2NPipeline (end-to-end)
│   │   └── sampler.py              #     Euler sampler
│   ├── data/                       #   Data processing
│   │   ├── whisper_synthesis.py    #     4 synthesis methods
│   │   └── preprocess.py           #     Data preprocessing
│   └── evaluation/                 #   Evaluation pipeline
│       ├── metrics.py              #     WER, UTMOS, DNSMOS, SpkSim
│       └── evaluate.py             #     Evaluation runner
├── scripts/                        # Entry-point scripts
│   ├── train_vae.py
│   ├── train_dit.py
│   ├── cache_features.py
│   ├── generate_whisper.py
│   ├── inference.py
│   ├── evaluate.py
│   └── preprocess_data.py
├── tests/                          # Tests (72 passing)
│   ├── test_vae.py                 #   9 tests
│   ├── test_dit.py                 #   9 tests
│   ├── test_losses.py              #   12 tests
│   ├── test_pipeline.py            #   16 tests
│   ├── test_dataset.py             #   7 tests
│   ├── test_whisper_synthesis.py    #   11 tests
│   └── test_training.py            #   8 tests
└── docs/                           # Documentation
    ├── 01_overview.md              #   Paper overview
    ├── 02_method.md                #   Method & formulas
    ├── 03_experiments_results.md   #   Experiments & results
    ├── 04_implementation_guide.md  #   Implementation guide
    ├── 05_plan_model.md            #   VAE + DiT implementation plan
    ├── 06_plan_data.md             #   Data pipeline plan
    ├── 07_plan_conditioning_inference.md  # Conditioning & inference plan
    └── 08_plan_overall.md          #   Overall project plan
```

## Development

### Tests

```bash
# Run all tests (72 tests)
uv run pytest tests/ -v

# Run specific test modules
uv run pytest tests/test_vae.py -v        # VAE (9)
uv run pytest tests/test_dit.py -v        # DiT (9)
uv run pytest tests/test_losses.py -v     # Losses (12)
uv run pytest tests/test_pipeline.py -v   # Pipeline (16)
uv run pytest tests/test_dataset.py -v    # Dataset (7)
uv run pytest tests/test_whisper_synthesis.py -v  # Whisper synthesis (11)
uv run pytest tests/test_training.py -v   # Training utils (8)
```

### Linting & Formatting

```bash
uv run ruff check src/ tests/ scripts/          # Lint
uv run ruff check --fix src/ tests/ scripts/     # Lint with auto-fix
uv run ruff format src/ tests/ scripts/          # Format
uv run ruff format --check src/ tests/ scripts/  # Format check
```

## Synthetic Whisper Generation

学習データは HiFi-TTS-2 の通常音声から以下の 4 手法で合成ウィスパーを生成します（各 25% の等確率）。

| Method | Description |
|--------|-------------|
| LPC-based Devoicing | LPC で声道フィルタを推定し、有声励振をノイズに置換 |
| Glottal Source Removal | 声門波源を除去して声帯振動なしの信号を生成 |
| Formant Bandwidth Modification | フォルマント帯域幅を増加させウィスパー共振特性をシミュレート |
| Praat Vocal Toolkit | Praat (parselmouth) によるウィスパー化処理 |

## Key Design Decisions

- **Gaussian noise prior**: Paired flow matching (z0=whisper, z1=normal) は時間的ミスアライメントで phoneme boundary blur が発生するため、z0 ~ N(0, I) + 外部条件付けを採用
- **Whisper layer 5**: `l* = argmax[Invariance(l) x CCA(l)]` の基準でドメイン不変性とコンテンツ情報量を最大化するレイヤーを選択
- **Cross-attention > Prepending**: コンテンツ特徴の注入は cross-attention が全指標で優位
- **bf16 mixed precision**: GradScaler 不要、gradient checkpointing で VRAM 削減

## References

- [FlowW2N Paper](https://arxiv.org/abs/2603.04296) - Ritter-Gutierrez et al., 2026
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) - VAE & DiT base implementation
- [DiT](https://arxiv.org/abs/2212.09748) - Peebles & Xie, ICCV 2023
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Lipman et al., ICLR 2023
- [OpenAI Whisper](https://arxiv.org/abs/2212.04356) - Radford et al., ICML 2023
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) - Desplanques et al., Interspeech 2020
- [HiFi-TTS-2](https://huggingface.co/datasets/nvidia/hifitts-2) - NVIDIA

## License

This is a research reproduction project. Please refer to the original paper and respective model licenses for usage terms.

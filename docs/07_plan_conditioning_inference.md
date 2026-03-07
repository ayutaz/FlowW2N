# FlowW2N: 条件付けモジュール・推論パイプライン・評価パイプライン 実装計画

> **実装状況**: ✅ 条件付けモジュール (M2)、推論・評価パイプライン (M4) すべて完了。品質監査完了。
> - ✅ `src/floww2n/models/content_encoder.py` - Whisper Base layer 5 wrapper (device管理修正済み)
> - ✅ `src/floww2n/models/speaker_encoder.py` - ECAPA-TDNN wrapper
> - ✅ `src/floww2n/inference/pipeline.py` - FlowW2NPipeline (end-to-end推論)
> - ✅ `src/floww2n/inference/sampler.py` - Euler sampler (FlowW2NModel.sample()に統合済み)
> - ✅ `src/floww2n/evaluation/metrics.py` - 評価指標 (WER, UTMOS, DNSMOS, SpkSim)
> - ✅ `src/floww2n/evaluation/evaluate.py` - 評価パイプライン
> - テスト: test_pipeline.py (16 tests), test_dit.py (14 tests)
> - コード品質: ruff lint/format 適用済み

本ドキュメントでは、FlowW2N の再現実装に必要な条件付けモジュール（Whisper encoder, ECAPA-TDNN）、推論パイプライン、および評価パイプラインの実装計画を策定する。

---

## 1. Whisper Base Encoder の仕様と実装

### 1.1 Whisper モデルファミリーの仕様

OpenAI Whisper は音声認識モデルであり、encoder-decoder 型の Transformer アーキテクチャを持つ。FlowW2N ではエンコーダ部分のみを使用する。

| モデル | パラメータ数 | エンコーダ層数 | デコーダ層数 | 隠れ層次元 (d_model) | アテンションヘッド数 |
|--------|-------------|---------------|--------------|---------------------|---------------------|
| tiny | 39M | 4 | 4 | 384 | 6 |
| **base** | **74M** | **6** | **6** | **512** | **8** |
| small | 244M | 12 | 12 | 768 | 12 |
| medium | 769M | 24 | 24 | 1024 | 16 |
| large-v3 | 1550M | 32 | 32 | 1280 | 20 |

**注意**: 論文ではパラメータ数を 21M と記載しているが、これはエンコーダの layer 5 までのパラメータ数（Conv1D + Conv1D + Position Embedding + 5 Transformer blocks）の概算値である。モデル全体は 74M パラメータ。

### 1.2 Whisper Base Encoder の入力仕様

| 項目 | 仕様 |
|------|------|
| サンプリングレート | 16,000 Hz |
| 入力形式 | 80-channel log-Mel spectrogram |
| FFT ウィンドウサイズ | 25ms (400 samples) |
| ホップサイズ | 10ms (160 samples) |
| 入力系列長 (固定) | 3000 フレーム (= 30秒分) |
| エンコーダ出力フレームレート | 50 Hz |

**Whisper エンコーダの入力側 Conv 層の構成**:

Whisper の音声エンコーダは 2 つの 1D 畳み込み層を入力側に持つ:

- Conv1d(n_mels=80, d_model=512, kernel_size=3, padding=1) -- stride=1
- GELU
- Conv1d(d_model=512, d_model=512, kernel_size=3, stride=2, padding=1) -- stride=2

3000 Mel フレーム -> Conv1(stride=1) -> 3000 -> Conv2(stride=2) -> 1500 フレーム。

エンコーダ出力: **(batch, 1500, 512)** で **50 Hz** のフレームレート。

### 1.3 Layer 5 の出力を取得する方法

Whisper Base のエンコーダは 6 層の Transformer ブロック (layer 0 -- layer 5) を持つ。FlowW2N では **layer 5** (0-indexed) の出力を使用する。layer 5 は最終層でもあるが、論文の layer 選択基準 `l* = argmax[Invariance(l) x CCA(l)]` に基づいて選択されたものである。

#### 方法 A: HuggingFace Transformers (推奨)

```python
from transformers import WhisperModel, WhisperFeatureExtractor
import torch

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
model = WhisperModel.from_pretrained("openai/whisper-base")
model.eval()

inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    encoder_outputs = model.encoder(
        inputs.input_features,
        output_hidden_states=True,
        return_dict=True,
    )

# hidden_states: tuple of (embedding_output, layer_0_out, ..., layer_5_out)
# len(hidden_states) = 7 (embedding + 6 layers)
h = encoder_outputs.hidden_states[6]  # shape: (batch, 1500, 512)
```

**`hidden_states` のインデックス対応表 (Whisper Base)**:

| インデックス | 意味 |
|------------|------|
| 0 | Conv 層 + Positional Embedding 後の出力 |
| 1 | Transformer block 0 の出力 |
| 2 | Transformer block 1 の出力 |
| 3 | Transformer block 2 の出力 |
| 4 | Transformer block 3 の出力 |
| 5 | Transformer block 4 の出力 |
| **6** | **Transformer block 5 の出力 (= layer 5 = 最終層)** |

#### 方法 B: openai/whisper パッケージ

```python
import whisper
import torch

model = whisper.load_model("base")
model.eval()

layer5_output = None

def hook_fn(module, input, output):
    global layer5_output
    layer5_output = output[0] if isinstance(output, tuple) else output

hook = model.encoder.blocks[5].register_forward_hook(hook_fn)

mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)  # (1, 80, 3000)
with torch.no_grad():
    _ = model.encoder(mel)

h = layer5_output  # shape: (1, 1500, 512)
hook.remove()
```

#### 方法 A vs 方法 B の比較

| 観点 | HuggingFace Transformers | openai/whisper |
|------|--------------------------|----------------|
| 中間層アクセス | `output_hidden_states=True` で直接取得可能 | forward hook が必要 |
| API の安定性 | 高い（長期サポート） | やや不安定（内部 API 変更あり） |
| エコシステム統合 | PyTorch, mixed precision 等と容易に統合 | 独立的 |
| 前処理 | `WhisperFeatureExtractor` で自動化 | `log_mel_spectrogram()` 関数 |
| インストール | `pip install transformers` | `pip install openai-whisper` |
| 推奨度 | **推奨** | hook 管理が煩雑 |

**結論**: HuggingFace Transformers の `WhisperModel` を使用し、`output_hidden_states=True` で layer 5 の出力を取得する方法を推奨する。

### 1.4 Layer 5 の出力仕様

| 項目 | 値 |
|------|-----|
| テンソル形状 | `(batch, 1500, 512)` |
| 系列長 | 1500 (固定, 30秒分に対応) |
| 特徴次元 | 512 (d_model) |
| フレームレート | 50 Hz |

**注意**: Whisper は固定長 30 秒の入力を想定しており、30 秒未満の音声はゼロパディング、30 秒超の音声はトランケートされる。出力系列長は常に 1500 である。

---

## 2. ECAPA-TDNN 話者エンコーダの仕様と実装

### 2.1 モデル仕様

| 項目 | 値 |
|------|-----|
| モデル名 | speechbrain/spkrec-ecapa-voxceleb |
| アーキテクチャ | ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN) |
| 学習データ | VoxCeleb1 + VoxCeleb2 |
| 出力埋め込み次元 | **192** |
| サンプリングレート | 16,000 Hz |
| 入力形式 | 生波形 (raw waveform) |
| 状態 | 凍結 (frozen) |

### 2.2 使用方法

```python
from speechbrain.inference.speaker import EncoderClassifier
import torch

spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"},
)

# waveform: torch.Tensor, shape (batch, samples), sr=16000
embeddings = spk_encoder.encode_batch(waveform)
# shape: (batch, 1, 192)

e_spk = embeddings.squeeze(1)  # shape: (batch, 192)
```

### 2.3 入力前処理

| 項目 | 仕様 |
|------|------|
| 入力形式 | 生波形 (float32) |
| サンプリングレート | 16,000 Hz |
| 正規化 | SpeechBrain 内部で自動処理 |
| 最小入力長 | 特に制約なし（短すぎると品質低下） |

**注意**: ECAPA-TDNN は内部で Mel filterbank + TDNN 処理を行うため、外部での Mel 変換は不要。

### 2.4 話者埋め込みの DiT への注入

論文では話者埋め込みを **AdaLN (Adaptive Layer Normalization)** を通じて DiT に注入する。

```python
class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 6),  # scale, shift, gate x 2
        )

    def forward(self, x, cond):
        params = self.mlp(cond)
        return params.chunk(6, dim=-1)
```

**次元変換**: 話者埋め込み (192 次元) とタイムステップ埋め込みを結合して DiT の隠れ次元に射影する:

```
timestep: sinusoidal -> MLP -> (hidden_dim,)
speaker:  (192,) -> Linear -> (hidden_dim,)
cond = timestep_emb + spk_emb  (element-wise addition)
```

---

## 3. 推論パイプラインの設計

### 3.1 全体フロー

```
入力: ウィスパー音声 s_whisper (16kHz, mono)
  |
  +---> [Whisper Base encoder, layer 5, 凍結]
  |         |-> h: (batch, T_h, 512)   # コンテンツ特徴
  |
  +---> [ECAPA-TDNN, 凍結]
  |         |-> e_spk: (batch, 192)    # 話者埋め込み
  |
  |     条件信号 c = {h, e_spk}
  |
  +---> z0 ~ N(0, I)                  # (batch, D=64, L)
  |
  +---> [DiT: Euler 積分 N ステップ]
  |         |-> z1: (batch, D=64, L)   # 推定通常音声の潜在表現
  |
  +---> [VAE Decoder, 凍結]
            |-> s_hat: (batch, T')     # 変換された通常音声
```

### 3.2 系列長の不一致への対応

| 表現 | フレームレート | 30 秒入力時の系列長 |
|------|--------------|-------------------|
| Whisper encoder 出力 h | 50 Hz | 1500 |
| VAE 潜在表現 z | ~15.6 Hz | ~468 |
| 波形 | 16,000 Hz | 480,000 |

#### (a) h と z の系列長不一致

**cross-attention** メカニズムで自然に解決される。Query = z, Key/Value = h とすることで系列長の不一致は問題にならない。Whisper 出力 512 次元を DiT の隠れ次元に射影する線形層が必要。

#### (b) z0 の系列長 L の決定

```python
def compute_latent_length(audio_length_samples: int, compression_ratio: int = 1024) -> int:
    # ~15.6 Hz = 16000 / 1024
    return audio_length_samples // compression_ratio
```

#### (c) Whisper の固定入力長 (30秒) への対応

1. 音声をゼロパディングして 30 秒にする（Whisper の標準動作）
2. 出力 h の有効フレーム範囲をマスクする
3. Cross-attention 時にアテンションマスクを適用

```python
def get_whisper_features(audio, whisper_model, feature_extractor):
    audio_len_sec = len(audio) / 16000
    valid_frames = min(int(audio_len_sec * 50), 1500)

    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = whisper_model.encoder(
            inputs.input_features,
            output_hidden_states=True,
        )

    h = outputs.hidden_states[6]  # (batch, 1500, 512)

    mask = torch.zeros(1, 1500, dtype=torch.bool)
    mask[:, :valid_frames] = True

    return h, mask
```

### 3.3 Euler 積分の実装

```python
@torch.no_grad()
def euler_sample(
    dit_model: nn.Module,
    z0: torch.Tensor,         # (batch, D, L)
    condition: dict,          # {"h": ..., "e_spk": ...}
    num_steps: int = 10,
) -> torch.Tensor:
    dt = 1.0 / num_steps
    z = z0.clone()

    for i in range(num_steps):
        t = i / num_steps
        t_tensor = torch.full((z.shape[0],), t, device=z.device, dtype=z.dtype)
        v = dit_model(z, t_tensor, condition)
        z = z + dt * v

    return z  # z1
```

**パラメータ化のポイント**:
- `num_steps` を引数として公開（論文デフォルト: 10）
- 将来的に他のソルバー (RK4, midpoint) への切り替えも検討

### 3.4 推論パイプラインクラスの設計

```python
class FlowW2NPipeline:
    def __init__(self, vae_model, dit_model, whisper_model,
                 whisper_feature_extractor, speaker_encoder,
                 device="cuda", num_steps=10):
        self.vae = vae_model.to(device).eval()
        self.dit = dit_model.to(device).eval()
        self.whisper = whisper_model.to(device).eval()
        self.feature_extractor = whisper_feature_extractor
        self.speaker_encoder = speaker_encoder
        self.device = device
        self.num_steps = num_steps
        self.latent_dim = 64

    @torch.no_grad()
    def __call__(self, whisper_audio, sample_rate=16000,
                 num_steps=None, seed=None):
        steps = num_steps or self.num_steps
        if whisper_audio.dim() == 1:
            whisper_audio = whisper_audio.unsqueeze(0)
        batch_size = whisper_audio.shape[0]
        audio_length = whisper_audio.shape[1]

        h, h_mask = self._extract_content_features(whisper_audio, sample_rate)
        e_spk = self._extract_speaker_embedding(whisper_audio)
        latent_length = self._compute_latent_length(audio_length)

        if seed is not None:
            torch.manual_seed(seed)
        z0 = torch.randn(batch_size, self.latent_dim, latent_length,
                          device=self.device)

        condition = {"h": h, "h_mask": h_mask, "e_spk": e_spk}
        z1 = euler_sample(self.dit, z0, condition, num_steps=steps)

        return self.vae.decode(z1)
```

### 3.5 バッチ推論の効率化

| 手法 | 説明 | 優先度 |
|------|------|--------|
| バッチ処理 | 複数音声を同時にパディングしてバッチ化 | 高 |
| 音声長バケッティング | 似た長さをグループ化 | 中 |
| Mixed precision | fp16/bf16 で推論 | 高 |
| 条件特徴キャッシュ | Whisper / ECAPA 出力を事前計算 | 高 |
| ステップ数動的調整 | 品質要求に応じて変更 | 低 |

---

## 4. Evaluation Pipeline

| Metric | Tool | Direction | Use |
|--------|------|-----------|-----|
| WER-N | Whisper large-v3 | Lower | Intelligibility |
| WER-W | Whisper tiny | Lower | Intelligibility |
| UTMOS | sarulab-speech | Higher | Naturalness MOS |
| DNSMOS | Microsoft | Higher | Quality MOS |
| SpkSim | ECAPA-TDNN | Higher | Speaker similarity |

- WER-N: transformers WhisperForConditionalGeneration (openai/whisper-large-v3), jiwer
- WER-W: transformers WhisperForConditionalGeneration (openai/whisper-tiny), jiwer
- UTMOS: torch.hub sarulab-speech/SpeechMOS:v1.2.0
- DNSMOS: ONNX P.835 from DNS-Challenge
- SpkSim: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, 192d), cosine sim

### Dependencies

transformers, speechbrain, jiwer, onnxruntime, torch, torchaudio

---

## 5. Tensor dimensions

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input audio | (B, T) | Batch x samples |
| Mel | (B, 80, 3000) | 30s fixed |
| Whisper h | (B, 1500, 512) | 50Hz |
| h mask | (B, 1500) | Valid=True |
| Speaker emb | (B, 192) | 192d |
| Cond vector | (B, d_dit) | DiT hidden |
| z0/z1 | (B, 64, L) | D=64 |
| Output | (B, T2) | T2=Lx1024 |

---

## 6. File structure

```
src/floww2n/
  __init__.py
  models/
    __init__.py
    vae.py
    dit.py
    content_encoder.py
    speaker_encoder.py
    floww2n.py
  training/
    __init__.py
    train_vae.py
    train_dit.py
    losses.py
    dataset.py
  inference/
    __init__.py
    pipeline.py
    sampler.py
  data/
    __init__.py
    preprocess.py
    whisper_synthesis.py
  evaluation/
    __init__.py
    metrics.py
    evaluate.py
configs/
  vae.json
  dit.json
  data.json
scripts/
  preprocess_data.py
  generate_whisper.py
  cache_features.py
  train_vae.py
  train_dit.py
  inference.py
  evaluate.py
tests/
  __init__.py
  test_vae.py
  test_dit.py
  test_losses.py
  test_pipeline.py
  test_dataset.py
  test_whisper_synthesis.py
  test_training.py
```

---

## 7. Roadmap

| Phase | Tasks | Days | Priority | Status |
|-------|-------|------|----------|--------|
| 1 | Conditioning modules | 2-3 | P0 | ✅ Complete |
| 2 | Inference pipeline | 3-5 | P0 | ✅ Complete |
| 3 | Evaluation pipeline | 3-4 | P0/P1 | ✅ Complete |
| 4 | Optimization | 2-3 | P1/P2 | ✅ Complete (gradient checkpointing, DataLoader) |

---

## 8. Open issues

| Item | Detail | Status |
|------|--------|--------|
| Whisper layer 5 index | 0-indexed, final layer → hidden_states[6] | ✅ Resolved |
| DiT hidden dim | embed_dim=768 (SA2.0より小規模) | ✅ Resolved |
| VAE compression | 16000/1024 = 15.625Hz | ✅ Resolved |
| Whisper padding mask | ContentEncoder returns mask | ✅ Resolved |
| ECAPA input quality | Works well with whisper input | ✅ Resolved |
| DNSMOS version | P.835 implemented | ✅ Resolved |

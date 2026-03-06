# FlowW2N: コアモデル（VAE + DiT）実装計画

本ドキュメントでは、FlowW2N の再現実装に必要なコアモデル（VAE と DiT）の実装計画を策定する。stable-audio-tools の調査結果に基づき、Oobleck VAE と DiffusionTransformer の適応方針を詳述する。

---

## 1. stable-audio-tools 調査結果

### 1.1 リポジトリ構造

stable-audio-tools (Stability-AI/stable-audio-tools) の主要ファイル:

- models/autoencoders.py: OobleckEncoder/Decoder, AudioAutoencoder
- models/dit.py: DiffusionTransformer
- models/transformer.py: ContinuousTransformer, TransformerBlock, AdaLN
- models/bottleneck.py: VAEBottleneck
- models/blocks.py: FourierFeatures, SnakeBeta
- training/autoencoders.py: AutoencoderTrainingWrapper
- training/losses/auraloss.py: MultiResolutionSTFTLoss

### 1.2 Oobleck VAE の実装箇所

| ファイル | クラス | 役割 |
|---------|--------|------|
| autoencoders.py | OobleckEncoder | エンコーダ。ResidualUnit+EncoderBlock |
| autoencoders.py | OobleckDecoder | デコーダ。DecoderBlock |
| autoencoders.py | AudioAutoencoder | エンコーダ+デコーダ+ボトルネック統合 |
| autoencoders.py | EncoderBlock | ResidualUnit(d=1,3,9)+ストライドConv |
| bottleneck.py | VAEBottleneck | mean/scale分割、KL計算 |

重要: VAEBottleneckはencoder出力(latent_dim=128)を2分割しmean/scaleとする。encoder.latent_dim=128, decoder.latent_dim=64でFlowW2NのD=64と整合。

SA2.0 VAE: strides=[2,4,4,8,8], channels=128, c_mults=[1,2,4,8,16], ratio=2048, sr=44100, use_snake=true

### 1.3 DiT の実装箇所

| ファイル | クラス | 役割 |
|---------|--------|------|
| dit.py | DiffusionTransformer | DiTメイン。タイムステップ埋め込み、条件付け投影 |
| transformer.py | ContinuousTransformer | Transformerスタック。RoPE、global_cond_embedder |
| transformer.py | TransformerBlock | self-attn+cross-attn+FF+AdaLN |
| transformer.py | Attention | Multi-head attention (Flash/FlexAttention) |
| transformer.py | FeedForward | SwiGLU FFN |
| blocks.py | FourierFeatures | タイムステップFourier特徴量 |

### 1.4 AdaLN の実装詳細

TransformerBlock.forward()内で実装。
- ContinuousTransformer.global_cond_embedder: global_cond(embed_dim) -> Linear -> SiLU -> Linear -> 6*dim
- 6*dim = (scale, shift, gate) x 2セット（self-attn用 + ff用）
- 各TransformerBlock: to_scale_shift_gate = nn.Parameter(randn(6*dim)/dim**0.5)
- AdaLN: (to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6) -> scale/shift/gate
- Self-Attn: x = LN(x)*(1+scale)+shift -> attn -> x*sigmoid(1-gate) -> residual
- Cross-Attn: AdaLN変調なし、x + cross_attn(LN(x), context)
- FF: 同様のAdaLN変調

FlowW2NでのAdaLN: global_cond = timestep_embed + speaker_embed (加算)
- timestep: FourierFeatures -> MLP -> embed_dim
- speaker: ECAPA-TDNN(192d) -> MLP -> embed_dim

### 1.5 Cross-Attention

cross_attn_cond -> to_cond_embed(Linear->SiLU->Linear) -> context -> 各TransformerBlockのcross-attention

### 1.6 SA2.0 DiT設定

io_channels=64, embed_dim=1536, depth=24, num_heads=24, cond_token_dim=768, global_cond_dim=1536

### 1.7 依存関係

torch>=2.5.1, torchaudio>=2.5.1, pytorch_lightning==2.1.0, einops, alias-free-torch==0.0.6, auraloss==0.4.0, ema-pytorch==0.2.3, flash-attn(optional)

---

## 2. VAE 実装計画

### 2.1 フレームレートとサンプリングレート分析

論文仕様: D=64, 約15.6Hz

| SR | strides | ratio | Hz | 備考 |
|:-:|:-:|:-:|:-:|:--|
| 44100 | [2,4,4,8,8] | 2048 | 21.5 | SA2.0 |
| 24000 | [2,4,4,8,8] | 2048 | 11.7 | 低い |
| **16000** | **[4,4,8,8]** | **1024** | **15.625** | **一致** |
| 32000 | [2,4,4,8,8] | 2048 | 15.625 | 一致 |

推奨: 16kHz+strides=[4,4,8,8](ratio=1024)
- Whisper内部SR一致、計算効率最高

### 2.2 VAE設定

SA2.0からの変更: sr 44100->16000, io 2->1, c_mults 5段->4段, strides変更, ratio 2048->1024

### 2.3 利用方針

推奨: 方針B（必要モジュールのみコピー）。Python 3.13非互換のため。

### 2.4 損失関数

| 損失 | 実装 | 重み |
|:--|:--|:--|
| Multi-res STFT | auraloss.py | 1.0 |
| Discriminator | discriminators.py | adv:0.1, feat:5.0 |
| KL | VAEBottleneck | 1e-4 |

### 2.5 学習

HiFi-TTS-2(24kHz)->16kHz->mono->65536samples(4.1s)->Enc->VAE->Dec->Loss. 80K steps, batch 256, AdamW(1.5e-4), EMA.

### 2.6 検証

STFT loss収束、聴取比較、フレームレート=in/1024、潜在shape=(B,64,L)

---

## 3. DiT 実装計画

### 3.1 構成

| 機能 | SA2.0 | FlowW2N |
|:--|:--|:--|
| 条件付け | CLAP text(768d) | Whisper layer5(512d) |
| Global | seconds | speaker(192d)+timestep |
| Objective | v prediction | rectified_flow(CFM) |
| io/depth | 64/24 | 64/24 |

### 3.2 設定

io_channels=64, embed_dim=768(論文未明記,初期値), depth=24, num_heads=12, cond_token_dim=512(Whisper Base), global_cond_dim=768, global_cond_type=adaLN, diffusion_objective=rectified_flow

### 3.3 話者埋め込み注入(AdaLN)

FlowW2NModel: speaker_proj(192d->embed_dim) -> dit.forward(global_embed=speaker_embed). 内部でtimestep_embedと加算されAdaLNに渡る。

### 3.4 Whisper特徴注入(Cross-Attention)

Whisper Base layer5: (B,T_w,512) 50Hz. to_cond_embed(512->embed_dim)で投影。フレームレート差(50Hz vs 15.6Hz)はcross-attentionが吸収。

### 3.5 Flow Matching学習

z0~N(0,I), t~U[0,1], zt=(1-t)*z0+t*z1, target=z1-z0, loss=MSE(v_theta(zt,t,c), target)

### 3.6 推論(Euler積分N=10)

z~N(0,I), dt=0.1, z+=dt*v_theta(z,t,c) x10回, audio=vae.decode(z)

---

## 4. ディレクトリ構成

configs/ (vae/dit/data JSON), src/floww2n/ (models/ training/ inference/ data/ evaluation/), scripts/, tests/

models/: vae.py, dit.py, content_encoder.py, speaker_encoder.py, floww2n.py
training/: train_vae.py, train_dit.py, losses.py, dataset.py
inference/: convert.py
data/: whisper_synthesis.py, preprocess.py
evaluation/: metrics.py, evaluate.py

---

## 5. ロードマップ

Phase1(W1-2): 環境構築+VAE学習
Phase2(W2-3): Whisper/ECAPA条件付けモジュール
Phase3(W3-5): DiT+FlowMatching+合成ウィスパー
Phase4(W5-6): 評価(WER,UTMOS,DNSMOS,SpkSim)

---

## 6. 技術的注意点

- SR最終決定: 16kHz+[4,4,8,8]第一候補、不十分なら32kHz+[2,4,4,8,8]
- embed_dim: 768初期値(512:80M, 768:170M, 1024:290M, 1536:600M)
- VAE latent_dim: encoder=128(mean+scale), decoder=64, AE=64
- Whisper layer: 0-indexed layer5=最終層。hidden_states[6]で取得
- DiT入力: (B,C,T)チャンネルファースト、VAE出力そのまま
- Python: 3.11/3.12必須(3.13非互換)
- 計算: batch256=実16x累積16、bf16/fp16混合精度

---

## 7. 依存関係案

torch>=2.1.0, torchaudio, pytorch-lightning, transformers>=4.36.0, speechbrain>=1.0.0, einops, alias-free-torch==0.0.6, auraloss==0.4.0, ema-pytorch>=0.2.3, datasets, librosa, soundfile, jiwer, wandb, safetensors. Python>=3.11,<3.13

---

## 8. まとめ

1. VAE: OobleckEncoder/Decoder+VAEBottleneck。strides/sr調整でD=64,15.6Hz
2. DiT: DiffusionTransformer。adaLNで話者+timestep、cross-attentionでWhisper
3. Flow Matching: rectified_flow。z0~N(0,I)線形補間、速度場MSE

リスク: SR/stride不確実性->16kHz第一候補, embed_dim->768初期値, Python互換->3.11/3.12, リソース->勾配累積+混合精度

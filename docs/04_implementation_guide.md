# FlowW2N: 再現実装ガイド

本ドキュメントでは、FlowW2N（Whispered-to-Normal Speech Conversion via Flow-Matching）の再現実装に必要な情報を体系的に整理する。

---

## 1. 必要なコンポーネント一覧

FlowW2N は以下の5つの主要コンポーネントから構成される。

### 1.1 VAE（Oobleck Encoder-Decoder）

| 項目 | 詳細 |
|------|------|
| ソース | stable-audio-tools [22] |
| アーキテクチャ | Oobleck encoder-decoder |
| 入力 | 波形 s in R^T |
| 出力（エンコーダ） | 潜在表現 z = E(s) in R^{D x L}、D=64、L=T/r |
| フレームレート | 約15.6 Hz |
| 役割 | 波形を潜在空間に圧縮（エンコーダ E）、潜在表現から波形を復元（デコーダ D） |
| 学習データ | normal speech のみ（HiFi-TTS-2） |
| 学習ステップ | 80,000 ステップ |
| バッチサイズ | 256 |
| 損失関数 | multi-resolution STFT loss + multi-scale discriminator losses + KL divergence regularization |

**実装ノート**: stable-audio-tools リポジトリから Oobleck VAE の実装を取得し、HiFi-TTS-2 の normal speech データで再学習する。ささやき音声は VAE の学習には使用しない。

### 1.2 DiT（Diffusion Transformer）

| 項目 | 詳細 |
|------|------|
| ソース | DiT [23]、stable-audio-tools [22] からの適応 |
| ブロック数 | 24 transformer blocks |
| タイムステップ注入 | AdaLN（adaptive layer normalization）で t を注入 |
| 話者埋め込み注入 | AdaLN で e_spk を注入 |
| コンテンツ特徴注入 | cross-attention（最良）または prepending |
| 条件信号 | c = {e_spk, h} |

**実装ノート**: DiT は元々画像生成向けに設計されたアーキテクチャであるが、stable-audio-tools での音声向け適応版をベースとする。条件付けは cross-attention が最良の結果を示している。

### 1.3 Content Encoder（Whisper Base）

| 項目 | 詳細 |
|------|------|
| モデル | OpenAI Whisper Base encoder |
| 使用レイヤー | layer 5（l*=5、Whisper Base の場合の最適レイヤー） |
| パラメータ数 | 21M |
| 学習時の状態 | 凍結（frozen） |

**実装ノート**: Whisper Base の encoder 部分のみを使用する。decoder は不要。layer 5 の出力がコンテンツ情報量とクロスドメイン不変性のバランスが最も良い。レイヤー番号は 0-indexed か 1-indexed かに注意すること。

### 1.4 Speaker Encoder（ECAPA-TDNN）

| 項目 | 詳細 |
|------|------|
| モデル | speechbrain/spkrec-ecapa-voxceleb |
| 学習時の状態 | 凍結（frozen） |
| 注入方法 | AdaLN で DiT に注入 |

**実装ノート**: SpeechBrain の事前学習済みモデルをそのまま使用する。話者埋め込みは推論時のターゲット話者を指定するために利用される。

### 1.5 合成ウィスパー生成モジュール

学習データの構築に使用する。以下の4手法を等確率（各25%）でランダムにサンプリングし、通常音声からささやき音声を合成する。

| 手法 | ソース | 説明 |
|------|--------|------|
| (1) LPC-based devoicing | github.com/zeta-chicken/toWhisper | LPC分析に基づく無声化 |
| (2) Glottal source removal | Lin et al., ASRU 2023 [5] | 声門音源の除去 |
| (3) Formant bandwidth modification | Lin et al., ASRU 2023 [5] | フォルマント帯域幅の変更 |
| (4) Praat Vocal Toolkit | praatvocaltoolkit.com/whisper.html | Praat による合成ウィスパー生成 |

**実装ノート**: 4手法を組み合わせることで、合成ウィスパーの多様性を確保し、モデルの汎化性能を向上させている。各手法は独立に実装可能。`generate_whisper.py --num-workers` でマルチプロセス並列処理が可能。

---

## 2. 学習パイプライン

学習は2段階で構成される。

### 2.1 Stage 1: VAE の学習

```
入力: normal speech 波形 s (HiFi-TTS-2)
  |
  v
[Oobleck Encoder E] --> 潜在表現 z in R^{64 x L}
  |
  v
[Oobleck Decoder D] --> 復元波形 s_hat
  |
  v
損失計算:
  L_vae = L_multi-res-STFT(s, s_hat)
        + L_multi-scale-disc(s, s_hat)
        + L_KL(z)
```

- 学習データ: HiFi-TTS-2 の normal speech のみ
- ステップ数: 80,000
- バッチサイズ: 256

### 2.2 Stage 2: DiT（Flow Matching モデル）の学習

```
[Step A: 合成ウィスパー生成]
normal speech s_normal --> 合成ウィスパー生成 (4手法から等確率選択) --> s_whisper

[Step B: 潜在表現への変換（VAE エンコーダは凍結）]
s_normal  --> E(s_normal)  = z1  (ターゲット)
s_whisper --> Whisper Base encoder (layer 5, 凍結) --> h (コンテンツ特徴)
s_normal  --> ECAPA-TDNN (凍結) --> e_spk (話者埋め込み)

[Step C: Flow Matching 学習]
z0 ~ N(0, I)           # ノイズサンプル
t  ~ U[0, 1]           # タイムステップ
zt = (1 - t) * z0 + t * z1   # 線形補間パス
c  = {e_spk, h}        # 条件信号

損失:
  L_CFM = E_{t, z0~N(0,I), z1} [ || v_theta(zt, t, c) - (z1 - z0) ||^2 ]
```

**学習の要点**:
- VAE は Stage 1 で学習済みのものを凍結して使用する
- DiT の学習には合成ウィスパー-通常音声ペアのみを使用する
- 実ウィスパー音声は学習時に一切使用しない
- Whisper Base encoder と ECAPA-TDNN は事前学習済みモデルを凍結して使用する

### 2.3 学習データフロー図

```
HiFi-TTS-2 (normal speech)
    |
    +-----> VAE 学習 (Stage 1)
    |
    +-----> 合成ウィスパー生成 (4手法, 等確率)
    |           |
    |           v
    |       s_whisper (合成ウィスパー)
    |           |
    |           +---> Whisper Base (layer 5, 凍結) ---> h
    |
    +-----> ECAPA-TDNN (凍結) ---> e_spk
    |
    +-----> VAE Encoder (凍結) ---> z1
    |
    +-----> DiT 学習 (Stage 2): v_theta(zt, t, c) を学習
```

---

## 3. 推論パイプライン

```
入力: ささやき音声 s_whisper（実ウィスパーまたは合成ウィスパー）

[Step 1: 条件信号の抽出]
s_whisper --> Whisper Base encoder (layer 5) --> h (コンテンツ特徴)
s_whisper --> ECAPA-TDNN --> e_spk (話者埋め込み)
c = {e_spk, h}

[Step 2: Flow Matching によるサンプリング]
z0 ~ N(0, I)  # ガウスノイズからスタート

Euler 積分 (N=10 ステップ):
  dt = 1/N = 0.1
  for i in 0, 1, ..., N-1:
      t_i = i / N
      z_{i+1} = z_i + dt * v_theta(z_i, t_i, c)
  z1 = z_N  # 最終的な潜在表現

[Step 3: 波形復元]
z1 --> VAE Decoder D --> s_hat (通常音声の推定)

出力: s_hat（変換された通常音声）
```

**推論の要点**:
- サンプリングは z0 ~ N(0, I) から開始し、学習された速度場 v_theta に沿って積分する
- Euler 法による離散化で N=10 ステップのみで十分な品質を達成
- 全ての条件付けモデル（Whisper Base, ECAPA-TDNN）と VAE Decoder は凍結状態で使用

**Heun ソルバー（2次精度）**:
- Euler ソルバーの代替として Heun（改良 Euler）ソルバーを実装
- 各ステップで 2 回の関数評価（NFE）を行う2次精度の ODE ソルバー
- ステップ数が少ない場合（N<10）に Euler より高品質な結果を得られる
- `FlowW2NModel.sample(solver="euler"|"heun")` で切り替え可能

---

## 4. 使用するデータセットと事前学習モデル

### 4.1 学習データセット

| データセット | 用途 | ソース |
|-------------|------|--------|
| HiFi-TTS-2 | VAE 学習、DiT 学習（合成ウィスパー生成元） | huggingface.co/datasets/nvidia/hifitts-2 [27] |

### 4.2 評価データセット

| データセット | 詳細 | 規模 |
|-------------|------|------|
| wTIMIT | TIMIT のささやき版、US ダイアレクトテストセット | 50 話者、1,404 発話、約2時間 |
| CHAINS | ささやき音声コーパス | 36 話者、約1,332 ウィスパー発話、約2.45 時間 |

### 4.3 事前学習モデル

| モデル | 用途 | ソース/識別子 |
|--------|------|--------------|
| Whisper Base | コンテンツエンコーダ（layer 5） | OpenAI Whisper Base（openai/whisper-base） |
| ECAPA-TDNN | 話者エンコーダ | speechbrain/spkrec-ecapa-voxceleb |
| Oobleck VAE | 潜在空間変換（自前学習が必要） | stable-audio-tools [22] のアーキテクチャ |

### 4.4 評価用モデル

| モデル | 用途 | 指標 |
|--------|------|------|
| NeMo FastConformer | 音声認識 | WER |
| Whisper tiny | 音声認識 | WER |
| UTMOS | 自然さ推定 | MOS スコア |
| DNSMOS | 音質推定 | MOS スコア |
| Resemblyzer | 話者類似度 | SpkSim |

---

## 5. ハイパーパラメータと設定値

### 5.1 VAE 関連

| パラメータ | 値 |
|-----------|-----|
| 潜在次元 D | 64 |
| フレームレート | 約 15.6 Hz |
| ダウンサンプリング比 r | T/L（入力長 T と潜在長 L の比） |
| 学習ステップ数 | 80,000 |
| バッチサイズ | 256 |
| 学習データ | normal speech のみ |
| 損失関数 | multi-resolution STFT + multi-scale discriminator + KL divergence |

### 5.2 DiT 関連

| パラメータ | 値 |
|-----------|-----|
| Transformer ブロック数 | 24 |
| タイムステップ条件付け | AdaLN |
| 話者埋め込み条件付け | AdaLN |
| コンテンツ特徴条件付け | cross-attention（推奨） |
| 条件信号 | c = {e_spk, h} |

### 5.3 Flow Matching 関連

| パラメータ | 値 |
|-----------|-----|
| ノイズ分布 | z0 ~ N(0, I) |
| タイムステップ分布 | t ~ U[0, 1] |
| 補間パス | zt = (1 - t) * z0 + t * z1（線形） |
| CFM 損失 | L = E[ \|\| v_theta(zt, t, c) - (z1 - z0) \|\|^2 ] |

### 5.4 推論関連

| パラメータ | 値 |
|-----------|-----|
| サンプリング手法 | Euler 積分（デフォルト）/ Heun 積分（2次精度） |
| ステップ数 N | 10（Euler）。Heun は少ないステップで同等品質 |
| 初期分布 | z0 ~ N(0, I) |

### 5.5 Content Encoder 関連

| パラメータ | 値 |
|-----------|-----|
| モデル | Whisper Base encoder |
| 最適レイヤー | layer 5 (l*=5) |
| パラメータ数 | 21M |
| 状態 | 凍結 |

### 5.6 合成ウィスパー生成

| パラメータ | 値 |
|-----------|-----|
| 手法数 | 4 |
| サンプリング確率 | 各手法 25%（等確率） |

---

## 6. 最適化

再現実装では、学習・推論の効率化のために以下の最適化を実装している。

### 6.1 学習最適化

| 最適化項目 | 設定方法 | 説明 |
|-----------|---------|------|
| torch.compile() | `train_cfg["compile"]: true` | モデルの JIT コンパイルによる高速化 |
| CUDA 最適化 | `setup_cuda_optimizations()` | cudnn.benchmark、TF32、matmul precision の自動設定 |
| Fused AdamW | 自動（CUDA 環境） | `fused=True` による最適化された AdamW オプティマイザ |
| zero_grad 最適化 | 自動 | `zero_grad(set_to_none=True)` による勾配メモリの効率化 |
| Gradient checkpointing | `dit.json: "gradient_checkpointing": true` | メモリ使用量の削減（計算時間とのトレードオフ） |
| 非同期チェックポイント保存 | 自動 | ThreadPoolExecutor による非同期保存、max_keep=3 |

### 6.2 推論最適化

| 最適化項目 | 説明 |
|-----------|------|
| Weight norm 除去 | `AudioAutoencoder.remove_weight_norm()` で推論速度向上 |
| Heun ソルバー | 2次精度ソルバー。少ないステップ数で高品質（2 NFE/step） |
| ピン留めメモリ | パイプラインで pinned memory を使用した非ブロッキング GPU 転送 |

### 6.3 データ処理最適化

| 最適化項目 | 設定方法 | 説明 |
|-----------|---------|------|
| バッチ特徴量キャッシュ | `cache_features.py --batch-size` | バッチ処理による特徴量抽出の高速化 |
| マルチプロセス合成ウィスパー | `generate_whisper.py --num-workers` | 複数プロセスによる並列合成 |

---

## 7. 多言語対応

### 7.1 言語埋め込みアーキテクチャ

FlowW2NModel レベルで言語埋め込みを実装。DiT 内部ではなく、FlowW2NModel の条件付けで統合される。

```
global_cond = speaker_proj(e_spk) + language_emb(language_id)
```

- `speaker_proj`: ECAPA-TDNN 出力 (192d) を embed_dim に投影
- `language_emb`: `nn.Embedding(num_languages, embed_dim)` による言語埋め込み
- 加算された `global_cond` が DiT に AdaLN 経由で渡される

### 7.2 設定

`dit.json` の model セクションに以下のフィールドを追加:

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `num_languages` | int | 言語数（例: 2） |
| `languages` | list[str] | 言語識別子のリスト（例: `["en", "ja"]`） |

### 7.3 後方互換性

- `num_languages <= 1` または `language_id=None` の場合、言語埋め込みは無効化される
- 既存の単一言語モデル・設定との完全な後方互換性を維持

---

## 8. 参考文献リスト（実装に必要なもの）

以下は再現実装に直接関連する参考文献の一覧である。

### 8.1 コアアーキテクチャ関連

| 番号 | 文献 | 関連コンポーネント |
|------|------|-------------------|
| [22] | Z. Evans et al., "Fast timing-conditioned latent audio diffusion," ICML 2024 | stable-audio-tools（VAE, DiT のベース実装） |
| [23] | W. Peebles and S. Xie, "Scalable diffusion models with transformers," ICCV 2023 | DiT アーキテクチャの原論文 |

### 8.2 Flow Matching 理論

| 番号 | 文献 | 関連内容 |
|------|------|---------|
| [17] | Y. Lipman et al., "Flow matching for generative modeling," ICLR 2023 | Flow matching の理論的基礎 |
| [20] | X. Liu et al., "Flow straight and fast: Learning to generate and transfer data with rectified flow," ICLR 2023 | Rectified flow（線形補間パスの理論的背景） |

### 8.3 条件付けモデル

| 番号 | 文献 | 関連コンポーネント |
|------|------|-------------------|
| [30] | A. Radford et al., "Robust speech recognition via large-scale weak supervision," ICML 2023 | Whisper モデル（コンテンツエンコーダ） |
| [24] | B. Desplanques et al., "ECAPA-TDNN," Interspeech 2020 | 話者エンコーダ |

### 8.4 合成ウィスパー生成

| 番号 | 文献 | 関連内容 |
|------|------|---------|
| [5] | Z. Lin et al., "Improving whispered speech recognition performance using pseudo-whispered based data augmentation," ASRU 2023 | Glottal source removal、Formant bandwidth modification の手法 |

### 8.5 データセットとツール

| 番号 | 文献/リソース | 関連内容 |
|------|-------------|---------|
| [27] | NVIDIA, "HiFi-TTS-2 dataset" (huggingface.co/datasets/nvidia/hifitts-2) | 学習データセット |
| [25] | M. Bain et al., "WhisperX," Interspeech 2023 | 強制アライメント（必要に応じて使用） |

### 8.6 実装で参照すべき外部リポジトリ/ツール

| リソース | URL | 用途 |
|---------|-----|------|
| stable-audio-tools | github.com/Stability-AI/stable-audio-tools | VAE（Oobleck）と DiT の実装ベース |
| toWhisper | github.com/zeta-chicken/toWhisper | LPC-based devoicing |
| Praat Vocal Toolkit | praatvocaltoolkit.com/whisper.html | 合成ウィスパー生成 |
| SpeechBrain ECAPA-TDNN | huggingface.co/speechbrain/spkrec-ecapa-voxceleb | 話者エンコーダ |
| OpenAI Whisper | github.com/openai/whisper | コンテンツエンコーダ |

---

## 付録: 実装チェックリスト

再現実装を進める際の確認項目を以下にまとめる。

### A. 環境構築
- [x] Python 環境のセットアップ
- [x] stable-audio-tools からの VAE/DiT アーキテクチャ移植
- [x] OpenAI Whisper ラッパー (ContentEncoder) 実装
- [x] SpeechBrain ECAPA-TDNN ラッパー (SpeakerEncoder) 実装
- [x] 合成ウィスパー生成モジュール (4手法) 実装
- [x] pyproject.toml に全依存定義、ruff lint/format 設定

### B. データ準備
- [ ] HiFi-TTS-2 データセットのダウンロード
- [ ] wTIMIT データセットの準備（評価用）
- [ ] CHAINS データセットの準備（評価用）
- [x] 合成ウィスパー生成コード完成（4手法、等確率サンプリング）
- [x] 特徴量キャッシュスクリプト (cache_features.py) 完成（`--batch-size` でバッチ処理対応）
- [x] データ前処理スクリプト (preprocess_data.py) 完成

### C. モデル学習
- [x] VAE（Oobleck）学習コード完成 (train_vae.py, bf16, EMA, discriminator)
- [ ] VAE の学習実行と復元品質の検証
- [x] DiT 学習コード完成 (train_dit.py, CFM loss, masked loss, EMA, warmup)
- [ ] DiT の学習実行と品質確認

### D. 推論と評価
- [x] 推論パイプライン構築 (FlowW2NPipeline, Euler 積分 N=10)
- [x] 評価パイプライン構築 (WER-N, WER-W, UTMOS, DNSMOS, SpkSim)
- [x] 統合テスト通過 (84テスト)
- [ ] wTIMIT での評価実行
- [ ] CHAINS での評価実行
- [ ] 結果の論文値との比較

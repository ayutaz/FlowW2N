# データパイプライン計画

> **実装状況**: ✅ 合成ウィスパー生成 (M2) およびデータパイプライン実装完了。品質監査完了。
> - `src/floww2n/data/whisper_synthesis.py` - 4手法 (LPC, glottal, formant, praat) 実装完了
> - `src/floww2n/training/dataset.py` - VAEDataset + DiTDataset + dit_collate_fn
> - `scripts/generate_whisper.py` - バッチ合成ウィスパー生成スクリプト (`--num-workers` でマルチプロセス対応)
> - `scripts/cache_features.py` - 特徴量キャッシュスクリプト (`--batch-size` でバッチ処理対応)
> - `scripts/preprocess_jvs_jsut.py` - JVS/JSUT 日本語データセット前処理スクリプト
> - テスト: test_dataset.py (9 tests), test_whisper_synthesis.py (11 tests), test_training.py (8 tests)
> - コード品質: ruff lint/format 適用済み

本ドキュメントでは、FlowW2N 再現実装に必要なデータパイプラインの全体計画を記載する。
HiFi-TTS-2 データセットの調査、合成ウィスパー生成4手法の実装方針、評価データセットの準備、
DataLoader 設計、キャッシュ戦略をカバーする。

---

## 1. 学習データセット: HiFi-TTS-2

### 1.1 データセット概要

| 項目 | 値 |
|------|-----|
| 名前 | HiFi-TTS-2 |
| ソース | huggingface.co/datasets/nvidia/hifitts-2 |
| 用途 | VAE 学習（normal speech）、DiT 学習（合成ウィスパー生成元） |
| サンプリングレート | 24,000 Hz（原音）→ 16,000 Hz にリサンプリング |
| フォーマット | WAV / FLAC |
| 言語 | 英語 (多言語対応時は日本語も追加) |

### 1.2 ダウンロード方法

```python
from datasets import load_dataset

# Streaming モードで段階的にダウンロード
dataset = load_dataset("nvidia/hifitts-2", streaming=True)

# または全量ダウンロード
dataset = load_dataset("nvidia/hifitts-2")
```

HuggingFace CLI でも可:
```bash
uv run huggingface-cli download nvidia/hifitts-2 --repo-type dataset
```

### 1.3 前処理パイプライン

```
HiFi-TTS-2 (24kHz)
  |
  +---> リサンプリング: 24kHz → 16kHz (torchaudio.transforms.Resample)
  |
  +---> モノラル化: ステレオ → モノラル (mean across channels)
  |
  +---> 正規化: peak normalization to [-1, 1]
  |
  +---> セグメント化: 固定長セグメントに分割
  |       - VAE 学習: 65,536 samples (4.096秒 @ 16kHz)
  |       - DiT 学習: 可変長（バッチ内パディング）
  |
  +---> 保存: 前処理済み音声を .wav / .pt で保存
```

### 1.4 セグメント長の設計

| 用途 | セグメント長 | 理由 |
|------|------------|------|
| VAE 学習 | 65,536 samples (4.1秒) | SA2.0 の設定に準拠。2の累乗で効率的 |
| DiT 学習 | 可変長（最大 ~10秒） | 実運用を想定、パディングで対応 |
| 推論 | 可変長 | 入力音声の長さに依存 |

VAE の潜在表現長: 65,536 / 1024 = 64 フレーム（ratio=1024 の場合）

### 1.5 多言語データセット: JVS / JSUT (日本語)

多言語対応として、日本語データセットも使用可能。`configs/data.json` の `datasets_multilingual` セクションで設定する。

| データセット | 概要 | 話者数 | 総時間 |
|-------------|------|--------|--------|
| JVS (Japanese Versatile Speech) | 日本語多様音声コーパス | 100 | ~30h |
| JSUT (Saruwatari-lab) | 日本語音声コーパス | 1 | ~10h |

前処理スクリプト `scripts/preprocess_jvs_jsut.py` で 16kHz モノラルに変換:

```bash
uv run python scripts/preprocess_jvs_jsut.py --jvs-dir <path> --jsut-dir <path> --output-dir <path>
```

多言語学習時は `scripts/merge_manifests.py` で英語・日本語のマニフェストを結合し、`language_id` フィールドを付与する。

---

## 2. 合成ウィスパー生成

### 2.1 全体設計

FlowW2N は実ウィスパー音声を一切使用せず、合成的に生成されたウィスパー音声のみで学習する。
4手法を **等確率（各25%）** でランダムにサンプリングし、多様な音響アーティファクトを導入する。

```
normal speech (16kHz, mono)
  |
  +---> Random selection (25% each)
  |       |
  |       +---> (1) LPC-based devoicing
  |       +---> (2) Glottal source removal
  |       +---> (3) Formant bandwidth modification
  |       +---> (4) Praat Vocal Toolkit
  |
  +---> synthetic whisper (16kHz, mono, time-aligned)
```

**重要**: 4手法はいずれも信号処理ベースであり、入出力が時間的に完全に整合する。

### 2.2 手法 (1): LPC-based Devoicing

| 項目 | 値 |
|------|-----|
| ソース | github.com/zeta-chicken/toWhisper |
| 原理 | LPC で声道フィルタを推定し、有声励振をノイズ励振に置換 |
| 入力 | normal speech 波形 |
| 出力 | devoiced（無声化）波形 |
| 時間整合 | 完全に保持 |

#### 実装方針

```python
# toWhisper リポジトリから必要な関数をインポートまたは移植
# 主要な処理フロー:
# 1. LPC 分析 (scipy.signal.lpc または手動実装)
# 2. 残差信号（励振源）の抽出
# 3. 有声区間の検出 (F0 推定ベース)
# 4. 有声励振をホワイトノイズに置換
# 5. LPC フィルタで再合成

import numpy as np
from scipy.signal import lfilter, lpc

def lpc_devoice(audio, sr=16000, lpc_order=16, frame_size=512, hop_size=128):
    # フレームごとに LPC 分析
    # 有声区間のみ励振をノイズに置換
    # LPC フィルタで再合成
    pass
```

#### 依存関係
- scipy (LPC 分析)
- numpy
- pysptk (オプション、より高品質な LPC)

### 2.3 手法 (2): Glottal Source Removal

| 項目 | 値 |
|------|-----|
| 参考文献 | Lin et al., ASRU 2023 [5] |
| 原理 | 声門波源（glottal source）を除去し、声帯振動なしの信号を生成 |
| 入力 | normal speech 波形 |
| 出力 | 声門音源除去済み波形 |

#### 実装方針

```python
# 声門逆フィルタリング (Glottal Inverse Filtering) ベース
# 1. 声道フィルタの推定 (IAIF: Iterative Adaptive Inverse Filtering)
# 2. 声門波形の推定
# 3. 声門波形をノイズまたはゼロで置換
# 4. 声道フィルタで再合成

def glottal_source_removal(audio, sr=16000):
    # IAIF または covariance-based GIF を使用
    pass
```

#### 依存関係
- numpy, scipy
- pyworld (F0 推定、有声/無声判定)

### 2.4 手法 (3): Formant Bandwidth Modification

| 項目 | 値 |
|------|-----|
| 参考文献 | Lin et al., ASRU 2023 [5] |
| 原理 | フォルマント帯域幅を増加させ、ウィスパー様の共振特性をシミュレート |
| 入力 | normal speech 波形 |
| 出力 | フォルマント帯域幅変更済み波形 |

#### 実装方針

```python
# フォルマント帯域幅の操作:
# 1. LPC 分析で極（poles）を抽出
# 2. 各極の角度を保持しつつ、大きさを縮小（帯域幅拡大に対応）
# 3. 修正した極から新しい LPC 係数を再構成
# 4. 修正フィルタで信号を再合成

def formant_bandwidth_mod(audio, sr=16000, bandwidth_factor=1.5):
    # bandwidth_factor: 帯域幅の拡大倍率
    pass
```

### 2.5 手法 (4): Praat Vocal Toolkit

| 項目 | 値 |
|------|-----|
| ソース | praatvocaltoolkit.com/whisper.html |
| 原理 | Praat 音声分析ソフトのウィスパー化機能 |
| 実行方法 | Praat スクリプトまたは parselmouth (Python バインディング) |

#### 実装方針

```python
# parselmouth (Praat の Python ラッパー) を使用
import parselmouth
from parselmouth.praat import call

def praat_whisperize(audio, sr=16000):
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    # Praat Vocal Toolkit の whisper スクリプトに相当する操作:
    # 1. 音声を無声化 (de-voicing)
    # 2. ノイズ成分の追加
    # 3. スペクトル包絡の調整
    return whispered_snd.values[0]
```

#### 依存関係
- parselmouth (Praat Python バインディング)
- Praat Vocal Toolkit スクリプト（必要に応じて移植）

### 2.6 統合: ウィスパー合成パイプライン

```python
import random

class WhisperSynthesizer:
    def __init__(self, sr=16000):
        self.sr = sr
        self.methods = [
            self.lpc_devoice,
            self.glottal_removal,
            self.formant_bandwidth_mod,
            self.praat_whisperize,
        ]

    def __call__(self, audio):
        """等確率でランダムに1手法を選択して合成ウィスパーを生成"""
        method = random.choice(self.methods)
        return method(audio)

    def lpc_devoice(self, audio):
        ...

    def glottal_removal(self, audio):
        ...

    def formant_bandwidth_mod(self, audio):
        ...

    def praat_whisperize(self, audio):
        ...
```

### 2.7 マルチプロセス合成ウィスパー生成

`scripts/generate_whisper.py` は `--num-workers` フラグでマルチプロセス生成をサポートする。

```bash
uv run python scripts/generate_whisper.py --data-dir <path> --output-dir <path> --num-workers 8
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--num-workers` | `min(cpu_count, 8)` | 並列ワーカー数。`None` 指定時はCPUコア数に基づき自動決定 |
| `--sample-rate` | 16000 | サンプリングレート |

`num_workers <= 1` の場合はシングルプロセスにフォールバックする。マルチプロセス時は `ProcessPoolExecutor` を使用し、各ワーカーが独立に `WhisperSynthesizer` を初期化して処理する。

### 2.8 品質検証

合成ウィスパーの品質は以下の基準で検証する:

| チェック項目 | 方法 |
|-------------|------|
| F0 の消失 | CREPE / pyworld で F0 推定、有声率が十分に低下しているか確認 |
| スペクトル包絡の保持 | Mel spectrogram の視覚比較 |
| 時間整合 | 入力/出力の波形長が一致するか確認 |
| 聴取テスト | 合成結果がウィスパー様に聞こえるか主観確認 |
| Whisper 特徴の不変性 | 合成ウィスパーと実ウィスパーの Whisper layer 5 特徴の Pearson 相関 |

---

## 3. 評価データセット

### 3.1 wTIMIT

| 項目 | 値 |
|------|-----|
| ベースコーパス | TIMIT |
| 形式 | ウィスパー音声 + 通常音声の並列録音 |
| 話者数 | 50 |
| 評価セット | US ダイアレクトテストセット |
| 発話数 | 1,404 |
| 総時間 | 約2時間 |

**注意**: wTIMIT は学術利用のため LDC (Linguistic Data Consortium) 経由での取得が必要な可能性がある。利用可能性を事前に確認すること。

### 3.2 CHAINS

| 項目 | 値 |
|------|-----|
| 正式名称 | CHaracterizing INdividual Speakers |
| 話者数 | 36 |
| スタイル | 通常音声、ウィスパー、他の発話スタイル |
| ウィスパー発話数 | 約1,332 |
| 総時間 | 約2.45時間 |
| 特徴 | 同一話者・同一内容の通常/ウィスパー/合成ウィスパーのトリプレットあり |

**特筆事項**: CHAINS は domain invariance の検証にも使用される（3種類のトリプレット比較）。

### 3.3 評価データの前処理

```
評価データセット (wTIMIT / CHAINS)
  |
  +---> リサンプリング: 元SR → 16kHz
  |
  +---> モノラル化
  |
  +---> 正規化
  |
  +---> メタデータ整理:
  |       - 発話ID
  |       - 話者ID
  |       - テキストトランスクリプト
  |       - ウィスパー/通常の区分
  |
  +---> 保存: 前処理済み音声 + メタデータ JSON/CSV
```

---

## 4. DataLoader 設計

### 4.1 VAE 学習用 DataLoader

```python
class VAEDataset(torch.utils.data.Dataset):
    """VAE 学習用: normal speech のみ、固定長セグメント"""

    def __init__(self, audio_dir, segment_length=65536, sr=16000):
        self.audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        self.segment_length = segment_length  # 65536 samples = 4.096s
        self.sr = sr

    def __getitem__(self, idx):
        audio, orig_sr = torchaudio.load(self.audio_files[idx])
        if orig_sr != self.sr:
            audio = torchaudio.transforms.Resample(orig_sr, self.sr)(audio)
        audio = audio.mean(dim=0)  # mono

        # ランダムクロップ
        if audio.shape[0] >= self.segment_length:
            start = random.randint(0, audio.shape[0] - self.segment_length)
            audio = audio[start:start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.shape[0]))

        return audio  # shape: (segment_length,)
```

### 4.2 DiT 学習用 DataLoader

```python
class DiTDataset(torch.utils.data.Dataset):
    """DiT 学習用: normal speech + 合成ウィスパーペア"""

    def __init__(self, audio_dir, whisper_synthesizer, sr=16000,
                 max_length=160000):
        self.audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
        self.whisper_synth = whisper_synthesizer
        self.sr = sr
        self.max_length = max_length  # 10秒 @ 16kHz

    def __getitem__(self, idx):
        audio_normal, orig_sr = torchaudio.load(self.audio_files[idx])
        if orig_sr != self.sr:
            audio_normal = torchaudio.transforms.Resample(orig_sr, self.sr)(audio_normal)
        audio_normal = audio_normal.mean(dim=0)  # mono

        # トランケート
        if audio_normal.shape[0] > self.max_length:
            start = random.randint(0, audio_normal.shape[0] - self.max_length)
            audio_normal = audio_normal[start:start + self.max_length]

        # 合成ウィスパー生成（オンラインまたはキャッシュ）
        audio_whisper = self.whisper_synth(audio_normal.numpy())
        audio_whisper = torch.from_numpy(audio_whisper).float()

        return {
            "normal": audio_normal,
            "whisper": audio_whisper,
            "length": audio_normal.shape[0],
        }
```

### 4.3 コレート関数

```python
def dit_collate_fn(batch):
    """可変長バッチのパディング"""
    max_len = max(item["length"] for item in batch)
    # VAE ratio でアラインメント
    ratio = 1024
    max_len = ((max_len + ratio - 1) // ratio) * ratio

    normals = []
    whispers = []
    lengths = []

    for item in batch:
        pad_len = max_len - item["length"]
        normals.append(F.pad(item["normal"], (0, pad_len)))
        whispers.append(F.pad(item["whisper"], (0, pad_len)))
        lengths.append(item["length"])

    return {
        "normal": torch.stack(normals),    # (B, max_len)
        "whisper": torch.stack(whispers),  # (B, max_len)
        "lengths": torch.tensor(lengths),   # (B,)
    }
```

---

## 5. キャッシュ戦略

### 5.1 合成ウィスパーのキャッシュ

| 方式 | メリット | デメリット | 推奨 |
|------|---------|-----------|------|
| オンライン生成 | ストレージ不要、多様性最大 | 学習速度低下 | 初期検証 |
| 事前生成（全量） | 学習時高速 | ストレージ2倍 | 本番学習 |
| ハイブリッド | バランス | 実装が複雑 | 中規模実験 |

**推奨**: 本番学習では事前生成。各手法で全データの合成ウィスパーを事前に生成しておき、
学習時にランダムに手法を切り替える（事前に4倍の合成データを作成）。

### 5.2 Whisper / ECAPA-TDNN 特徴のキャッシュ

DiT 学習時に毎回 Whisper encoder と ECAPA-TDNN を実行するのは計算コストが高い。

```
事前計算パイプライン:
  合成ウィスパー → Whisper Base (layer 5) → h をキャッシュ (.pt)
  通常音声      → ECAPA-TDNN           → e_spk をキャッシュ (.pt)
  通常音声      → VAE encoder          → z1 をキャッシュ (.pt)
```

**ストレージ見積もり** (1時間の音声あたり):
| データ | 形状 | dtype | サイズ |
|--------|------|-------|--------|
| Whisper h | (T/320, 512) | float32 | ~360 MB/h |
| Speaker e_spk | (192,) | float32 | ~768 B/utterance |
| VAE z1 | (64, T/1024) | float32 | ~14 MB/h |

### 5.3 バッチ特徴量キャッシュ

`scripts/cache_features.py` は `--batch-size` フラグでバッチ処理をサポートし、GPU を効率的に活用する。

```bash
uv run python scripts/cache_features.py \
    --data-dir <path> --output-dir <cache_path> \
    --vae-checkpoint <vae_ckpt> --batch-size 16
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch-size` | 16 | 一度に処理する音声ファイル数 |
| `--device` | `cuda` | 推論デバイス |
| `--language` | `None` | 多言語モデル用の言語コード |

バッチ内の音声ファイルをまとめて読み込み、パディング後に Whisper encoder、ECAPA-TDNN、VAE encoder を一括推論することで、ファイル単位の処理に比べスループットが大幅に向上する。

### 5.4 キャッシュ付き DataLoader

```python
class CachedDiTDataset(torch.utils.data.Dataset):
    """事前計算済み特徴量を使用する高速版"""

    def __init__(self, manifest_path):
        # manifest: JSON/CSV with paths to cached features
        self.manifest = load_manifest(manifest_path)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        z1 = torch.load(entry["z1_path"])        # (64, L)
        h = torch.load(entry["whisper_h_path"])   # (T_h, 512)
        e_spk = torch.load(entry["spk_path"])     # (192,)

        return {"z1": z1, "h": h, "e_spk": e_spk}
```

---

## 6. データ準備ロードマップ

| フェーズ | タスク | 優先度 | 所要日数 |
|---------|--------|--------|---------|
| 0 | HiFi-TTS-2 ダウンロード + 前処理スクリプト作成 | P0 | 1-2日 |
| 1 | 合成ウィスパー手法 (1) LPC devoicing 実装 | P0 | 2-3日 |
| 2 | 合成ウィスパー手法 (4) Praat 実装 | P0 | 1-2日 |
| 3 | 合成ウィスパー手法 (2)(3) Glottal/Formant 実装 | P1 | 3-4日 |
| 4 | 品質検証（F0消失、スペクトル比較） | P0 | 1日 |
| 5 | VAE 学習用 DataLoader 実装 | P0 | 1日 |
| 6 | DiT 学習用 DataLoader 実装 | P0 | 1-2日 |
| 7 | 特徴量事前キャッシュスクリプト作成 | P1 | 1日 |
| 8 | 評価データセット準備 (wTIMIT, CHAINS) | P1 | 2-3日 |

### 優先順位の根拠

- **Phase 0-2 を最優先**: VAE 学習にはデータが必要。合成ウィスパーは LPC と Praat を先に実装すれば2手法で学習開始可能。
- **Phase 3 は後回し可**: 残り2手法は多様性向上のためで、2手法でも学習は開始できる。
- **Phase 8 は学習と並行**: 評価データは学習完了後に使用するため、並行して準備可能。

---

## 7. 技術的注意点

1. **リサンプリング品質**: 24kHz → 16kHz の変換には `torchaudio.transforms.Resample` の `resampling_method="sinc_interp_kaiser"` を使用し、エイリアシングを防ぐ。

2. **合成ウィスパーの多様性**: 4手法の混合は過適合防止に重要。初期実験では2手法でも可だが、最終学習では4手法すべてを使用すべき。

3. **ウィスパー品質の非均一性**: 手法によって合成品質にばらつきがある。手法ごとの生成サンプルを確認し、明らかに品質が低い手法がある場合はサンプリング確率を調整する検討も必要。

4. **ストレージ管理**: HiFi-TTS-2 + 合成ウィスパー（4手法分）+ キャッシュで大量のストレージが必要。事前に見積もりを行い、十分な空き容量を確保すること。

5. **DataLoader のワーカー数**: 合成ウィスパーのオンライン生成はCPU-bound。`num_workers` を多めに設定し（8-16）、GPU のアイドルを防ぐ。

6. **乱数シード管理**: 再現性のため、合成ウィスパー手法の選択とセグメントのクロップ位置にシードを設定する。

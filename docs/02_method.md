# FlowW2N: 手法（Method）

## 1. FlowW2N アーキテクチャの全体像

FlowW2N は、ささやき音声から通常音声への変換を **潜在空間上の flow matching** によって実現するアーキテクチャである。パイプラインは大きく以下の3つのコンポーネントから構成される。

1. **VAE（Oobleck encoder-decoder）**: 波形を潜在表現に圧縮・復元する。
2. **Flow Matching モデル（DiT）**: 潜在空間上でガウスノイズからターゲット潜在表現への輸送を学習する。
3. **条件付けモジュール**: ウィスパー入力からコンテンツ特徴と話者埋め込みを抽出し、flow matching モデルに注入する。

推論時には、ウィスパー音声から条件信号を抽出し、ガウスノイズを起点として ODE を解くことで通常音声の潜在表現を生成し、VAE デコーダで波形に復元する。

---

## 2. 潜在空間（VAE: Oobleck Encoder-Decoder）

FlowW2N では、stable-audio-tools の **Oobleck encoder-decoder** に基づく完全畳み込み VAE を採用している。

### エンコード

波形 $s \in \mathbb{R}^T$ を潜在表現に圧縮する。

$$z = E(s) \in \mathbb{R}^{D \times L}$$

- 潜在次元: $D = 64$
- 圧縮比: $r$（約 15.6 Hz のフレームレート）

### デコード

潜在表現から波形を復元する。

$$\hat{s} = D(z)$$

### VAE の学習

- **学習データ**: 通常音声のみ（HiFi-TTS-2）
- **学習ステップ数**: 80,000 ステップ
- **バッチサイズ**: 256
- **損失関数**:
  - マルチ解像度 STFT 損失（multi-resolution STFT loss）
  - マルチスケール弁別器損失（multi-scale discriminator loss）
  - KL ダイバージェンス正則化（KL divergence regularization）

VAE は通常音声のみで学習されるため、ささやき音声の潜在表現の再構成品質は保証されないが、flow matching モデルが潜在空間上で通常音声の潜在表現を直接生成するため、この制約は問題とならない。

---

## 3. Flow Matching の定式化

### 速度場と ODE

Flow matching では、学習可能な速度場 $v_\theta(z, t)$ が以下の常微分方程式（ODE）を定義する。

$$\frac{dz}{dt} = v_\theta(z, t)$$

この ODE を $t = 0$ から $t = 1$ まで積分することで、ソース分布からターゲット分布への輸送を実現する。

### Conditional Flow Matching（CFM）目的関数

最適輸送（optimal transport）に基づく補間パスを用いて、CFM 目的関数を以下のように定義する。

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, z_0, z_1} \left[ \| v_\theta(z_t, t) - (z_1 - z_0) \|^2 \right]$$

ここで、

- **補間パス（最適輸送補間）**: $z_t = (1 - t) z_0 + t z_1$
- **ターゲット速度**: $v_t = z_1 - z_0$
- **時刻サンプリング**: $t \sim \mathcal{U}[0, 1]$

---

## 4. Paired Flow Matching の限界と「Phoneme Boundary Blur」問題

### Paired Flow Matching の定式化

ナイーブなアプローチとして、ウィスパー音声と通常音声のペアをそれぞれ VAE でエンコードし、直接的に flow matching を適用することが考えられる。

- ソース: $z_0 = E(s_{\text{whisper}})$（ウィスパー音声の潜在表現）
- ターゲット: $z_1 = E(s_{\text{normal}})$（通常音声の潜在表現）

### 時間的ミスアライメントの問題

この方法には **時間的アライメント（temporal alignment）** が必要であるという根本的な問題がある。

- フレーム $i$ における $z_0$ と $z_1$ が **異なる音素に対応** する場合、補間 $z_t = (1-t)z_0 + tz_1$ は **音響的に非一貫的（acoustically incoherent）** な中間表現を生成してしまう。
- この結果、速度場の監督信号（supervision）が破壊され、学習が不安定化する。
- この現象は **phoneme boundary blur（音素境界のぼやけ）** と呼ばれ、変換後の音声で音素境界付近の品質が著しく劣化する原因となる。

Dynamic Time Warping（DTW）を用いた時間整合を試みても、フレームレベルでの音韻一貫性は保証できず、この問題を完全に解決することはできない。

---

## 5. ガウスノイズ事前分布 + 外部条件付けによる解決策

### 設計原理

Paired flow matching の限界を克服するため、FlowW2N では以下の設計を採用する。

1. **ソース分布にガウスノイズを使用**: $z_0 \sim \mathcal{N}(0, I)$
2. **ウィスパー入力からの条件信号 $c$ を外部から注入**

### 修正された目的関数

$$\mathcal{L} = \mathbb{E}_{t,\, z_0 \sim \mathcal{N}(0, I),\, z_1} \left[ \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2 \right]$$

### なぜこの設計が有効か

- $z_0$ が **非構造化ノイズ（unstructured noise）** であるため、補間パス $z_t = (1-t)z_0 + tz_1$ がミスアラインされた音韻内容を混合することがない。
- ウィスパー音声の情報（コンテンツ、話者特性）は条件信号 $c$ として外部から供給されるため、flow matching の輸送パス自体は音韻的に一貫した軌道を描くことができる。
- この分離により、phoneme boundary blur 問題が根本的に解消される。

---

## 6. モデルアーキテクチャ

### DiT（Diffusion Transformer）

Flow matching モデルの主要アーキテクチャとして **DiT（Diffusion Transformer）** を採用している。

| 項目 | 仕様 |
|------|------|
| Transformer ブロック数 | 24 |
| タイムステップ注入 | AdaLN（Adaptive Layer Normalization） |
| 条件付けメカニズム | Cross-attention（-ca）/ Prepending（-p） |

### タイムステップの注入

タイムステップ $t$ は **AdaLN（adaptive layer normalization）** を通じて各 Transformer ブロックに注入される。AdaLN では、正規化後のスケールとシフトパラメータをタイムステップに応じて動的に調整する。

### 条件信号の構成

最終的な条件信号は以下の2つの要素から構成される。

$$c = \{e_{\text{spk}}, h\}$$

#### (a) 話者埋め込み $e_{\text{spk}}$

- **凍結された ECAPA-TDNN 話者エンコーダ** から抽出
- **AdaLN** を通じてモデルに注入（タイムステップと同様のメカニズム）

#### (b) コンテンツ特徴 $h$

- **Whisper エンコーダ** から抽出されたコンテンツ表現
- 以下の2つの注入方式を比較:
  - **Cross-attention（-ca）**: Transformer ブロック内で cross-attention 層を通じて注入
  - **Prepending（-p）**: コンテンツ特徴をシーケンスの先頭に連結して入力

---

## 7. 推論パイプライン

### 推論手順

1. ウィスパー音声から条件信号 $c = \{e_{\text{spk}}, h\}$ を抽出する。
2. ガウスノイズからサンプリングする: $z_0 \sim \mathcal{N}(0, I)$
3. ODE を Euler 積分で解く:

$$z_1 = z_0 + \int_0^1 v_\theta(z_t, t, c)\, dt$$

4. VAE デコーダで波形を復元する: $\hat{s} = D(z_1)$

### Euler 積分の詳細

- **積分ステップ数**: $N = 10$
- 各ステップで速度場を評価し、前進 Euler 法で更新する:

$$z_{t + \Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t, c), \quad \Delta t = \frac{1}{N}$$

わずか 10 ステップという少ないステップ数で高品質な変換を実現しており、推論の効率性が高い。

---

## 8. Domain Invariance とレイヤー選択

### 動機

条件信号 $c$ に含まれるコンテンツ特徴 $h$ は **ドメイン不変（domain-invariant）** である必要がある。すなわち、合成ウィスパー音声から抽出した特徴と、実ウィスパー音声から抽出した特徴が十分に近いことが求められる。これにより、合成データのみで学習したモデルが実ウィスパー音声に汎化できる。

### 評価方法

**CHAINS データセット** を使用して評価を行う。CHAINS には同一発話の以下の3種類の音声トリプレットが含まれる。

1. 通常音声（normal speech）
2. 実ウィスパー音声（real whispered speech）
3. 合成ウィスパー音声（synthetic whispered speech）

**WhisperX** で強制アライメントを行い、**単語レベル表現** を抽出した上で **Pearson 相関** を計算する。

### 2つのギャップ

| ギャップ | 定義 |
|---------|------|
| **Synthesis Gap** | 合成ウィスパー vs 実ウィスパーの特徴間の差異 |
| **Modality Gap** | 通常音声 vs 実ウィスパーの特徴間の差異 |

### Whisper 特徴 vs HuBERT Soft 特徴

- Whisper 特徴は HuBERT Soft と比較して、**全レイヤーでより高い Pearson 相関** を示す。
- 特に後段レイヤーでは **0.90 を超える Pearson 相関** を達成する。

### レイヤー選択基準

最適レイヤー $\ell^*$ は以下の基準で選択する。

$$\ell^* = \underset{\ell}{\arg\max} \left[ \text{Invariance}(\ell) \times \text{CCA}(\ell) \right]$$

ここで、

- **Invariance($\ell$)**: レイヤー $\ell$ における合成ウィスパー vs 実ウィスパーの Pearson 相関
- **CCA($\ell$)**: レイヤー $\ell$ におけるフレーム特徴と単語同一性（word identity）の正準相関分析（Canonical Correlation Analysis）

両メトリクスを $[0, 1]$ に min-max 正規化した上で乗算する。

### 選択結果

| モデル | 最適レイヤー |
|--------|-------------|
| Whisper Base | $\ell^* = 5$ |
| HuBERT | $\ell^* = 10$ |

この基準により、**ドメイン不変性（合成と実の間の汎化能力）** と **コンテンツ情報量（言語的内容の弁別力）** の両方を最大化するレイヤーが選択される。

---

## 9. 合成ウィスパーデータ生成の4手法

### 設計思想

FlowW2N は学習に実ウィスパー音声を一切使用せず、**合成的に生成されたウィスパー-通常音声ペア** のみを使用する。合成データは **HiFi-TTS-2** から生成され、以下の4手法が **等確率** で適用される。

多様な音響アーティファクトを増やすため、意図的に複数の手法を使用している。4手法とも構造的に **時間アラインされたペア** を生成するため、paired flow matching における時間整合の問題を回避できる（ただし FlowW2N ではガウスノイズ事前分布を採用しているため、この時間整合はそもそも不要である）。

### 手法一覧

| # | 手法 | 説明 |
|---|------|------|
| (1) | **LPC-based devoicing** | 有声励振（voiced excitation）をノイズ励振（noise excitation）に置換する。LPC（線形予測符号化）で声道フィルタを推定し、励振源のみを操作することで、スペクトル包絡を保持したままウィスパー化する。 |
| (2) | **Glottal source removal** | 有声化情報（voicing information）を除去する。声門波源（glottal source）を取り除くことで、声帯振動のないウィスパー様の音声を生成する。 |
| (3) | **Formant bandwidth modification** | フォルマント帯域幅を増加させることで、ウィスパー音声に特徴的な共振特性（whisper resonance）をシミュレートする。ウィスパー音声ではフォルマント帯域幅が広がる傾向があることに基づく。 |
| (4) | **Praat Vocal Toolkit** | Praat 音声分析ソフトウェアの Vocal Toolkit プラグインを使用してウィスパー化を行う。 |

### 各手法の特徴

- 4手法はいずれも **信号処理ベース** のアプローチであり、ニューラルネットワークを用いた合成ではない。
- 入力波形と出力波形が **時間的に完全に整合** しているため、フレームレベルのペアデータとして直接利用可能である。
- 複数手法を混合することで、モデルが特定の合成アーティファクトに過適合することを防ぎ、実ウィスパー音声への汎化性能を向上させている。

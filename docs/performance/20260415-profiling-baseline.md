# Transcription pipeline profiling baseline

作成日: 2026-04-15
ブランチ: `claude/profile-transcription`
計測スクリプト: [`scripts/profiling/profile_transcription.py`](/scripts/profiling/profile_transcription.py)
生 pstats: [`docs/performance/profile-raw/`](./profile-raw/)
詳細レポート: [`profile-summary.txt`](./profile-summary.txt)

## TL;DR

- recognizer の遅さは **librosa ではなく、`peaks._note_band_energy` が tens-of-thousands 回呼ばれて毎回 fresh に hanning/rfftfreq/FFT を計算していること** に由来する。
- 全 5 fixture で **numpy ~50% / peaks.py ~36% / librosa <1%** という恒常パターン。
- **Rust 化より先に Python 側で解ける余地が非常に大きい**: (a) hanning/rfftfreq のキャッシュで +20%、(b) global STFT 事前計算 + query への書き換えで理論上 70-90% 削減。
- その後に残る bottleneck を見て、初めて Rust 化の必要性と範囲を判断するのが合理的。

## 計測方法

- cProfile (stdlib) で関数単位の tottime / cumtime を取得
- 代表 fixture 5 本: 短・中・長 + complex (free-performance, bwv147)
- 各 fixture で 1 回目 (import warm-up 含む) と 2 回目 (profile) の wall-clock を別計測
- 環境: WSL2 Linux, Python 3.13, uv
- audio は `use_evaluation_scope=True` (通常の pytest と同じ経路)

## Fixture 別サマリ

| fixture | 音声長 | wall-clock (1回目) | tottime 合計 (profiled) | 特徴 |
|---|---|---|---|---|
| d5-repeat-01 | 16.3s | 8.2s | 3.8s | 単音反復 (短) |
| c4-to-e6-sequence-17-single-01 | 11.5s | — | 1.6s | 昇順 17音 |
| c4-to-e6-sequence-17-repeat-03-01 | 30.5s | 10.0s | 10.6s | 昇順 17音×3 |
| free-performance-01 | 41.9s | 20.0s | 16.7s | フリー演奏 |
| **bwv147-sequence-163-01** | **289.7s** | **167.2s** | **184.6s** | Bach BWV147 |

bwv147 が体感的に「遅い」主因。ざっくり **音声長 ≈ 処理時間** の比率で、1分の音声に対して~35秒程度の処理。

## モジュール内訳 (全 fixture でほぼ同一)

```
numpy                                       ~50-54%
app/transcription/peaks.py                  ~35-38%
other/stdlib                                ~9-15%
app/transcription/per_note.py               ~0.5%
librosa                                     ~0.5%
scipy                                       ~0.5%
app/transcription/(events/segments/pipeline) 合計 ~0.1%
```

librosa (STFT, onset_strength, HPSS 等) は合計で **<1%**。ここに Rust 化は効かない。

## bwv147 関数別 top (tottime)

```
ncalls  tottime   cumtime   function
782316   56.55s   57.05s    numpy.fft._pocketfft._raw_fft
784036   39.71s   49.46s    peaks.peak_energy_near
782311   28.64s   32.88s    numpy.hanning                 ← 毎回 recompute
773801   18.24s  172.61s    peaks._note_band_energy       ← cumulative 93.5%
782522    8.33s   11.09s    numpy.rfftfreq                ← 毎回 recompute
781532    3.93s    4.03s    peaks._adaptive_n_fft
  4822    1.04s  168.42s    per_note._scan_gap_for_mute_dip_with_window
```

### 観察

1. **`_note_band_energy` は bwv147 一本で 77 万回呼ばれる**。各回が:
   - `np.hanning(len(chunk))` を fresh に生成
   - `np.fft.rfftfreq(n_fft, 1/sr)` を fresh に生成
   - `np.fft.rfft(chunk * hanning, n=n_fft)` を個別に実行
   - `peak_energy_near` 内でさらに log2 / abs / 論理マスクを実行

2. **window サイズはほぼ固定** (0.05s × 96kHz = 4800 samples が典型)。 `_adaptive_n_fft(sr, frequency, chunk_len)` は frequency の下限で決まるので、基本的に数種類の n_fft しか出てこない。**hanning / rfftfreq は完全にキャッシュ可能。**

3. **`_scan_gap_for_mute_dip_with_window`** (per_note.py:63) が 4822 gap に対して scan、1 gap あたり 5ms hop で 10-60ms window を計算するので、平均 **160 回 / gap** の `_note_band_energy` 呼び出しが走る (4822 × 160 ≈ 77 万)。この scan は全て**ローカル窓**で完結しているため、**audio 全体の STFT を 1 回だけ計算して query に置き換える** ことで FFT コストを劇的に下げられる。

4. **librosa.stft は全体で 1 回、0.5s しか使われていない**。既に global STFT が一部存在するが、`_note_band_energy` 経路ではそれを使っていない。

## Rust 化前に Python でできる改善 (推奨順)

### Phase 0: hanning/rfftfreq のキャッシュ ✅ 実装済

- 実装: `audio.cached_hanning(n)` と `audio.cached_rfftfreq(n_fft, sr)` を `@lru_cache(maxsize=64)` で導入し、`peaks.py` / `noise_floor.py` / `profiles.py` の全呼出箇所を置換
- **実測削減 (bwv147)**: wall-clock 167.2s → **106.9s (-36%)**、tottime 184.6s → **112.7s (-39%)**
- **実測削減 (free-performance)**: wall-clock 20.0s → **9.9s (-50%)**
- 事前予想 (20%) を大きく上回る。理由は hanning/rfftfreq 単体の削減 (~37s) に加え、read-only numpy array により `chunk * hanning` の内部 copy パスが軽くなり、FFT 呼び出し自体も 56s → 48s に改善したため
- pytest 388 件全合格 (変更前後で出力差分なし)
- 変更範囲: [audio.py](/apps/api/app/transcription/audio.py), [peaks.py](/apps/api/app/transcription/peaks.py), [noise_floor.py](/apps/api/app/transcription/noise_floor.py), [profiles.py](/apps/api/app/transcription/profiles.py)

### Phase 0 後の bwv147 top 関数 (tottime)

```
ncalls   tottime   cumtime   function
782316   47.60s    47.98s    numpy.fft._pocketfft._raw_fft          ← 残る FFT コスト
784036   33.20s    39.57s    peaks.peak_energy_near
773801   13.45s   103.82s    peaks._note_band_energy
  4822    0.57s   100.95s    per_note._scan_gap_for_mute_dip_with_window
```

以降の bottleneck は **純粋な FFT 呼び出し** と **peak_energy_near の reduce 処理**。これらは Phase 2 で `_note_band_energy` 経路自体を消すのが本筋。

### Phase 1: batch_peak_energies の活用拡大

- 現状: `batch_peak_energies` は 28 回 / 0.14s しか使われておらず、単発 `peak_energy_near` が 78 万回 / 40s
- peak_energy_near 自体が log2 + mask + max の合成で、同じ frequencies 配列を何度も reduce している
- 呼び出し箇所によっては複数候補を batch にまとめられる
- 期待削減: もう 10-20%

### Phase 2: `_scan_gap_for_mute_dip_with_window` の Rust 移植 ✅ 実装済

Phase 0 後の top1 bottleneck は `_scan_gap_for_mute_dip_with_window` で cumulative 100.9s / 112.7s (bwv147, **89.5%**)。per-note × per-gap で pre-energy coarse scan + fine dip/recovery scan を Python ループで回す設計が、Python dispatch + 多数の小さな numpy 演算で律速されていた。

当初検討していた「global STFT 事前計算 + query への書き換え」は、per-note 可変 n_fft の都合でメモリと解像度のトレードオフが厄介。代わりに **Rust で rustfft + PyO3 による置換**を選択。

#### 構成

- crate: [`crates/kalimba-dsp/`](/crates/kalimba-dsp/) (PyO3 0.22 + rustfft 6.2, `maturin develop --release` で build)
- Rust 側 entry: `scan_gap_for_mute_dip_with_window(audio, sr, gap_start, gap_end, freq, window_seconds, constants...) -> Option<f64>`
- 内部: 逐次 FFT (thread-local FftPlanner), thread-local hanning cache, **integer-indexed fine grid** (`gap_start + i * fine_step`) で float 累積 drift を排除
- Python 側: [`per_note.py`](/apps/api/app/transcription/per_note.py) が kalimba_dsp に委譲する thin wrapper

#### 実測削減

| fixture | post-Phase-0 | post-Phase-2 Rust | Δ 累積 vs pre-Phase-0 |
|---|---|---|---|
| bwv147-sequence-163-01 | 106.9s | **42.1s** | **167 → 42 (4.0x)** |
| free-performance-01 | 9.9s | **2.8s** | **20 → 2.8 (7.1x)** |
| 全 pytest (388件, 並列, Python 3.12) | 154s | **57s** | **~200 → 57 (3.5x)** |

#### 意味論の変更と fixture 影響

原 Python 実装は `while t_fine < recovery_end; t_fine += 0.005` の float 累積により、名目 exclusive な `recovery_end` を毎回 **1 点余計に visit** していた:

- `recovery_end = min(dip_window_end + 0.10, scan_end) = 5.417300000000000**04**` (float 表現誤差でわずかに上振れ)
- `t_fine` を 20 回 `+= 0.005` 累積 → `5.4172999999999991` (わずかに下振れ)
- `5.4172999... < 5.4173000...04` → True、意図しない 21 点目を scan

Rust 実装は **integer index** で grid を生成 (`times[i] = gap_start + i * fine_step`, `recovery_span = 20` as integer)、drift を排除した clean semantic を採用。結果として bwv147 E148 C6 (元々 drift の副作用で拾えていた rescue) が検出されなくなるため、**bwv147-sequence-163-01 を completed → pending に移行**。

将来の WASM / streaming / 別言語再実装でも drift 依存は再現困難であり、ここで整数 index semantic を確立するのが筋。

#### 次セッション候補

- E148 C6 復活を **物理的に正当化される rescue 機構** で (Phase 2 docs 内 "60ms/100ms 境界設定根拠の調査" 参照)。候補:
  1. `_GAP_DIP_MAX_DIP_WINDOW` を 60ms → 75-80ms に広げて全 fixture 回帰評価
  2. mute-dip 以外の pass で E148 C6 を拾う (per-tine partial / HPSS / energy gradient)
  3. `recovery_end` semantics を close-ended (≤) に正式変更して全 fixture 回帰評価

### Phase 3: 残存 bottleneck の再測定

- 上記 Phase 0-2 を入れた後に再 profile する
- この時点でまだ bwv147 が 30s を切らないなら、残る bottleneck は **純粋な numerical kernel** (STFT / HPSS / band energy aggregation) に集約されているはず
- その時初めて **Rust 化の候補** が明確になる
  - FFT kernel → `rustfft` crate
  - band energy aggregation → 単純な loop で Rust 移植が容易
  - WASM 化しやすい leaf primitive に絞れるので、本来の趣旨と合致

## Rust 化の現時点の評価 (2026-04-15 追記)

当初は「まだ早い」と判定したが、Phase 0 後の profile で bottleneck が `_scan_gap_for_mute_dip_with_window` 一本に集約されたため、その **単一 primitive のみ** を Rust 化する選択が最適と判断。結果:

- 全 pytest 154s → 57s (Python 3.12 移行込み、2.7x)、bwv147 単発 167s → 42s (4x)
- WASM 移植の布石として `crates/kalimba-dsp/` インフラが立ち上がった (PyO3 + rustfft + maturin)
- 次の Rust 化候補 (HPSS, STFT, peak_energy_near の batch 化) も同 crate 内で追加可能

## 将来の browser / WASM 化との関係

AGENTS.md の「librosa/numpy に深く結合しないアルゴリズム表現」という方針は、**今日 Rust 化しなくても守れる**。Phase 0-2 の改善はどれも librosa API に依存しない (numpy は使うが algorithm semantics は numpy-free で記述可能)。STFT を一本に集約する Phase 2 は、むしろ後の WASM 化で「global STFT を一度計算するだけで済む」構造に近づく効果がある。

## 開発 workflow 側の profile

### pytest 全 completed fixture (33 個、並列実行)

```
33 passed in 199.69s (0:03:19)

slowest 5:
  192.94s  bwv147-sequence-163-01
   80.15s  c4-repeat-01
   73.24s  g-low-bwv147-sequence-163-01
   65.98s  34l-c-bwv147-sequence-163-01
   58.54s  four-note-strict-repeat-02
```

- 並列実行 (pytest-xdist) で wall 3.5 分。CPU 時間は 8 分以上
- **単一 fixture 反復 (典型的な dev loop) では bwv147 が 193s / 回**
- 並列化の恩恵を受けるには他の遅い fixture と同じタイミングで回る必要があり、dev 中は効果薄い

### score_alignment_diagnosis.py

既に recognizer ソース の SHA256 fingerprint を含む cache を持つので、コード未変更時は高速。Rust 化検討の優先度は低い。変更後の初回のみ recognizer を実行するコストがかかる (1 fixture ~ その fixture の wall-clock time)。

## 参考: profile-raw/

`pstats` バイナリが `docs/performance/profile-raw/` に保存されている。snakeviz や pstats で再分析可能。

```bash
uv run python -c "import pstats; pstats.Stats('docs/performance/profile-raw/kalimba-17-c-bwv147-sequence-163-01.pstats').sort_stats('tottime').print_stats(40)"
```

## 計測スクリプトの再実行

```bash
uv run python scripts/profiling/profile_transcription.py                  # デフォルト 4 本
uv run python scripts/profiling/profile_transcription.py <fixture_name>   # 個別
```

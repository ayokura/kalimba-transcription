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

### Phase 0: hanning/rfftfreq のキャッシュ (即効、低リスク)

- 期待削減: bwv147 で hanning 28.6s + rfftfreq 8.3s = **~37s (20%)**
- 実装: `peaks.py` の module scope で `@lru_cache(maxsize=32)` の `_hanning_window(n)` と `_rfftfreq(n_fft, sr)` を用意して差し替える
- リスク: ほぼゼロ (純粋な値 caching、変更前後で bit-exact)

### Phase 1: batch_peak_energies の活用拡大

- 現状: `batch_peak_energies` は 28 回 / 0.14s しか使われておらず、単発 `peak_energy_near` が 78 万回 / 40s
- peak_energy_near 自体が log2 + mask + max の合成で、同じ frequencies 配列を何度も reduce している
- 呼び出し箇所によっては複数候補を batch にまとめられる
- 期待削減: もう 10-20%

### Phase 2: global STFT + query へのアーキテクチャ変更 (最大効果)

- 期待削減: bwv147 で _raw_fft 56s + _note_band_energy overhead ~18s = **~74-140s (40-75%)**
- 発想: audio 全体を適切な hop (e.g. 2.5ms = 256 samples @ 96kHz) で一度 STFT しておき、`_note_band_energy(t, f)` をフレーム補間 + 周波数 bin lookup に置き換える
- **単発 FFT を tens-of-thousands 回実行する現設計の代替**
- リスク: 中。周波数/時間解像度のトレードオフを fixture で検証する必要がある。ignoredRanges/splice との整合にも注意
- 注意: 上記に加え、`_adaptive_n_fft` は最低周波数で決まる n_fft を要求しているので、STFT の周波数解像度が低音側で足りないと正確性が落ちる。低音用 / 高音用の 2 解像度 STFT を併用するなどの工夫が要る可能性

### Phase 3: 残存 bottleneck の再測定

- 上記 Phase 0-2 を入れた後に再 profile する
- この時点でまだ bwv147 が 30s を切らないなら、残る bottleneck は **純粋な numerical kernel** (STFT / HPSS / band energy aggregation) に集約されているはず
- その時初めて **Rust 化の候補** が明確になる
  - FFT kernel → `rustfft` crate
  - band energy aggregation → 単純な loop で Rust 移植が容易
  - WASM 化しやすい leaf primitive に絞れるので、本来の趣旨と合致

## Rust 化の現時点の評価

- **まだ早い**、が **方向性自体は正しい**
- 今日いきなり peaks.py を Rust crate に切り出すのは、ほぼ同じ速度改善を Python キャッシュで得られてしまうため非効率
- 一方で Phase 0-1 は 1-2 日で片付き、Phase 2 は中期課題 (アーキテクチャ変更を伴うため recognizer チューニング方針への影響を見ながら)
- Phase 2 完了後に残る bottleneck は、**現状の peaks.py 全体** ではなく、より限定された primitive になる。そこが本来の Rust 化の最適ターゲット

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

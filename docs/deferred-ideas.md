# Deferred Ideas

検討・実装したが不採用になったアイデア、または将来再検討の余地があるアプローチをリストアップする。
各アイデアに不採用理由と再検討条件を記録し、同じ問題に再遭遇したときの参考にする。

## Carryover rejection: backward_attack_gain guard on mute-dip exemption

- **Issue**: #107
- **日付**: 2026-04-01
- **概要**: `_has_mute_dip_reattack()` が True を返す場合に `onset_backward_attack_gain()` で二重検証し、bwg < 2.0 なら reject
- **動機**: mute-dip の false positive (sympathetic excitation) が E17 A4, E129 C5 で発生。backward_attack_gain は genuine attack (>7.9) と carryover (<0.8) を数桁の差で分離可能
- **不採用理由**: mute-dip energy window を 50ms に拡大したところ false positive が解消。根本原因は 15ms 窓の測定不安定であり、窓の修正で backward_attack_gain guard は不要になった
- **再検討条件**: mute-dip window 拡大後も false positive が発生するケースが見つかった場合。特に F3 など低音域で mute-dip と窓サイズが拮抗する場合に有用な可能性がある
- **コミット**: `0587562` (追加) → `0f79e7d` (削除)

## Upper secondary carryover: backward_attack_gain-based elif gate

- **Issue**: #107
- **日付**: 2026-04-01
- **概要**: upper secondary で `primary_onset_gain < 20.0` の場合に、`onset_gain < 2.5 AND mute_dip=False AND backward_attack_gain < 2.0` で reject する代替パス
- **動機**: E28 <B4,G4> に extra C5。upper secondary gate の `primary_onset_gain >= 20.0` 条件が不成立 (B4=16.0) で gate が発火しない
- **不採用理由**: 降順シーケンス (E6→C4) で同パターンの genuine carryover note (post-processing で除去される前提) が elif で reject → 別 candidate が accept → 回帰。segment_peaks レベルでの区別が困難
- **再検討条件**: E28 の extra C5 を解決する別アプローチが見つからない場合。post-processing 側での対応、または `primary_onset_gain` 閾値の引き下げ (20.0→12.0) も候補
- **関連データ**: E28 C5: onset_gain=0.87, backward_attack_gain=0.76, mute_dip=False

## HARMONIC_BAND_CENTS 拡大 (±40→±60-80 cents)

- **Issue**: #101 handoff で言及
- **日付**: 2026-03-31
- **概要**: FFT 帯域幅を広げて n_fft を変えずに低音域の bin 不足を解消
- **動機**: 96kHz + n_fft=4096 で D4 帯域に bin がゼロになる問題
- **不採用理由**: 未検証。隣接半音 (C4-D4 間 ~200 cents) との分離が悪化するトレードオフ。dynamic n_fft (#101) で代替解決
- **再検討条件**: dynamic n_fft のパフォーマンスオーバーヘッドが問題になった場合、または帯域幅拡大が他の品質指標を改善する場合

## Goertzel algorithm for note-band energy

- **Issue**: #101 handoff で言及
- **日付**: 2026-03-31
- **概要**: FFT の代わりに Goertzel アルゴリズムで特定周波数のエネルギーを直接計算
- **動機**: FFT の n_fft / zero-padding / spectral leakage 問題を根本的に回避。ブラウザ移植 (WebAudio) に最適
- **不採用理由**: 変更規模が大きく、現状の dynamic n_fft + 50ms window で十分な安定性が得られている
- **再検討条件**: ブラウザサイド実装フェーズ、またはリアルタイム処理でのパフォーマンス最適化が必要になった場合

## Subband onset detection for narrowband attacks

- **Issue**: #117 (closed — 前提の誤診)
- **日付**: 2026-04-02
- **概要**: mel spectrogram を 8 帯域に分割し、各帯域で独立に onset peak-picking → consensus voting (min_votes=6) で union
- **動機**: E143 A5 / E163 chord が broadband onset detection で検出されないと考えた
- **不採用理由**: 調査の結果、broadband onset は両方とも検出済み。E163 は waveform_stats kurtosis filter (#118)、E143 は octave alias + residual-decay (#119) が原因。subband detection は E163 を偶然的に改善（5ms 近接 onset 追加で AVC collector trigger）したが根本解決ではない
- **実験結果**: 8-band 全バンド → 2949 onset (14x増)、consensus min_votes=6 → 85 novel onset。gap-only filter で回帰なし (305 passed) だが E143 は改善せず
- **再検討条件**: genuine な narrowband onset 未検出が確認された場合（今回は broadband で検出済みだった）

## Short segment extension for octave-alias resolution

- **Issue**: #119
- **日付**: 2026-04-02
- **概要**: 0.5s 未満の短い segment を 0.5s まで延長（次 segment とぶつかる場合を除く）して FFT データ量を増やす
- **動機**: E143 A5 の segment が 141ms で短く、spectral resolution 不足で A4 octave alias が発生
- **不採用理由**: 0.32s / 0.5s いずれに延長しても primary は A4 のまま (residual-decay-no-reattack で棄却)。A4 の倍音構造 (440+880+1320...) が A5 (880+1760...) を必然的に上回るため、segment 長に関わらず octave alias は解消しない。0.5s では 15 regressions。
- **再検討条件**: `trimmed_from` と統合した形で、segment 管理の一般的改善として再設計する場合。octave alias 自体は別のアプローチ（residual-decay での octave 候補チェック等）が必要

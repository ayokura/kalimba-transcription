# Broadband patch vs per-note onset detection — 方針の詳細

> AGENTS.md「Recognizer Strategy Notes › Broadband patch vs per-note onset detection」の規範の背景・詳細。規範本体 (禁止事項・トリガー一覧) は AGENTS.md が正。このファイルは 2026-07-05 に AGENTS.md 軽量化のため詳細を移設したもの。

現在の recognizer は broadband onset detection（pure-numpy 化された spectral flux ベース。librosa からの移植コードだが、recognizer 自体は #187 / #193 で librosa-free）をベースに、個別の rescue/gate patch を積み上げて精度を上げている。一方 [#141](https://github.com/ayokura/kalimba-transcription/issues/141) では per-note onset detection という根本的な architecture 変更が提案されている。

**既定方針 (2026-07-04 改訂)**: broadband ベースは維持するが、**events.py への新規 suppression pass の追加は禁止** — トリガー 4 (下記) が限界域に達したため (pass 32 / gate reason ~40 vs fixture 35、2026-07-04 監査)。新規 pass に相当する変更は #141 research spike (research branch + dual-run) 経由でのみ試す。**非 pass 形の改修 (候補保持 / 降格 / provenance / 既存 pass の除去・簡素化) は従来どおり可**。per-note への全面移行は以下のトリガーのいずれかが発生した時点で判断する (トリガーの数値判定は ablation observatory — 第 2 期 S4 — で自動化予定):

1. **Patch が衝突する** — ある patch が別の patch の前提を壊し、全体として整合的な物理モデルにならなくなったとき
2. **broadband で物理的に検出不能な音が出る** — weak attack で spectral flux が閾値に届かないケース。broadband detection が通っているケース (2026-07-04 の 10.939s D5 など) は patch で拾える
3. **リアルタイム要求 (streaming transcription)** — batch 前提の broadband 解析では間に合わなくなったとき。per-note state machine (`OFF → ATTACK → BODY → LATE_DECAY`) への移行が必要
4. **Patch 数が fixture 数に近づく** — 一般化できないローカル解決が蓄積したとき

**streaming / WASM 適合性は直交**: broadband patch も per-note も FFT / band energy ベースで WASM 化できる。recognizer は既に librosa-free (#187 / #193 で pure-numpy 化済み) なので、ライブラリ独立は per-note を選ぶ理由にはならない。

**並行路線を推奨**: main line は patch で完成度を上げ、research line (別 branch) で per-note を実験的に検証する。patch で解けないケースを per-note 側で解く、が明確になった時点で merge を判断する。

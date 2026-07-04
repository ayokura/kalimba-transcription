# カリンバ記譜規約サーベイ (S7 記譜法 v1 決定支援)

作成: 2026-07-05 (S7 先行分)。**記譜法 v1 の決定はユーザー主体** — 本書は判断材料の整理であり、決定ではない。モックレンダリング 3 案は本書と対で提示する (代表フレーズ: きらきら星冒頭、PD)。

## 1. 世の中のカリンバ記譜方式

コミュニティで流通している方式は大きく 5 系統。

### 1a. 数字譜 (number tabs) — コミュニティの事実上の主流

- 音を移動ド数字で書く: C=1, D=2, … B=7。オクターブ上は数字の上に点 1 個 (2 オクターブ上は 2 個 / アスタリスク表記もある)
- 和音は括弧やスタックで併記 (「同時に弾く」印)
- リズムは規約が弱い: 下線 (8 分)、ダッシュ (延長)、スペースの広さ、小節線程度。**曲を知らないと音価が再現できない**ことが多い
- 17-key C 調の普及と一体で広まった (中国系メーカーの付属楽譜が数字譜)
- 現行実装との対応: `labelNumber` / NotationViews.numbered がこの系統

### 1b. 文字譜 (letter tabs)

- 音名 + オクターブ (C5, D5, …) の列。数字譜と併記されることも多い
- 現行実装との対応: NotationViews.western

### 1c. 縦型タブ (KTabS / vertical tablature)

- **tine の物理配置を模した縦スクロール表**。列 = tine、下から上へ時間が進む
- 音価・リズムも表せる完全表記 (KTabS はシェアウェア由来の規約)
- 「どの tine を弾くか」が視覚で分かるため初心者の再現性が最も高い、という主張がコミュニティに多い
- 弱点: 曲の構造 (フレーズ/反復) が読みにくい、幅が tuning 依存 (17/21/34-key で紙面が変わる)
- 現行実装との対応: なし (verticalDoReMi は「イベントごとの音名スタック」であり物理配置ではない)

### 1d. 五線譜

- 音価・休符・拍子の表現力は最強、既存楽譜資産と互換
- カリンバ奏者コミュニティでは「読めない人が多い」が繰り返し指摘される (数字譜が主流である理由)

### 1e. ハイブリッド (五線 + 数字併記)

- 市販カリンバ楽譜集の主流形式。五線の各音符の上/下に数字譜を併記
- 五線からリズムを、数字から tine を読む分業。読者層を選ばない
- 制作コストは最も高い (両方をレイアウトする)

## 2. リズム表現の現状

- 数字譜系のリズム規約は弱く、「原曲を知っている」前提で流通している
- 本プロジェクトの文脈では致命的: **自由演奏の転写が目的なので「原曲を知っている読者」を仮定できない**。リズム (音価・休符) を落とすと弾き戻せない
- → v1 でどの方式を選んでも、durationBeat / 休符の表示は必須になる見込み (S7 の「DoReMiScore への休符・音価導入」は方式決定に依存しない先行実装が可能)

## 3. v1 決定のための評価軸 (モック 3 案をこの軸で見る)

| 軸 | 数字譜+リズム | 縦型タブ | ハイブリッド |
|---|---|---|---|
| 弾き戻せるか (再現性) | リズム拡張すれば○ | ◎ (物理配置直結) | ◎ |
| カリンバ奏者の読解慣習 | ◎ (主流) | ○ (KTabS 知名度) | ○ |
| 認識結果の修正 UI との相性 | ◎ (イベント列と 1:1) | ○ (時間軸が縦) | △ (2 層同期が必要) |
| 印刷 / export 適性 | ◎ (テキスト寄り) | △ (縦長) | ○ |
| 実装コスト | 小 | 中 | 大 |
| tuning 汎用性 (17/21/34) | ◎ (移動ド) | △ (幅が配置依存) | ○ |

## 4. 実装現況との接続

- NotationViews は western / numbered / verticalDoReMi の 3 テキスト view (`notation.py`)。音価・休符は未表示 (schema の durationBeat は既存)
- ScoreEvent は startBeat / durationBeat / isGlissLike / gesture を持ち、リズム表示の材料は揃っている
- 奏法記号 (グリッサンド/トレモロ/ミュート) の v1 範囲は**ユーザーが決める** (sprint plan S7)。gesture 分類 (slide_chord 等) は既にあるため、記号を割り当てれば表示は可能

## 5. モック 3 案 (対になる deliverable)

sprint plan の指定どおり **五線 / 数字タブ / ハイブリッド** の 3 案を、きらきら星冒頭 (C C G G A A G / F F E E D D C、4/4) でレンダリングした。判定観点は「**楽譜だけ見て弾き戻せるか**」。

- 案 A: 数字譜 + リズム拡張 (下線=8 分、ダッシュ=延長、休符記号)
- 案 B: 縦型タブ (KTabS 風、tine 配置列 + 音価)
- 案 C: ハイブリッド (簡易五線 + 数字併記)

→ モックはセッション Artifact として提示 (ユーザー判定用)。判定後、選ばれた方式を `docs/notation-v1.md` に規約として文書化するのが S7 本体。

## 出典

- [KALIMBA CLASSES: How To Read Number Tabs](https://www.kalimbaclasses.com/kalimba-guides/how-to-read-number-tabs)
- [KalimbaTabs.net: Quick Guide On How To Read Kalimba Tablature](https://www.kalimbatabs.net/kalimba-tabs-tutorials/quick-guide-on-how-to-read-kalimba-tablature-pdf-for-beginners/)
- [Kalimba Magic: How to Read and Write Kalimba Tablature](https://www.kalimbamagic.com/info/how-to-play/how-to-read-and-write-kalimba-tablature)
- [thekalimba.com: Kalimba Tablatures: How to Read Them Easily](https://thekalimba.com/en/comment-lire-les-tablatures-kalimba/)
- [TabWhale: How to Read Number and Letter Kalimba Tabs](https://www.tabwhale.com/blog/post/how-to-read-number-and-letter-kalimba-tabs)

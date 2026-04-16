# API Contract Design — Phase α

## 目的

Web UI リビルドに向けて、API の契約（入出力仕様）を整理・設計する。
recognizer 内部がどう変わっても、この契約を満たせば UI は壊れない状態を目指す。

## 現行 API エンドポイント

| Method | Path | 用途 |
|--------|------|------|
| GET | `/api/health` | ヘルスチェック |
| GET | `/api/tunings` | 利用可能な調律一覧 |
| POST | `/api/transcriptions` | 音声ファイルの転写 |

## 現行リクエスト: POST /api/transcriptions

multipart/form-data:

| Field | Type | Required | 説明 |
|-------|------|----------|------|
| file | UploadFile | yes | WAV 音声ファイル |
| tuning | string (JSON) | yes | InstrumentTuning を JSON シリアライズ |
| debug | bool | no | debug 情報をレスポンスに含める |
| disabledRepeatedPatternPasses | string | no | 無効化する pattern pass の ID |
| midPerformanceStart | bool | no | 先頭の gap onset を除外 |
| midPerformanceEnd | bool | no | 末尾の gap onset を除外 |

## 現行レスポンス: TranscriptionResult

```
TranscriptionResult
├── instrumentTuning: InstrumentTuning
├── tempo: float (BPM)
├── events: ScoreEvent[]
├── candidateSlots: CandidateSlot[] (default: [])
├── notationViews: NotationViews
├── warnings: string[]
└── debug: dict | null
```

### ScoreEvent

```
ScoreEvent
├── id: string ("evt-1", "evt-2", ...)
├── startBeat: float (0.25 step)
├── durationBeat: float (min 0.25)
├── notes: ScoreNote[]
├── isGlissLike: bool
├── gesture: string ("run" | "gliss" | "chord" | "isolated" | "ambiguous")
└── alternateGroupings: AlternateGrouping[] | null
```

### ScoreNote

```
ScoreNote
├── key: int (物理キー位置, 1-indexed)
├── pitchClass: string ("C", "D#", ...)
├── octave: int
├── labelDoReMi: string ("do", "re.", "mi..", ...)
├── labelNumber: string ("1", "2'", "..3", ...)
└── frequency: float (Hz)
```

### NotationViews

```
NotationViews
├── western: string[]        ["C4", "D4|E4", "G4"]
├── numbered: string[]       ["1", "2|3", "5"]
└── verticalDoReMi: string[][] [["do","mi"], ["re"], ["sol"]]
```

### Web 側の型定義ギャップ（現状）

| フィールド | Python models | TypeScript types.ts | 状態 |
|-----------|--------------|---------------------|------|
| ScoreEvent.alternateGroupings | あり | **なし** | 未実装 |
| TranscriptionResult.candidateSlots | あり | **なし** | 未実装 |
| TuningNote.partials | あり | **なし** | 未反映 |
| TuningNote.layer | あり | **なし** | 未反映 |
| AlternateGrouping 型 | あり | **なし** | 未定義 |
| CandidateSlot 型 | あり | **なし** | 未定義 |

---

## Phase α 設計課題

### 1. TransactionId とサーバーサイド保管

**目的**: テストユーザーが転写結果を共有する際、zip ダウンロード不要にする。

**設計案**:
```
POST /api/transcriptions → レスポンスに transactionId を追加

TranscriptionResult
├── transactionId: string (UUID)    ← NEW
├── ... (既存フィールド)

サーバー側保管:
├── audio.wav
├── request.json (tuning, params)
├── response.json (TranscriptionResult)
└── debug.json (debug=True 時)
```

**検討事項**:
- 保管先: ローカルファイルシステム or オブジェクトストレージ
- 保持期間: テストフェーズは無期限、将来は TTL 付き
- 取得 API: `GET /api/transcriptions/{transactionId}` で結果再取得
- audio 取得: `GET /api/transcriptions/{transactionId}/audio` で元音声

### 2. セッション永続化

**現状**: sessionStorage のみ。ブラウザを閉じると Review 画面のデータが消える。

**設計案 (段階的)**:

Step 1: transactionId があれば API から再取得できるので、transactionId を localStorage に保管するだけで最低限の永続化は成立。

Step 2: ユーザーが保存したい楽譜データ (修正済み events 等) は別途永続化が必要。
- `PUT /api/transcriptions/{transactionId}/edits` で修正内容を保存
- or クライアントサイド (IndexedDB) で完結

### 3. 時間モデル — 絶対時間 vs 相対時間

**現状**: events は `startBeat` (拍位置) で表現。debug 内部では秒単位の絶対時間。

**Streaming 想定**:
- 「演奏開始判定点 = 0秒」モデル
- 録音全体の絶対時間ではなく、演奏部分の相対時間で扱う
- API レスポンスに `performanceStartSec` (演奏開始の絶対時間) を追加し、UI 側で相対変換

**設計案**:
```
TranscriptionResult
├── performanceStartSec: float | null  ← NEW (null = 録音先頭から)
├── performanceDurationSec: float      ← NEW
├── events: ScoreEvent[]
│   └── startTimeSec: float            ← NEW (絶対秒, debug でなく常に返す)
│   └── durationSec: float             ← NEW
│   └── startBeat: float               (既存, 維持)
│   └── durationBeat: float            (既存, 維持)
```

### 4. 音声再生

**要件**:
- 録音データの再生 (全体)
- event 単位の部分再生 (startTimeSec〜endTimeSec の切り出し)
- 将来: 合成音の再生 (MIDI 的な)

**設計案**:
- 音声データは transactionId 経由で取得: `GET /api/transcriptions/{transactionId}/audio`
- 部分再生はクライアントサイドで WebAudio API を使って切り出し再生
- API 側は startTimeSec/durationSec を返すだけでよい

### 5. 楽譜出力

#### 方向性 (2026-04-16 決定)

3つの出力形式を段階的に実装する。ドレミ visual がテストユーザー提供のブロッカー。

| 形式 | 主な用途 | 優先度 | 追加データ要否 |
|------|---------|--------|--------------|
| テキスト表記 | 開発・デバッグ・簡易共有 | 現行維持 | 不要（NotationViews で対応済み） |
| **ドレミ visual** | **カリンバ学習者（主ターゲット）** | **最優先** | ScoreAnnotation（手動補正メタデータ） |
| 五線譜 | 一般的な音楽共有・他楽器奏者 | 将来 | 拍子記号・正確な音価・調号（recognizer 拡張が前提） |

#### 5.1 テキスト表記（現行 NotationViews）

現行の `NotationViews` (western / numbered / verticalDoReMi) を維持。
開発用途が主だが、テキストコピーでの共有にも一定の有用性がある。
TypeScript 型定義の同期（§Web 側の型定義ギャップ）のみ対応。

#### 5.2 ドレミ visual 表記

カリンバ学習者が手書きで使っているドレミ縦並び楽譜のデジタル再現。
テストユーザー（録音提供者）が Web UI で転写結果を確認・印刷できるようにすることが目標。

**視覚要素** (手書き譜面から抽出):

| 要素 | 表現方法 |
|------|---------|
| 単音（主音） | テキスト横並び（ド、レ、ミ...） |
| オクターブ | 記号付き（ド. = 高い、_ド = 低い） — 現行 `labelDoReMi` |
| 同時打鍵 | 楕円グループ内に縦並び |
| 主音/伴奏の分離 | 主音行（上）+ 伴奏行（下）、縦線で接続 |
| シャープ/フラット | 音名に付加（ソ#、ミb） |
| セクション区切り | 水平線 + セクション番号 |
| 曲名 | 譜面上部に表示 |
| 歌詞 | 音名の上に表示（オプション） |
| リピート | ×2 等の記号 |

**レンダリング方式**: SVG（クライアントサイド描画）
- ベクター → 印刷で劣化しない
- ブラウザでそのまま表示可能
- CSS `@media print` で UI 要素を隠し、楽譜 SVG のみを印刷
- Canvas 経由で PNG 変換、または SVG → PDF 変換も可能
- クライアントサイド完結（将来の WASM 方針と整合）

**最終出力**: 印刷可能な形式が必須
- ブラウザ印刷（楽譜のみの clean CSS、UI 要素を含めない）
- 画像ファイル (PNG/SVG) ダウンロード
- PDF ダウンロード（いずれか）

**voice 判定ルール**:
- デフォルト: primary note（または最高音）= 主音、残りが伴奏
- 伴奏のみ（主音なし）のパターン:
  - 個別イベント: score_structure の `,` prefix 記法（例: `,C4`）、BWV147 E1 等
  - セクション単位: イントロ/アウトロ全体が伴奏コードのみ（score-in 等）
- 手動補正で override 可能（ScoreAnnotation の `voiceMode` / `voiceOverrides`、§5.4 参照）

**データソース**:
- 音名・オクターブ・同時打鍵 → `TranscriptionResult.events` から自動取得
- 行区切り・主音識別・曲名・歌詞・リピート → `ScoreAnnotation`（手動補正、§5.4 参照）

#### 5.3 五線譜（将来）

標準的な西洋音楽記譜法。実装には recognizer 側の拡張が前提:
- 拍子記号の検出（3/4、4/4 等）— 現在未実装
- 正確な音価（現行 `durationBeat` は 0.25 step 量子化で粗い）
- 調号の推定
- 小節線の配置

**技術候補**:
- MusicXML を中間形式（MuseScore 等への読み込みも可能）
- VexFlow (JS) でブラウザ描画
- abc.js（軽量な代替）
- サーバー側 LilyPond → SVG/PDF

Phase β（recognizer 安定化）以降に本格検討。

#### 5.4 ScoreAnnotation — 手動補正メタデータ

recognizer が自動で出せないが楽譜表示に必要な情報の器。
Web UI の編集機能でユーザーが付加する。

```jsonc
{
  "version": 1,
  "title": "曲名",
  // 行の区切り（自動認識不可 → ユーザーが UI で設定）
  // voiceMode: セクション全体の voice 指定（省略時は "default"）
  //   "default"            — primary/最高音=主音、残り=伴奏
  //   "accompaniment_only" — セクション全体が伴奏のみ（イントロ/アウトロ等）
  "lines": [
    {
      "id": "intro",
      "eventRange": [0, 5],
      "label": "イントロ",
      "voiceMode": "accompaniment_only"
    },
    {
      "id": "1",
      "eventRange": [6, 20],
      "label": "1"
      // voiceMode 省略 = "default"
    }
  ],
  // 個別イベントの voice override（セクション voiceMode より優先）
  // デフォルトは primary/最高音=主音。例外のみ指定。
  "voiceOverrides": {
    "42": "accompaniment_only"   // events[42] は伴奏のみ（主音なし）
  },
  // 歌詞（オプション）
  "lyrics": [
    { "eventIndex": 6, "text": "さ" },
    { "eventIndex": 7, "text": "ん" }
  ],
  // リピート記号（オプション）
  "repeats": [
    { "startEventIndex": 0, "endEventIndex": 5, "times": 2 }
  ]
}
```

**voice 判定の優先順位**:
1. `voiceOverrides` の個別イベント指定（最優先）
2. `lines[].voiceMode` のセクション指定
3. デフォルトルール: primary/最高音 = 主音

**使い分けの想定**:
- BWV147 のような曲: 大半は default、E1 だけ `voiceOverrides` で個別指定
- score-in のような曲: イントロ/アウトロのセクション全体を `voiceMode: "accompaniment_only"` で指定

**保存先**: transactionId に紐づけてサーバーサイド保管（§1 参照）、
またはクライアントサイド (IndexedDB) で完結する選択肢もあり。

**score_structure との関係**:
- score_structure.json はテスト用の楽譜定義（開発資産）
- ScoreAnnotation はユーザーが Web UI で作成する手動補正（プロダクト資産）
- 将来的に score_structure を「楽譜（不変）」と「録音固有データ」に分離する構想があり、
  ScoreAnnotation はその「録音固有データ」側と統合される可能性がある

#### 5.5 score_structure content 記法リファレンス

score_structure.json の `content` フィールドで使用する記法一覧:

| 記法 | 意味 | 例 |
|------|------|-----|
| `Note` | 単音（主音） | `C5`, `F#5` |
| `[A,B]` | 同時打鍵（A=主音, B=伴奏） | `[E5,C4]` |
| `<A,B,...>` | スライドコード | `<C5,A4>`, `<D5,B4,G4>` |
| `[A,<B,...>]` | 主音 + スライド伴奏 | `[E5,<G4,E4,C4>]` |
| `,Note` | 伴奏のみ（主音なし） | `,C4` |
| `,<A,...>` | 伴奏のみスライド（将来用） | `,<G4,E4,C4>` |

イベント間は ` / ` (space-slash-space) で区切る。

### 6. 開発ツール (Debug Capture) の分離

**現状**: `mode="debug"` でユーザー向けと開発者向けが同一コンポーネント内に混在。

**方針**:
- エンドプロダクトと開発ツールは別ルート (例: `/debug/*` は開発専用)
- Debug capture のデータ収集は transactionId + サーバー保管で置換可能
  - zip ダウンロードは廃止方向
  - expected performance 入力は開発ツール側のみ
- Range Map (カリンバキー配置図) は両方で使える共通コンポーネント

---

## 変更不要な部分

| 項目 | 理由 |
|------|------|
| GET /api/tunings の構造 | 安定しており UI でそのまま使われている |
| ScoreNote の基本フィールド | pitchClass/octave/frequency/label は十分 |
| InstrumentTuning の構造 | プリセットの仕組みが確立している |

---

## 次のアクション

1. ~~楽譜出力の方向性をユーザーと議論~~ → **決定済み** (2026-04-16, §5)
2. **ドレミ visual SVG レンダリングのプロトタイプ** — テストユーザー提供のブロッカー
3. transactionId + サーバー保管の実装 (最小 MVP)
4. TypeScript 型定義の同期 (alternateGroupings, candidateSlots)
5. startTimeSec/durationSec の ScoreEvent への追加
6. Review 画面の永続化 (transactionId ベースの再取得)
7. ScoreAnnotation の UI 編集機能 (行区切り、曲名、歌詞等)

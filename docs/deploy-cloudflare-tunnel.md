# Cloudflare Tunnel 経由でのテスター公開

モバイル端末 (スマホ) から動作確認できる公開URLを、開発マシン (WSL2) から直接 Cloudflare Tunnel 経由で配信する運用手順。getUserMedia が secure context を必要とする制約を HTTPS で満たし、Cloudflare Access + Google SSO で許可メールアドレスのみに限定する。

## 構成

```
テスター端末 → Cloudflare edge (Access 認証) → WSL2 内 cloudflared
                                                ├─ /api/*  → FastAPI :8000
                                                └─ /*      → Next.js :3000
```

- 単一ホスト名で same-origin 配信のため CORS 不要
- Next.js production build / FastAPI ともに WSL2 上で常駐
- `data/transactions/` はローカルFSに保管され、開発マシンと同じ内容

## 事前準備 (一度のみ)

### 1. cloudflared インストール

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

### 2. Cloudflare アカウントにログイン

```bash
cloudflared tunnel login
```

ブラウザが開き、対象ドメインの zone を選択するとローカルに証明書 (`~/.cloudflared/cert.pem`) が保存される。

### 3. Named tunnel を作成

```bash
cloudflared tunnel create kalimba-score
```

出力される tunnel UUID と credentials JSON path (`~/.cloudflared/<uuid>.json`) を控える。

### 4. DNS レコードを作成

```bash
cloudflared tunnel route dns kalimba-score kalimba.example.com
```

対象ドメインの zone 配下に CNAME が自動生成される。

### 5. Ingress 設定ファイル

`~/.cloudflared/config.yml` を作成:

```yaml
tunnel: <tunnel-uuid>
credentials-file: /home/<user>/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: kalimba.example.com
    path: ^/api/.*$
    service: http://localhost:8000
  - hostname: kalimba.example.com
    service: http://localhost:3000
  - service: http_status:404
```

### 6. Cloudflare Access Application

Zero Trust ダッシュボード (`one.dash.cloudflare.com`) で:

1. Access → Applications → Add an application → Self-hosted
2. Application name: `Kalimba Score`
3. Session duration: 24h 程度
4. Application domain: `kalimba.example.com`
5. Identity providers: 既存の Google SSO を選択
6. Policy: `Allow` で、対象テスターのメールアドレス (または Google Workspace グループ) を include rule に設定

## 起動 (毎回)

Terminal 3 枚 (または tmux / systemd user service):

### FastAPI (production mode、`--reload` なし)

```bash
KALIMBA_ALLOWED_ORIGINS=https://kalimba.example.com \
  uv run uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000 --workers 1
```

### Next.js (production build + start)

```bash
cd apps/web
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

### cloudflared

```bash
cloudflared tunnel run kalimba-score
```

## 動作確認

1. `https://kalimba.example.com` にスマホブラウザでアクセス
2. Cloudflare Access の Google SSO 画面に遷移、ログイン
3. 許可メールなら SimpleHome 画面が表示される
4. 調律選択 → マイク録音 or WAV アップロード → 転写結果 → Score 画面で再生位置ハイライト確認

## トラブルシュート

| 症状 | 確認項目 |
|---|---|
| CORS エラー | Next.js が `/api/*` を相対URLで呼んでいるか、cloudflared ingress で `/api/*` が :8000 に向いているか |
| 502 / 522 | `uvicorn` / `next start` が 127.0.0.1 で Listen しているか。cloudflared は localhost 接続前提 |
| Access 画面が出ない | Application domain が ingress hostname と完全一致しているか (path 限定でも domain 全体に Access が適用される) |
| マイク拒否 | HTTPS 済みの公開URLからのみ可。Access ログイン完了後のURLで確認すること |
| Transaction data が見えない | `KALIMBA_DATA_DIR` の指定がない場合は repo root 直下 `data/` を参照。必要なら env で固定 |

## 運用メモ

- 開発マシンを再起動したら 3 プロセス全て立て直す。systemd user service に包めば自動起動可 (TODO)
- 録音音声・memo は `data/transactions/<uuid>/` に保存される。テスターに共有 URL を渡す際はこの UUID を含む path を提示
- Tunnel を閉じる: `cloudflared tunnel delete kalimba-score` (Access Application も削除する)

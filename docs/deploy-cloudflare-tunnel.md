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

## 常駐運用 (systemd user service)

推奨は systemd user service で常駐させる方法。WSL2 再起動後も自動起動、セッション独立、`journalctl` でログ永続。

### 一度だけの前準備

```bash
sudo loginctl enable-linger $USER
systemctl --user is-system-running   # "running" が返ればOK
```

`linger` 有効化でログアウト後も user-level サービスが継続実行。`/etc/wsl.conf` に `[boot] systemd=true` が書かれていることも前提。

### Unit ファイル配置

`~/.config/systemd/user/` に 3 ファイルを配置する (内容は repo の `docs/deploy-cloudflare-tunnel.md` の末尾参照):

- `kalimba-api.service` — FastAPI / uvicorn
- `kalimba-web.service` — Next.js prod
- `kalimba-tunnel.service` — cloudflared (前2本に After/Wants)

配置後:

```bash
systemctl --user daemon-reload
systemctl --user enable --now kalimba-api kalimba-web kalimba-tunnel
```

### ログ追跡

```bash
journalctl --user -u kalimba-api -f
journalctl --user -u kalimba-web -f
journalctl --user -u kalimba-tunnel -f
```

### 本番 worktree 分離 (2026-06-13 以降の構成)

本番サービスは開発用 worktree からではなく、**専用 worktree `~/kalimba-prod`** から起動する。開発側でのブランチ checkout / build / `.next` 削除が本番に影響しないようにするため (2026-06-13 のダウンインシデントの再発防止)。

```bash
# 初期セットアップ (済)
git worktree add ~/kalimba-prod main
git -C ~/kalimba-prod checkout --detach origin/main   # ローカル main を開発側に解放
cd ~/kalimba-prod && uv sync --dev && npm install && (cd apps/web && npm run build)
```

prod worktree は **detached HEAD で origin/main を指す** (ローカル `main` ブランチは開発 worktree 側で通常通り checkout できる)。unit の `WorkingDirectory` は `~/kalimba-prod` 系を指す。**データディレクトリだけは開発 worktree 側を共有**する (`KALIMBA_DATA_DIR=/home/<user>/kalimba-transcription/data`) — triage 等の分析ツールが同じ transaction 群を直接読むため。

### デプロイ (main 更新の本番反映)

```bash
cd ~/kalimba-prod && git fetch origin && git checkout --detach origin/main

# API 側 (Python) 変更を含む場合
systemctl --user restart kalimba-api

# Web 側 (TSX/CSS) 変更を含む場合 — build 必須
(cd ~/kalimba-prod/apps/web && npm run build)
systemctl --user restart kalimba-web

# 依存変更 (pyproject/uv.lock, package.json) を含む場合は事前に
(cd ~/kalimba-prod && uv sync --dev)   # または npm install

# cloudflared config.yml 変更後
systemctl --user restart kalimba-tunnel
```

### 手動起動 (systemd を使わない場合)

Terminal 3 枚 (または tmux) で以下を同時起動:

```bash
# FastAPI (production、`--reload` なし)
KALIMBA_ALLOWED_ORIGINS=https://kalimba.example.com \
  uv run uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000 --workers 1

# Next.js (production build + start)
cd apps/web && npm run build && npm run start -- --hostname 127.0.0.1 --port 3000

# cloudflared
cloudflared --config ~/.cloudflared/config.yml tunnel run kalimba-score
```

## systemd unit リファレンス

### `~/.config/systemd/user/kalimba-api.service`

```ini
[Unit]
Description=Kalimba Score API (FastAPI / uvicorn, production)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<user>/kalimba-prod
Environment=PATH=/home/<user>/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=KALIMBA_ALLOWED_ORIGINS=https://<your-domain>
Environment=KALIMBA_DATA_DIR=/home/<user>/kalimba-transcription/data
ExecStart=/home/<user>/.local/bin/uv run uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### `~/.config/systemd/user/kalimba-web.service`

```ini
[Unit]
Description=Kalimba Score Web (Next.js production)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<user>/kalimba-prod/apps/web
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=NODE_ENV=production
ExecStart=/usr/local/bin/npm run start -- --hostname 127.0.0.1 --port 3000
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### `~/.config/systemd/user/kalimba-tunnel.service`

```ini
[Unit]
Description=Cloudflare Tunnel for Kalimba Score
After=network-online.target kalimba-api.service kalimba-web.service
Wants=network-online.target kalimba-api.service kalimba-web.service

[Service]
Type=simple
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/cloudflared --config /home/<user>/.cloudflared/config.yml tunnel run kalimba-score
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
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

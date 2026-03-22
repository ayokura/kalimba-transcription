import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "404 | Kalimba Score",
  description: "The requested page could not be found.",
};

export default function GlobalNotFoundPage() {
  return (
    <html lang="ja">
      <body>
        <main className="shell">
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Not Found</p>
                <h2>ページが見つかりません</h2>
              </div>
            </div>
            <p className="muted">指定されたページは存在しません。トップページまたは debug capture 画面へ戻ってください。</p>
            <div className="row wrap">
              <Link href="/" className="debug-link-card">
                <strong>トップページへ</strong>
                <span>利用者向け採譜画面を開きます。</span>
              </Link>
              <Link href="/debug/capture" className="debug-link-card">
                <strong>Debug capture へ</strong>
                <span>手動テスト用の再現データ収集画面を開きます。</span>
              </Link>
            </div>
          </section>
        </main>
      </body>
    </html>
  );
}

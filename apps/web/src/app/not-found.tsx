export default function NotFoundPage() {
  return (
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
          <a href="/" className="debug-link-card">
            <strong>トップページへ</strong>
            <span>利用者向け採譜画面を開きます。</span>
          </a>
          <a href="/debug/capture" className="debug-link-card">
            <strong>Debug capture へ</strong>
            <span>手動テスト用の再現データ収集画面を開きます。</span>
          </a>
        </div>
      </section>
    </main>
  );
}

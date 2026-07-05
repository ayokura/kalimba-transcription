"use client";

import { useCallback, useState } from "react";

/**
 * transaction id の短縮表記 (先頭 8 文字 = gt-drafts 系 API の tx8 と同じ規約)。
 * クリックでフル UUID をコピーする。Link や button の内側に置かれるため、
 * ネスト不正 (button in button) を避けて code + role="button" で実装し、
 * クリックは親のナビゲーションへ伝播させない。
 */
export function TxIdBadge({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const short = id.slice(0, 8);

  const handleCopy = useCallback(
    (e: React.MouseEvent | React.KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!navigator.clipboard) return;
      navigator.clipboard
        .writeText(id)
        .then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        })
        .catch(() => {});
    },
    [id],
  );

  return (
    <code
      className={`tx-id-badge${copied ? " copied" : ""}`}
      title={`${id} (クリックでフル ID をコピー)`}
      role="button"
      tabIndex={0}
      onClick={handleCopy}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") handleCopy(e);
      }}
    >
      {copied ? "コピー済" : short}
    </code>
  );
}

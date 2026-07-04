"use client";

import { useCallback, useState, type RefObject } from "react";

// S7: 譜面 export (SVG / PNG)。テスターが「持ち帰れる成果物」を得るための
// クライアントサイド書き出し。表示中の DoReMiScore SVG を複製し、CSS 変数
// (var(--ink) 等) をアプリ外でも解決できる固定値へ焼き込んでから保存する。

const EXPORT_INK = "#1a1a1a";
const EXPORT_BG = "#ffffff";
const EXPORT_FONT = "Georgia, 'Times New Roman', serif";
const PNG_SCALE = 2;

function prepareSvg(svg: SVGSVGElement): { markup: string; width: number; height: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  // currentColor / フォントは root の style から継承されるため、export 用の
  // 固定値をここで確定させる (アプリの CSS 変数はファイル単体では消える)。
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.style.color = EXPORT_INK;
  clone.style.fontFamily = EXPORT_FONT;
  clone.style.backgroundColor = EXPORT_BG;
  clone.style.width = "";
  clone.style.height = "";
  // debug オーバーレイは成果物に含めない。
  clone.querySelectorAll(".score-debug-overlay").forEach((el) => el.remove());
  const viewBox = svg.viewBox.baseVal;
  const width = viewBox?.width || svg.clientWidth || 800;
  const height = viewBox?.height || svg.clientHeight || 200;
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  return { markup: new XMLSerializer().serializeToString(clone), width, height };
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function ScoreExportButtons({
  scoreAreaRef,
  fileBaseName,
}: {
  scoreAreaRef: RefObject<HTMLElement | null>;
  fileBaseName: string;
}) {
  const [error, setError] = useState<string | null>(null);

  const findSvg = useCallback((): SVGSVGElement | null => {
    return scoreAreaRef.current?.querySelector<SVGSVGElement>(".doremi-score-container svg") ?? null;
  }, [scoreAreaRef]);

  const exportSvg = useCallback(() => {
    const svg = findSvg();
    if (!svg) {
      setError("譜面が見つかりません");
      return;
    }
    setError(null);
    const { markup } = prepareSvg(svg);
    downloadBlob(
      new Blob([markup], { type: "image/svg+xml;charset=utf-8" }),
      `${fileBaseName}.svg`,
    );
  }, [findSvg, fileBaseName]);

  const exportPng = useCallback(() => {
    const svg = findSvg();
    if (!svg) {
      setError("譜面が見つかりません");
      return;
    }
    setError(null);
    const { markup, width, height } = prepareSvg(svg);
    const image = new Image();
    const svgUrl = URL.createObjectURL(new Blob([markup], { type: "image/svg+xml;charset=utf-8" }));
    image.onload = () => {
      URL.revokeObjectURL(svgUrl);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(width * PNG_SCALE);
      canvas.height = Math.round(height * PNG_SCALE);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setError("PNG の生成に失敗しました");
        return;
      }
      ctx.fillStyle = EXPORT_BG;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) {
          setError("PNG の生成に失敗しました");
          return;
        }
        downloadBlob(blob, `${fileBaseName}.png`);
      }, "image/png");
    };
    image.onerror = () => {
      URL.revokeObjectURL(svgUrl);
      setError("PNG の生成に失敗しました");
    };
    image.src = svgUrl;
  }, [findSvg, fileBaseName]);

  return (
    <div className="score-export-row">
      <button type="button" className="score-export-btn" onClick={exportSvg}>
        SVG 保存
      </button>
      <button type="button" className="score-export-btn" onClick={exportPng}>
        PNG 保存
      </button>
      <button type="button" className="score-export-btn" onClick={() => window.print()}>
        印刷
      </button>
      {error && <span className="score-export-error">{error}</span>}
    </div>
  );
}

import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Kalimba Score",
  description: "Record kalimba audio and convert it into multiple notation styles.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}

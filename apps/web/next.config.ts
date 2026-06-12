import type { NextConfig } from "next";

const API_PROXY_TARGET = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // 本番 (next start) は .next を serve する。ブランチ検証用の dev server は
  // NEXT_DIST_DIR=.next-dev で分離し、本番ビルド成果物 (.next) に触れないこと。
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  experimental: {
    globalNotFound: true,
  },
  typedRoutes: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_PROXY_TARGET}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

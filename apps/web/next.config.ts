import type { NextConfig } from "next";

const API_PROXY_TARGET = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // 本番 (next start) は .next を serve する。ブランチ検証用の dev server は
  // NEXT_DIST_DIR=.next-dev で分離し、本番ビルド成果物 (.next) に触れないこと。
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  // prod の zero-downtime デプロイ (socket-proxyd + blue-green standalone backend)
  // 用に env gate で有効化する。未設定なら通常ビルド = CI の `next start` / dev は無影響。
  output: process.env.NEXT_OUTPUT_STANDALONE ? "standalone" : undefined,
  // deploy skew 対策: prod build 時のみ commit SHA を焼き込む (?dpl= / x-deployment-id)。
  // mismatch 検知で client-side navigation を full page reload に落とす。未設定なら無効。
  deploymentId: process.env.NEXT_DEPLOYMENT_ID,
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

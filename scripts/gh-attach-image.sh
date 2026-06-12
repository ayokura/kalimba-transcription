#!/usr/bin/env bash
# Issue/PR に画像を貼るためのヘルパー。
#
# gh CLI / GitHub API にはコメントへの画像添付機能がない (Web UI のアップロードは
# 非公開エンドポイント) ため、merge しない専用 `assets` ブランチに Git Data API で
# 画像を積み、raw.githubusercontent.com の URL を Markdown で参照する。
#
# Usage:
#   scripts/gh-attach-image.sh <image-file> <dest-path>
#   scripts/gh-attach-image.sh /tmp/screenshot.png pr-190/review-ui-final.png
#
# 出力: raw URL と、そのまま Issue/PR に貼れる Markdown スニペット。
set -euo pipefail

IMAGE_FILE="${1:?usage: gh-attach-image.sh <image-file> <dest-path>}"
DEST_PATH="${2:?usage: gh-attach-image.sh <image-file> <dest-path>}"
ASSETS_BRANCH="assets"

REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - "$IMAGE_FILE" > "$TMP_DIR/blob.json" << 'PYEOF'
import base64, json, sys
data = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"content": data, "encoding": "base64"}, sys.stdout)
PYEOF

BLOB_SHA=$(gh api "repos/$REPO/git/blobs" --input "$TMP_DIR/blob.json" --jq '.sha')

# assets ブランチが既にあれば、その tree を base にして履歴を積む
PARENT_SHA=$(gh api "repos/$REPO/git/ref/heads/$ASSETS_BRANCH" --jq '.object.sha' 2>/dev/null || true)
BASE_TREE=""
if [[ -n "$PARENT_SHA" ]]; then
  BASE_TREE=$(gh api "repos/$REPO/git/commits/$PARENT_SHA" --jq '.tree.sha')
fi

python3 - "$DEST_PATH" "$BLOB_SHA" "$BASE_TREE" > "$TMP_DIR/tree.json" << 'PYEOF'
import json, sys
payload = {"tree": [{"path": sys.argv[1], "mode": "100644", "type": "blob", "sha": sys.argv[2]}]}
if sys.argv[3]:
    payload["base_tree"] = sys.argv[3]
json.dump(payload, sys.stdout)
PYEOF

TREE_SHA=$(gh api "repos/$REPO/git/trees" --input "$TMP_DIR/tree.json" --jq '.sha')

if [[ -n "$PARENT_SHA" ]]; then
  COMMIT_SHA=$(gh api "repos/$REPO/git/commits" \
    -f message="assets: add $DEST_PATH" -f tree="$TREE_SHA" -f "parents[]=$PARENT_SHA" --jq '.sha')
  gh api -X PATCH "repos/$REPO/git/refs/heads/$ASSETS_BRANCH" -f sha="$COMMIT_SHA" --jq '.ref' > /dev/null
else
  COMMIT_SHA=$(gh api "repos/$REPO/git/commits" \
    -f message="assets: add $DEST_PATH" -f tree="$TREE_SHA" --jq '.sha')
  gh api "repos/$REPO/git/refs" -f ref="refs/heads/$ASSETS_BRANCH" -f sha="$COMMIT_SHA" --jq '.ref' > /dev/null
fi

RAW_URL="https://raw.githubusercontent.com/$REPO/$ASSETS_BRANCH/$DEST_PATH"
echo "url: $RAW_URL"
echo "markdown: ![$(basename "$DEST_PATH")]($RAW_URL)"

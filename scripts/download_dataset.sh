#!/usr/bin/env bash
# Download the flat dataset from a ModelScope dataset Git repository.
# Usage: DATA_REPO_URL=https://www.modelscope.cn/datasets/<ns>/<name>.git \
#          bash scripts/download_dataset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -z "${DATA_REPO_URL:-}" ]; then
  echo "DATA_REPO_URL is not set; assuming data/ is already present" >&2
  exit 0
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 "$DATA_REPO_URL" "$TMP/data"
mkdir -p data
cp -a "$TMP/data/." data/
echo "data downloaded into data/"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  if TAG=$(git -C "$ROOT" describe --tags --abbrev=0 2>/dev/null); then
    VERSION="$TAG"
  fi
fi

if [[ -z "$VERSION" ]]; then
  VERSION="1.0.0"
fi

if [[ "$VERSION" == v* ]]; then
  VERSION="${VERSION:1}"
fi

ARCH="$(uname -m)"
case "$ARCH" in
  arm64) ARCH_SUFFIX="arm64" ;;
  x86_64) ARCH_SUFFIX="x64" ;;
  *) ARCH_SUFFIX="$ARCH" ;;
esac

cd "$ROOT"
pyinstaller AirfoilFitter.spec --noconfirm --clean

APP_PATH="$ROOT/dist/AirfoilFitter.app"
if [[ ! -d "$APP_PATH" ]]; then
  APP_PATH="$ROOT/dist/AirfoilFitter/AirfoilFitter.app"
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "App bundle not found. Expected $ROOT/dist/AirfoilFitter.app or $ROOT/dist/AirfoilFitter/AirfoilFitter.app" >&2
  exit 1
fi

DMG_OUT="$ROOT/dist/AirfoilFitter-${VERSION}-${ARCH_SUFFIX}.dmg"
create-dmg \
  --volname "AirfoilFitter" \
  --window-size 600 400 \
  --icon-size 120 \
  --app-drop-link 450 200 \
  "$DMG_OUT" \
  "$APP_PATH"

echo "Built DMG: $DMG_OUT"

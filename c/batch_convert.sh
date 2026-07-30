#!/bin/bash
SRC="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4"
DST="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4-cfse"
CFSE="./cfse_pack"
cd /c/Users/max/Projects/colibri-hy3/c
export PATH=/mingw64/bin:/usr/bin:$PATH
for f in "$SRC"/out-*.safetensors; do
  b=$(basename "$f")
  dst="$DST/$b"
  # Check if already CFSE by looking for "cfse" in header
  if dd if="$dst" bs=1 skip=8 count=200 2>/dev/null | tr -d '\0' | grep -q cfse; then
    echo "SKIP $b (already CFSE)"
    continue
  fi
  echo "CONVERT $b ..."
  $CFSE "$f" "$dst" || { echo "FAILED $b"; exit 1; }
done
echo "ALLDONE"

#!/bin/bash
SRC="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4"
DST="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4-cfse"
CFSE="./cfse_pack"
for f in "$SRC"/out-*.safetensors; do
  bname=$(basename "$f")
  dst="$DST/$bname"
  # Check if already converted (CFSE marker in metadata)
  if [ -f "$dst" ] && strings "$dst" | grep -q 'cfse' 2>/dev/null; then
    echo "SKIP $bname (already CFSE)"
    continue
  fi
  echo "CONVERT $bname ..."
  $CFSE "$f" "$dst"
done
echo "DONE"

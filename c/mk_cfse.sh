#!/bin/bash
SRC="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4"
DST="/c/Users/max/Projects/quantProbe/Hy3-colibri-int4-cfse"
mkdir -p "$DST"
for f in "$SRC"/*.safetensors "$SRC"/*.json "$SRC"/*.py "$SRC"/tokenizer* "$SRC"/config* "$SRC"/generation*; do
  [ -f "$f" ] && ln -sf "$f" "$DST/" 2>/dev/null || echo "skip $f"
done
# Convert shard 05 (the small one we already converted) was already converted
echo "Links done. Converting more shards..."
for shard in "$SRC"/out-00005.safetensors; do
  bname=$(basename "$shard")
  if [ ! -f "$DST/$bname" ] || [ "$(stat -c%s "$DST/$bname")" -lt 1000000 ]; then
    ./cfse_pack "$shard" "$DST/$bname"
  fi
done
ls -la "$DST" | head -30

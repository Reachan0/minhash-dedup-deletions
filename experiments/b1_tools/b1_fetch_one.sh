#!/usr/bin/env bash
# Download one WET segment. Ranks 0..103 are checked against the frozen sha256; later ranks
# record their measured sha256 (no frozen value exists). Args: rank index name url sha
set -u
W=/path/to/scratch/dedupjoin
rank=$1; url=$4; sha=$5; out=$W/wet/$rank.warc.wet.gz
[ -f "$W/wet/$rank.ok" ] && exit 0
[ -f "$W/eng/$rank.parquet" ] && exit 0
for attempt in 1 2 3 4 5; do
  if curl -sSfL --retry 3 --retry-delay 5 -m 1200 -o "$out.part" "$url"; then
    got=$(sha256sum "$out.part" | cut -d' ' -f1)
    if [ "$sha" = "-" ] || [ "$got" = "$sha" ]; then
      mv "$out.part" "$out"; echo "$got" > "$W/wet/$rank.ok"; echo "$rank ok $(stat -c %s "$out")"; exit 0
    fi
    echo "$rank sha_mismatch"
  else
    echo "$rank curl_fail attempt=$attempt"
  fi
  rm -f "$out.part"; sleep $((attempt*10))
done
echo "$rank FAILED"; exit 1

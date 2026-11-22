#!/usr/bin/env bash
# Download one WET segment and verify its sha256 against the frozen manifest.
# Args: rank url sha256 bytes ; writes $W/wet/<rank>.warc.wet.gz and a status line.
set -u
W=/path/to/home/dedupjoin-postreveal
rank=$1; url=$2; sha=$3; bytes=$4
out=$W/wet/$rank.warc.wet.gz
if [ -f "$out.ok" ]; then echo "$rank skip"; exit 0; fi
for attempt in 1 2 3 4 5; do
  t0=$(date +%s.%N)
  if curl -sSfL --retry 3 --retry-delay 5 -m 900 -o "$out.part" "$url"; then
    got=$(sha256sum "$out.part" | cut -d' ' -f1)
    if [ "$got" = "$sha" ]; then
      mv "$out.part" "$out"; touch "$out.ok"
      echo "$rank ok attempt=$attempt bytes=$(stat -c %s "$out") secs=$(echo "$(date +%s.%N) - $t0" | bc)"
      exit 0
    fi
    echo "$rank sha_mismatch attempt=$attempt"
  else
    echo "$rank curl_fail attempt=$attempt"
  fi
  rm -f "$out.part"; sleep $((attempt*10))
done
echo "$rank FAILED"; exit 1

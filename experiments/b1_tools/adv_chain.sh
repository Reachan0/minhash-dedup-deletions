#!/usr/bin/env bash
cd /path/to/scratch/dedupjoin
OP=/path/to/scratch/dedupjoin/dedupjoin-newsnap-selection/code/operator
rm -f results/adv_bucket.jsonl
env/bin/python tools/adv_bucket.py $OP results/adv_bucket.jsonl two 1000 10000 30000 100000
env/bin/python tools/adv_bucket.py $OP results/adv_bucket.jsonl distinct 1000 10000 30000 100000
echo ADV_CHAIN_DONE

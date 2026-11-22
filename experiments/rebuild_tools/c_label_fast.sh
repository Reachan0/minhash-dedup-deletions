#!/usr/bin/env bash
# C labelling with 24 concurrent requests per model, then the analysis.
G=/path/to/home/dedupjoin-postreveal
cd $G
P=dedupjoin-newsnap-selection/code/post_reveal/c_content_audit.py
C_WORKERS=24 env/bin/python $P label gpt-6.2-sol > logs/c_label_gpt.log 2>&1 &
C_WORKERS=24 env/bin/python $P label qwen3.7-plus > logs/c_label_qwen.log 2>&1 &
wait
env/bin/python $P analyze > logs/c_analyze.log 2>&1
echo C_DONE

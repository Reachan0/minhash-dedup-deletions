#!/usr/bin/env bash
# label the kept-document audit sample with both models on AutoDL, then summarise.
# LLM_API_KEY comes from the caller's environment and is never written to disk.
A=/path/to/scratch/dedupjoin
cd $A/audit || exit 1
[ -n "${LLM_API_KEY:-}" ] || { echo "no LLM_API_KEY"; exit 1; }
export C_VARIANT=kept C_WORKERS=24
$A/env/bin/python audit_autodl.py label gpt-6.2-sol > $A/audit/label_sol.log 2>&1 &
$A/env/bin/python audit_autodl.py label qwen3.7-plus > $A/audit/label_qwen.log 2>&1 &
wait
$A/env/bin/python audit_autodl.py analyze
echo LABEL_DONE

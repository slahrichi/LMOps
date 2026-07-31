#!/usr/bin/env bash
set -euo pipefail

export GAD_WORK=${GAD_WORK:-$HOME/gad_run}
export GAD_REPO=${GAD_REPO:-$HOME/LMOps/gad}

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }
warn() { echo "WARN: $*" >&2; }

[ -r "$GAD_WORK/env.sh" ] || fail "missing $GAD_WORK/env.sh"
source "$GAD_WORK/env.sh"

python - <<'PY'
import flash_attn
import ray
import torch
import verl
import vllm
import wandb

print(
    "PASS: imports",
    f"torch={torch.__version__}",
    f"cuda_devices={torch.cuda.device_count()}",
)
PY

python -m pip check
pass "Python dependencies"

[ -r "$GAD_WORK/pyinclude/Python.h" ] || fail "missing vendored Python.h"
[ -d "$GAD_REPO/verl" ] || fail "missing verl checkout under $GAD_REPO"
[ -w "$GAD_WORK" ] || fail "$GAD_WORK is not writable"
pass "workspace, source, and headers"

command -v hf >/dev/null || fail "hf CLI is not installed"
python -c 'import huggingface_hub; print("PASS: huggingface_hub", huggingface_hub.__version__)'

for model in \
  /checkpoints/jasonjx/models/Qwen2.5-7B-Instruct/config.json \
  /checkpoints/xinyulin/pretrained_models/Qwen2.5-0.5B/config.json; do
  [ -r "$model" ] && pass "readable $model" || warn "not readable: $model"
done

if command -v sacctmgr >/dev/null && command -v sinfo >/dev/null; then
  if assoc=$(sacctmgr -nP show assoc user="$USER" format=User,Account,Partition,QOS 2>/dev/null) \
    && partition=$(sinfo -h -p h200 -o '%P %a %l' 2>/dev/null | head -1) \
    && [ -n "$assoc" ] && [ -n "$partition" ]; then
    printf '%s\n%s\n' "$assoc" "$partition"
    pass "SLURM association and H200 visibility"
  else
    warn "SLURM commands exist but the controller is unreachable from this host"
  fi
else
  warn "SLURM client commands unavailable on this host"
fi

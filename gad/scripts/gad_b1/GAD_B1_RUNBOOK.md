# Running a verl-based OPD job on a restricted GPU cluster — Team Runbook

**Worked example:** GAD (Generative Adversarial Distillation, arXiv 2511.10643) — student generator + co-trained discriminator, run from the upstream fork (`YTianZHU/verl` + `microsoft/LMOps/gad`) on an 8×H200 box.

**Use this when** you need to stand up a vendored-verl RL/distillation job on a sandboxed cluster (no root, restricted egress, small `/dev/shm`). The steps below are the validated happy path; every constraint has already been solved in the provided scripts — follow top to bottom.

**Owner:** Saad Lahrichi · validated 2026-07-22 on `claude-sandbox-pod` (8×H200), work on the shared NFS home `/home/saadlahrichi`.

---

## 0. Cluster constraints you design around
| Constraint | What it forces |
|---|---|
| No root / no `apt` / no nvcc | prebuilt wheels only; build any missing headers from source |
| `/dev/shm` = 63 MB | vLLM **TP=1** (not TP>1) and dataloader **num_workers=0** |
| `/tmp` = 1 GB tmpfs | set `TMPDIR` to the big disk |
| HF blocked on-box; pip/git egress OK | download models/data on an egress box → shared NFS home |
| Box is ephemeral; home is shared NFS | keep code/ckpts/docs on the home; they survive resets & are visible to your login node |

## 1. One-time setup

```bash
export WORK=/home/saadlahrichi/gad_run && mkdir -p $WORK/{tmp,pip_cache,hf}
```

**1a. Virtualenv** (`python -m venv` is unavailable — use `virtualenv`):
```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv -p python3.10 $WORK/venv && source $WORK/venv/bin/activate
```

**1b. `env.sh`** — source before every command. Sets the big-disk `TMPDIR`, the header path (step 1d), and the pip proxy. Uses the shared `.hf_token.sh` if present.
```bash
cat > $WORK/env.sh <<'EOF'
export GAD_WORK=/home/saadlahrichi/gad_run
export HF_HOME=$GAD_WORK/hf PIP_CACHE_DIR=$GAD_WORK/pip_cache
export TMPDIR=$GAD_WORK/tmp TEMP=$TMPDIR TMP=$TMPDIR
export C_INCLUDE_PATH=$GAD_WORK/pyinclude:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$GAD_WORK/pyinclude:$CPLUS_INCLUDE_PATH
export TOKENIZERS_PARALLELISM=false
export https_proxy=http://10.0.2.2:57269 http_proxy=http://10.0.2.2:57269
export HTTPS_PROXY=$https_proxy HTTP_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16 NO_PROXY=$no_proxy
[ -f $GAD_WORK/venv/bin/activate ] && source $GAD_WORK/venv/bin/activate
[ -f $GAD_WORK/.hf_token.sh ] && source $GAD_WORK/.hf_token.sh
EOF
source $WORK/env.sh
```
> The `10.0.2.2` proxies are sandbox-internal — **do not** source this `env.sh` on your own login node.

**1c. Install the stack** (pins are required — vLLM 0.8.5 has no upper bounds, so pip otherwise pulls incompatible majors):
```bash
pip install --upgrade pip wheel
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install "vllm==0.8.5"
pip install "transformers==4.51.3" "ray[default]==2.43.0" "tokenizers>=0.21.1,<0.22" \
            "protobuf==4.25.9" "setuptools<81"          # setuptools<81: verl needs pkg_resources
pip install pandas datasets accelerate codetiming hydra-core liger-kernel "tensordict<=0.6.2" \
            torchdata wandb pybind11 peft "pyarrow>=19.0.0" pylatexenc rouge-score latex2sympy2 \
            tabulate sentence_transformers
pip install "protobuf==4.25.9"                          # re-pin (sentence_transformers bumps it)
# flash-attn: prebuilt wheel matching torch2.6/cu12/cp310/cxx11abiFALSE (no nvcc to build)
curl -sL -o $WORK/tmp/fa.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install $WORK/tmp/fa.whl
# the two repos (algorithm fork + orchestration), editable, no deps
cd /home/saadlahrichi/LMOps/gad
git clone https://github.com/YTianZHU/verl.git      # branches: seqkd / warmup / gad / eval
pip install -e ./verl --no-deps && pip install -e . --no-deps
```

**1d. Python headers** (Triton compiles a CUDA helper at runtime and needs `Python.h`; `python3.10-dev` is absent and apt is blocked):
```bash
cd $WORK && git clone --depth 1 --branch v3.10.12 https://github.com/python/cpython cpython310
cd cpython310 && ./configure >/dev/null 2>&1        # generates pyconfig.h
mkdir -p $WORK/pyinclude && cp -r Include/* pyconfig.h $WORK/pyinclude/
```

**1e. Verify:**
```bash
python -c "import torch,verl,vllm,flash_attn,ray; print('OK', torch.__version__, torch.cuda.device_count())"
```

## 2. Models & data (fetch on an egress box → shared home)
- **Models** are already staged: 7B `/checkpoints/jasonjx/models/Qwen2.5-7B-Instruct`, 0.5B `/checkpoints/xinyulin/pretrained_models/Qwen2.5-0.5B`.
- **Teacher data** — on a box with real egress (e.g. your login node), using the shared venv and your own egress:
```bash
source /home/saadlahrichi/gad_run/venv/bin/activate && export HF_TOKEN=<token>
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
hf download ytz20/LMSYS-Chat-GPT-5-Chat-Response --repo-type dataset --local-dir ~/gad_run/data/hf_dl
cd ~/gad_run/data
cp hf_dl/data/train-00000-of-00001.parquet lmsys_train-00000-of-00001.parquet
cp hf_dl/data/test-00000-of-00001.parquet  lmsys_test-00000-of-00001.parquet
```

## 3. Run a job
All launchers are in `~/gad_run/`, self-source `env.sh`, and bake in the required flags (see §4). Run from `~/LMOps/gad` (the GAD stage needs `tools/merge_model2hf.py`).

```bash
cd /home/saadlahrichi/LMOps/gad
# 1) Smoke — validate the whole loop in ~2 min (0.5B, 2 GPUs). Do this first.
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash ~/gad_run/run_gad_smoke.sh > ~/gad_run/tmp/smoke.log 2>&1 &
#    expect: d_acc 0.5->0.9, d_loss down, D(teacher)-D(student) gap grows, no NaN.

# 2) Pick a stage (7B, all 8 GPUs). Checkout the matching branch first.
export N_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
(cd verl && git checkout seqkd) && bash ~/gad_run/run_b1_seqkd.sh  > ~/gad_run/tmp/seqkd.log  2>&1 &   # baseline
(cd verl && git checkout warmup) && bash ~/gad_run/run_b1_warmup.sh > ~/gad_run/tmp/warmup.log 2>&1 &   # stage 1 (SFT + D warmup)
(cd verl && git checkout gad) && RESUME_STEP=<N> bash ~/gad_run/run_b1_gad.sh > ~/gad_run/tmp/gad.log 2>&1 &  # stage 2 (adversarial, resumes a warmup ckpt)
```
Hyperparams (paper/fork, baked in): actor & discriminator lr 1e-6, grad_clip 0.2, GRPO n=8, temp 0.8, kl_loss_coef 0.001 (low_var_kl), critic_warmup 10 (warmup)/0 (gad), 7B. Throughput ≈ 71 s/step, 750 steps/epoch.

## 4. Required flags (already in the scripts — don't drop them)
| Flag | Why |
|---|---|
| `actor_rollout_ref.rollout.tensor_model_parallel_size=1` | vLLM TP>1 NCCL can't bootstrap on 63 MB `/dev/shm`; 7B fits on one H200 |
| `+data.dataloader_num_workers=0` | DataLoader workers use `/dev/shm`; the `+` appends the (non-struct) key |
| `NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1`, `VLLM_USE_V1=0` | container has only `lo`/`tap0` and tiny shm |
| `trainer.logger=['console']`, `trainer.default_local_dir=$WORK/ckpts/...` | no wandb; checkpoints on big NFS disk |

## 5. Monitor & manage
```bash
grep -E "Training Progress|step:[0-9]" ~/gad_run/tmp/warmup.log | tail   # progress + per-step metrics
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader   # who holds GPU mem
ls ~/gad_run/ckpts/*/global_step_*                                        # ckpts (every save_freq=50, durable on NFS)
ray stop --force ; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9   # stop cleanly
```

## 6. Reusing this pattern for other verl/OPD jobs
The generic recipe this example demonstrates:
1. Pin `torch`/`vllm` and freeze the transitive deps that lack upper bounds (`transformers`, `ray`, `protobuf`, `setuptools<81`).
2. Get `flash-attn` as an ABI-matched prebuilt wheel; build `Python.h` from CPython source when `-dev` is missing.
3. On a small-`/dev/shm` container: **TP=1 + `dataloader_num_workers=0`**, and route NCCL over `lo`.
4. Fetch weights/data on an egress box into a shared NFS path; keep the run + ckpts there.
5. Prove the loop with a tiny-model smoke before scaling; checkpoint to durable storage because boxes are ephemeral.

Swap the fork/branches, model paths, and data for your own OPD variant; everything else carries over.

---
**Companion docs:** design/planning `pastry P2430570417 > trackB_bundle.md`; discriminator experiments `B2_MULTIHEAD_DISCRIMINATOR_DESIGN.md`; memory note `gad-b1-gpu-box-run`.

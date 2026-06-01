#!/bin/bash
# lib/00-constants.sh — VERSION, tuning constants, defaults
# Sourced by the main entry point early. Config file overrides may follow.

# ── Version ────────────────────────────────────────────────────
VERSION="2.2.0"

# ── Config paths ───────────────────────────────────────────────
CONFIG_DIR="${LLM_CONFIG_DIR:-${LLM_APP_HOME:+$LLM_APP_HOME/config}}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/.config/llm-server}"

# ── Server defaults ────────────────────────────────────────────
PORT="${LLM_PORT:-8081}"
HOST="0.0.0.0"
CTX_SIZE="${LLM_CTX_SIZE:-65536}"
CTX_EXPLICIT=0
[[ -n "${LLM_CTX_SIZE:-}" ]] && CTX_EXPLICIT=1

# ── Tuning constants ───────────────────────────────────────────
MAX_RESTARTS="${LLM_MAX_RESTARTS:-5}"
KEEP_ALIVE="${LLM_KEEP_ALIVE:-0}"
TUNE_LONG_TIMEOUT=0
SYSTEM_HEADROOM_MB=5120
COMPUTE_PER_GPU_MB=512
MIN_CRAM_MB=512
VRAM_OVERHEAD_PERCENT=130
SINGLE_GPU_HEADROOM_MB=4096
COMPUTE_FLOOR_MB=1024
UPDATE_DISMISS_DAYS=7

# ── Paths ──────────────────────────────────────────────────────
MODEL_DIR="${LLM_MODEL_DIR:-${LLM_APP_HOME:+$LLM_APP_HOME/models}}"
MODEL_DIR="${MODEL_DIR:-$HOME/ai_models}"
CACHE_DIR="${LLM_CACHE_DIR:-${LLM_APP_HOME:+$LLM_APP_HOME/cache}}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llm-server}"
PROBE_CACHE_DIR="${LLM_PROBE_CACHE_DIR:-$CACHE_DIR/probes}"
LOG_DIR="${LLM_LOG_DIR:-${LLM_APP_HOME:+$LLM_APP_HOME/logs}}"
LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$MODEL_DIR" "$CACHE_DIR" "$LOG_DIR" 2>/dev/null || true
SERVER_LOG="$LOG_DIR/llm-server.log"
BENCHMARK_LOG="$LOG_DIR/llm-bench.log"
UPDATE_DISMISS_FILE="$CACHE_DIR/update_dismissed"

# ── Update ─────────────────────────────────────────────────────
LLM_SERVER_REPO="${LLM_SERVER_REPO:-$HOME/llm-server}"

# ── Flag variables ─────────────────────────────────────────────
RETUNE=0; USER_TUNE_ROUNDS=""; DOWNLOAD=0; DRY_RUN=0; VERBOSE=0
BENCHMARK=0; CPU_ONLY=0; MODEL_ARG=""; USER_SERVER_BIN=""; USER_LIB_PATH=""
EXPLICIT_TUNE_CACHE=""; EXTRA_LLAMA_FLAGS=(); TUNE_USER_LOCKED_KEYS=()
KV_QUALITY="low"; KV_QUALITY_EXPLICIT=0
KV_PLACEMENT="${LLM_KV_PLACEMENT:-auto}"; KV_PLACEMENT_EXPLICIT=0
[[ -n "${LLM_KV_PLACEMENT:-}" ]] && KV_PLACEMENT_EXPLICIT=1
PARALLEL_SLOTS=""; SPEC_TYPE=""; SPEC_DRAFT_N_MAX=""
GPUS_FILTER=""; RAM_BUDGET_MB=0; MMPROJ_PATH=""
DO_UPDATE=0; SHOW_CONFIGS=0; DO_AI_TUNE=0; DO_UNLIMITED=0; DO_CONVERGE=0
BACKEND=""

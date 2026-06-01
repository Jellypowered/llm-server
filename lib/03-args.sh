#!/bin/bash
# lib/03-args.sh — Argument parsing, tune key normalization, user lock tracking.
# Sourced after constants and defaults are set.

# ── Normalize tune key ─────────────────────────────────────────
normalize_tune_key() {
    case "$1" in
        --mg|--main-gpu) echo "-mg" ;;
        --batch-size) echo "-b" ;;
        --ubatch-size) echo "-ub" ;;
        --ngl|--n-gpu-layers) echo "-ngl" ;;
        --n-cpu-moe|-ncmoe) echo "--n-cpu-moe" ;;
        --override-tensor) echo "-ot" ;;
        --threads|-t) echo "--threads" ;;
        --threads-batch|-tb) echo "--threads-batch" ;;
        --parallel|-np) echo "--parallel" ;;
        --flash-attn|-fa) echo "--flash-attn" ;;
        --reasoning|-rea) echo "--reasoning" ;;
        --kv-offload|-kvo|--no-kv-offload|-nkvo) echo "--kv-offload" ;;
        *) echo "$1" ;;
    esac
}

add_tune_lock_key() {
    local key="$1" existing
    [[ -z "$key" || "$key" != -* ]] && return 0
    for existing in "${TUNE_USER_LOCKED_KEYS[@]}"; do
        [[ "$existing" == "$key" ]] && return 0
    done
    TUNE_USER_LOCKED_KEYS+=("$key")
}

record_user_tune_locks() {
    local i key
    for (( i=0; i<${#EXTRA_LLAMA_FLAGS[@]}; i++ )); do
        [[ "${EXTRA_LLAMA_FLAGS[$i]}" == -* ]] || continue
        key=$(normalize_tune_key "${EXTRA_LLAMA_FLAGS[$i]}")
        add_tune_lock_key "$key"
    done
    [[ -n "$PARALLEL_SLOTS" ]] && add_tune_lock_key "--parallel"
    if (( KV_QUALITY_EXPLICIT )); then
        add_tune_lock_key "--cache-type-k"
        add_tune_lock_key "--cache-type-v"
    fi
    if (( KV_PLACEMENT_EXPLICIT )); then
        add_tune_lock_key "--kv-offload"
        add_tune_lock_key "--no-kv-offload"
    fi
    [[ "$CTX_EXPLICIT" -eq 1 ]] && add_tune_lock_key "--ctx-size"
    if [[ -n "$GPUS_FILTER" ]]; then
        add_tune_lock_key "--device"
        add_tune_lock_key "--tensor-split"
        add_tune_lock_key "--split-mode"
        add_tune_lock_key "-mg"
        add_tune_lock_key "-ngl"
        add_tune_lock_key "-ot"
        add_tune_lock_key "--n-cpu-moe"
    fi
    if [[ -n "$MMPROJ_PATH" ]]; then
        add_tune_lock_key "--mmproj"
    fi
    if [[ -n "$SPEC_TYPE" ]]; then add_tune_lock_key "--spec-type"; fi
}

# ── Argument parsing ───────────────────────────────────────────
# Parse arguments. Known flags are handled here; unknown flags are
# forwarded verbatim to llama-server.
_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --retune)    RETUNE=1; shift ;;
            --dry-run)   DRY_RUN=1; shift ;;
            --verbose|-v) VERBOSE=1; shift ;;
            --benchmark) BENCHMARK=1; shift ;;
            --cpu)       CPU_ONLY=1; shift ;;
            --model-dir) MODEL_DIR="$2"; shift 2 ;;
            --port)      PORT="$2"; shift 2 ;;
            --ctx-size)  CTX_SIZE="$2"; CTX_EXPLICIT=1; shift 2 ;;
            --server-bin) USER_SERVER_BIN="$2"; shift 2 ;;
            --lib-path)   USER_LIB_PATH="$2"; shift 2 ;;
            --kv-quality)
                case "${2:-}" in high|mid|low) ;; *) echo "Error: --kv-quality must be one of: high, mid, low"; exit 1 ;; esac
                KV_QUALITY="$2"; KV_QUALITY_EXPLICIT=1; shift 2 ;;
            --kv-placement)
                case "${2:-}" in auto|gpu|cpu) ;; *) echo "Error: --kv-placement must be one of: auto, gpu, cpu"; exit 1 ;; esac
                KV_PLACEMENT="$2"; KV_PLACEMENT_EXPLICIT=1; shift 2 ;;
            --no-kv-offload|-nkvo)
                KV_PLACEMENT="cpu"; KV_PLACEMENT_EXPLICIT=1; shift ;;
            --kv-offload|-kvo)
                KV_PLACEMENT="gpu"; KV_PLACEMENT_EXPLICIT=1; shift ;;
            --gpus)      GPUS_FILTER="$2"; shift 2 ;;
            --ram-budget) RAM_BUDGET_MB="$2"; shift 2 ;;
            --mmproj)      MMPROJ_PATH="$2"; shift 2 ;;
            --vision)      MMPROJ_PATH="auto"; shift ;;
            --download)  DOWNLOAD=1; shift ;;
            --update)    DO_UPDATE=1; shift ;;
            --show-configs) SHOW_CONFIGS=1; shift ;;
            --ai-tune)   DO_AI_TUNE=1; shift ;;
            --converge)  DO_CONVERGE=1; shift ;;
            --tune-cache) EXPLICIT_TUNE_CACHE="$2"; shift 2 ;;
            --unlimited|--placement-tune|--full-auto-tune)
                [[ -t 2 ]] && echo "warning: $1 is deprecated — placement scope is now auto-decided by model arch (MoE/dense) and GPU count. Use --ai-tune; the right flags will be unlocked." >&2
                shift ;;
            --tune-long-timeout) TUNE_LONG_TIMEOUT=1; shift ;;
            --rounds)    USER_TUNE_ROUNDS="$2"; shift 2 ;;
            --parallel)  PARALLEL_SLOTS="$2"; shift 2 ;;
            --spec-type)
                case "${2:-}" in draft-mtp|draft-simple|draft-eagle3|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache|none) ;; *) echo "Error: --spec-type must be one of: draft-mtp, draft-simple, draft-eagle3, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, none"; exit 1 ;; esac
                SPEC_TYPE="$2"; shift 2 ;;
            --spec-draft-n-max) SPEC_DRAFT_N_MAX="$2"; shift 2 ;;
            --keep-alive) KEEP_ALIVE=1; shift ;;
            --backend)   BACKEND="$2"; shift 2 ;;
            --version)     echo "llm-server v$VERSION"; exit 0 ;;
            --help|-h)
                sed -n '3,/^$/s/^# \?//p' "$0"
                exit 0
                ;;
            --)
                shift
                while [[ $# -gt 0 ]]; do EXTRA_LLAMA_FLAGS+=("$1"); shift; done
                ;;
            -*)
                EXTRA_LLAMA_FLAGS+=("$1"); shift
                if [[ $# -gt 0 && "$1" != -* ]]; then
                    if [[ -z "$MODEL_ARG" && ( "$1" == *.gguf || -e "$1" ) ]]; then
                        MODEL_ARG="$1"; shift
                    else
                        EXTRA_LLAMA_FLAGS+=("$1"); shift
                    fi
                fi
                ;;
            *)
                if [[ -z "$MODEL_ARG" ]]; then
                    MODEL_ARG="$1"; shift
                else
                    EXTRA_LLAMA_FLAGS+=("$1"); shift
                fi
                ;;
        esac
    done
}

# ── Validate parsed values ─────────────────────────────────────
_validate_args() {
    # Parse --ram-budget human units (e.g. "60G" → 61440 MB)
    if [[ -n "$RAM_BUDGET_MB" && "$RAM_BUDGET_MB" != "0" ]]; then
        case "$RAM_BUDGET_MB" in
            *[Gg]) RAM_BUDGET_MB=$(( ${RAM_BUDGET_MB%[Gg]} * 1024 )) ;;
            *[Mm]) RAM_BUDGET_MB=${RAM_BUDGET_MB%[Mm]} ;;
            *[0-9]) ;;
            *) echo "Error: --ram-budget must be a number with optional G/M suffix (e.g. 60G, 4096M, 4096)"; exit 1 ;;
        esac
        if ! [[ "$RAM_BUDGET_MB" =~ ^[0-9]+$ ]]; then
            echo "Error: --ram-budget must be a number with optional G/M suffix (e.g. 60G, 4096M, 4096)"; exit 1
        fi
    fi

    # Validate --port
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
        echo "Error: --port must be a number between 1 and 65535 (got: $PORT)"
        exit 1
    fi

    # Validate --ctx-size
    if ! [[ "$CTX_SIZE" =~ ^[0-9]+$ ]] || (( CTX_SIZE < 1 )); then
        echo "Error: --ctx-size must be a positive number (got: $CTX_SIZE)"
        exit 1
    fi

    # Validate KV placement
    case "$KV_PLACEMENT" in
        auto|gpu|cpu) ;;
        *) echo "Error: LLM_KV_PLACEMENT must be one of: auto, gpu, cpu"; exit 1 ;;
    esac
}

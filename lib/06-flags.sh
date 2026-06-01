#!/bin/bash
# lib/06-flags.sh — BASE_FLAGS, COMMON_FLAGS, flag building, memory checks.
# All flag computation is wrapped in _build_flags() — called from main after
# hardware detection and model detection are complete.

# ── Context shift flag (available immediately on source) ─────────
CONTEXT_SHIFT_FLAG=""

# ── KV placement helper (available immediately on source) ───────
_apply_kv_placement_flag() {
    local placement="$1"
    local -a filtered=()
    for f in "${COMMON_FLAGS[@]}"; do
        case "$f" in --kv-offload|-kvo|--no-kv-offload|-nkvo) continue ;; esac
        filtered+=("$f")
    done
    COMMON_FLAGS=("${filtered[@]}")
    case "$placement" in
        cpu)   COMMON_FLAGS+=(--no-kv-offload); KV_PLACEMENT_EFFECTIVE="cpu" ;;
        gpu)   COMMON_FLAGS+=(--kv-offload);    KV_PLACEMENT_EFFECTIVE="gpu" ;;
        auto|"") KV_PLACEMENT_EFFECTIVE="gpu" ;;
    esac
}

# ── Build flags (called from main after hardware + model init) ───
_build_flags() {
    # Context shift flag
    if (( HAS_SSM == 1 )); then
        CONTEXT_SHIFT_FLAG="--no-context-shift"
        echo "  SSM/Mamba hybrid → context-shift disabled"
    fi

    # Compute totals
    TOTAL_VRAM_MB=0
    BEST_GPU_VRAM=0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        (( TOTAL_VRAM_MB += GPU_VRAM_FREE[$i] )) || true
        (( GPU_VRAM_FREE[$i] > BEST_GPU_VRAM )) && BEST_GPU_VRAM=${GPU_VRAM_FREE[$i]}
    done

    # Memory Analysis
    TOTAL_MEM_MB=$(( TOTAL_VRAM_MB + RAM_AVAIL_MB ))
    ESTIMATED_TOTAL_NEEDED=$(( TOTAL_SIZE_MB + TOTAL_SIZE_MB / 10 + SYSTEM_HEADROOM_MB ))
    if (( TOTAL_SIZE_MB > TOTAL_MEM_MB )); then
        echo "⚠️  WARNING: Model (${TOTAL_SIZE_GB}GB) is larger than your total available memory (${TOTAL_MEM_MB}MB)."
        echo "   A system crash or heavy swapping is likely. Proceeding anyway..."
    elif (( ESTIMATED_TOTAL_NEEDED > TOTAL_MEM_MB )); then
        echo "ℹ️  INFO: Model + KV Cache (~${ESTIMATED_TOTAL_NEEDED}MB) is very close to your total memory (${TOTAL_MEM_MB}MB)."
        echo "   Expect potential instability or slow performance if memory fills up."
    else
        echo "✓ Memory: Model and context estimated to fit within available RAM/VRAM."
    fi

    FITS_ON_GPU=0
    (( TOTAL_SIZE_MB * VRAM_OVERHEAD_PERCENT / 100 <= TOTAL_VRAM_MB )) && FITS_ON_GPU=1

    if (( FITS_ON_GPU )); then RAM_AFTER_LOAD=$RAM_AVAIL_MB
    else
        RAM_ON_CPU=$(( TOTAL_SIZE_MB - TOTAL_VRAM_MB ))
        (( RAM_ON_CPU < 0 )) && RAM_ON_CPU=0
        RAM_AFTER_LOAD=$(( RAM_AVAIL_MB - RAM_ON_CPU ))
        (( RAM_AFTER_LOAD < 0 )) && RAM_AFTER_LOAD=0
    fi

    # Batch sizes
    if (( FITS_ON_GPU && BEST_GPU_VRAM > TOTAL_SIZE_MB + SINGLE_GPU_HEADROOM_MB )); then
        BATCH=8192; UBATCH=1024
        echo "  VRAM headroom large → batch=$BATCH ubatch=$UBATCH"
    elif (( FITS_ON_GPU )); then
        BATCH=4096; UBATCH=512
        echo "  Model fits on GPU → batch=$BATCH ubatch=$UBATCH"
    else
        BATCH=2048; UBATCH=512
        echo "  GPU+CPU split → batch=$BATCH ubatch=$UBATCH"
    fi

    # KV cache type
    _kv_bytes_per_layer_per_token=$(( HEAD_COUNT_KV * (KEY_LENGTH + VALUE_LENGTH) ))
    KV_FAMILY="standard"
    KV_FAMILY_NOTE=""
    if (( KV_LORA_RANK > 0 )); then
        KV_FAMILY="mla"
        _kv_elems_total=$(( LAYER_COUNT * CTX_SIZE * (KV_LORA_RANK + ROPE_DIM) ))
        KV_FAMILY_NOTE="MLA (kv_lora=${KV_LORA_RANK}+rope=${ROPE_DIM})"
    elif (( HAS_SSM == 1 )); then
        KV_FAMILY="ssm"
        if (( FULL_ATTN_INTERVAL > 0 )); then
            _attn_layers=$(( LAYER_COUNT / FULL_ATTN_INTERVAL )); (( _attn_layers < 1 )) && _attn_layers=1
            KV_FAMILY_NOTE="SSM/hybrid (${_attn_layers}/${LAYER_COUNT} attn, interval=${FULL_ATTN_INTERVAL})"
        elif (( HEAD_COUNT_KV == 0 )); then
            _attn_layers=0; KV_FAMILY_NOTE="Pure SSM (no growing KV)"
        else
            _attn_layers=$(( (LAYER_COUNT + 1) / 2 ))
            KV_FAMILY_NOTE="SSM/hybrid (~${_attn_layers}/${LAYER_COUNT} attn, default)"
        fi
        _kv_elems_total=$(( _attn_layers * CTX_SIZE * _kv_bytes_per_layer_per_token ))
    elif (( SLIDING_WINDOW > 0 )); then
        KV_FAMILY="iswa"
        _swa_period=6
        case "$MODEL_ARCH" in gemma2|cohere2|exaone4|llama4) _swa_period=4 ;; gemma3) _swa_period=6 ;; plamo3) _swa_period=8 ;; esac
        _full_layers=$(( (LAYER_COUNT + _swa_period - 1) / _swa_period ))
        _swa_layers=$(( LAYER_COUNT - _full_layers ))
        _swa_ctx=$CTX_SIZE; (( _swa_ctx > SLIDING_WINDOW )) && _swa_ctx=$SLIDING_WINDOW
        _kv_elems_total=$(( _full_layers * CTX_SIZE * _kv_bytes_per_layer_per_token + _swa_layers * _swa_ctx * _kv_bytes_per_layer_per_token ))
        KV_FAMILY_NOTE="ISWA (${_full_layers} full + ${_swa_layers} sw@${_swa_ctx})"
    else
        _kv_elems_total=$(( LAYER_COUNT * CTX_SIZE * _kv_bytes_per_layer_per_token ))
    fi

    if (( HEAD_COUNT_KV > 0 && KEY_LENGTH > 0 && VALUE_LENGTH > 0 )) || [[ "$KV_FAMILY" == "mla" ]]; then
        KV_Q4_MB=$(awk "BEGIN {printf \"%.0f\", $_kv_elems_total * 0.5625 / 1048576}")
        KV_Q8_MB=$(awk "BEGIN {printf \"%.0f\", $_kv_elems_total * 1.0625 / 1048576}")
        KV_F16_MB=$(awk "BEGIN {printf \"%.0f\", $_kv_elems_total * 2.0 / 1048576}")
    else
        KV_Q4_MB=$(( LAYER_COUNT * 70 )); KV_Q8_MB=$(( LAYER_COUNT * 140 )); KV_F16_MB=$(( LAYER_COUNT * 280 ))
    fi

    _kv_family_tag="${KV_FAMILY_NOTE:+ [${KV_FAMILY_NOTE}]}"
    case "$KV_QUALITY" in
        high)   KV_TYPE="f16";   echo "  KV cache: f16 (${KV_F16_MB}MB) — highest quality${_kv_family_tag}" ;;
        mid)    KV_TYPE="q8_0";  echo "  KV cache: q8_0 (${KV_Q8_MB}MB) — balanced (default)${_kv_family_tag}" ;;
        low)    KV_TYPE="q4_0";  echo "  KV cache: q4_0 (${KV_Q4_MB}MB) — minimum VRAM${_kv_family_tag}" ;;
        *)      echo "Error: --kv-quality must be one of: high, mid, low"; exit 1 ;;
    esac
    case "$KV_TYPE" in
        f16)   KV_TOTAL_MB=$KV_F16_MB ;; q8_0)  KV_TOTAL_MB=$KV_Q8_MB ;; q4_0) KV_TOTAL_MB=$KV_Q4_MB ;;
    esac

    # Max-context-fit suggestion
    if (( ! CTX_EXPLICIT )) && (( KV_TOTAL_MB > 0 )) && [[ -t 0 ]] && (( ${LLM_ASSUME_YES:-0} == 0 )); then
        _kv_bpt_b=$(( KV_TOTAL_MB * 1048576 / CTX_SIZE ))
        if (( _kv_bpt_b > 0 )); then
            _total_hw_mb=$(( TOTAL_VRAM_MB + RAM_AVAIL_MB ))
            _fixed_overhead_mb=$(( TOTAL_SIZE_MB + 8192 ))
            if (( _total_hw_mb > _fixed_overhead_mb + KV_TOTAL_MB )); then
                _kv_budget_mb=$(( _total_hw_mb - _fixed_overhead_mb ))
                _max_ctx_raw=$(( _kv_budget_mb * 1048576 / _kv_bpt_b ))
                _hw_cap_ctx=$_max_ctx_raw
                if (( CTX_TRAIN > 0 && CTX_TRAIN < _hw_cap_ctx )); then _hw_cap_ctx=$CTX_TRAIN; fi
                _suggested_ctx=$CTX_SIZE
                for _c in 32768 65536 131072 262144 524288 1048576 2097152 4194304; do
                    (( _c <= _hw_cap_ctx )) && _suggested_ctx=$_c
                done
                if (( _suggested_ctx >= CTX_SIZE * 2 )); then
                    _new_kv_mb=$(( _kv_bpt_b * _suggested_ctx / 1048576 ))
                    echo ""; echo "  Context headroom available:"
                    echo "    Requested:  ${CTX_SIZE} tokens → ${KV_TOTAL_MB}MB KV (${KV_TYPE})"
                    echo "    Max fit:    ${_suggested_ctx} tokens → ${_new_kv_mb}MB KV"
                    read -r -t 20 -p "  Use max context ${_suggested_ctx}? [y/N] " _ctx_ans || _ctx_ans="n"
                    _ctx_ans="${_ctx_ans:-n}"
                    if [[ "$_ctx_ans" =~ ^[Yy] ]]; then
                        _num=$_suggested_ctx; _den=$CTX_SIZE; CTX_SIZE=$_suggested_ctx
                        KV_Q4_MB=$(( KV_Q4_MB * _num / _den )); KV_Q8_MB=$(( KV_Q8_MB * _num / _den ))
                        KV_F16_MB=$(( KV_F16_MB * _num / _den ))
                        case "$KV_TYPE" in f16) KV_TOTAL_MB=$KV_F16_MB ;; q8_0) KV_TOTAL_MB=$KV_Q8_MB ;; q4_0) KV_TOTAL_MB=$KV_Q4_MB ;; esac
                        echo "  Context bumped to ${CTX_SIZE} — KV now ${KV_TOTAL_MB}MB"
                    fi
                fi
            fi
        fi
    fi

    # Hadamard K-cache & prompt cache
    KHAD_FLAG=""
    if [[ "$KV_TYPE" == "q4_0" || "$KV_TYPE" == "q8_0" ]]; then
        KHAD_FLAG="-khad"; echo "  Quantized KV cache → Hadamard K-transform enabled"
    fi
    CRAM_MB=$(( RAM_AFTER_LOAD / 10 ))
    (( CRAM_MB > 16384 )) && CRAM_MB=16384
    (( CRAM_MB < MIN_CRAM_MB )) && CRAM_MB=0
    if (( GPU_COUNT <= 1 )); then
        if (( CRAM_MB > 0 )); then echo "  Prompt cache: ${CRAM_MB}MB (10% of free RAM)"
        else echo "  Prompt cache: disabled (not enough free RAM)"; fi
    fi

    # Thread count
    THREADS_GEN=$PHYSICAL_CORES; THREADS_BATCH=$PHYSICAL_CORES
    echo "  Threads: gen=$THREADS_GEN batch=$THREADS_BATCH (${PHYSICAL_CORES} physical cores)"

    # Build flags arrays
    BASE_FLAGS=( -m "$MODEL_PATH" --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" )

    COMMON_FLAGS=( -m "$MODEL_PATH" --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" \
        --flash-attn on -b "$BATCH" -ub "$UBATCH" \
        --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
        --jinja --threads "$THREADS_GEN" --threads-batch "$THREADS_BATCH" )

    # KV placement
    KV_PLACEMENT_EFFECTIVE="gpu"
    case "$KV_PLACEMENT" in
        cpu)   _apply_kv_placement_flag cpu; echo "  KV placement: CPU (--no-kv-offload from settings/CLI)" ;;
        gpu)   _apply_kv_placement_flag gpu; echo "  KV placement: GPU (--kv-offload from settings/CLI)" ;;
        auto)  echo "  KV placement: auto (GPU first; reserves KV before MoE layers)" ;;
    esac

    # Reasoning flag
    USER_REASONING_EXPLICIT=0
    for _f in "${EXTRA_LLAMA_FLAGS[@]:-}"; do [[ "$_f" == "--reasoning" || "$_f" == "-rea" ]] && USER_REASONING_EXPLICIT=1; done
    if (( SUPPORTS_REASONING_FLAG && ! USER_REASONING_EXPLICIT )); then
        COMMON_FLAGS+=(--reasoning off); echo "  Reasoning: off by default for OpenAI-compatible responses"
    fi

    # Parallel slots
    [[ -n "$PARALLEL_SLOTS" ]] && COMMON_FLAGS+=(--parallel "$PARALLEL_SLOTS")

    # Conditional flags (IK vs mainline)
    if [[ "$IS_IK_LLAMA" == "1" ]]; then
        COMMON_FLAGS+=(--run-time-repack)
        [[ -n "$KHAD_FLAG" ]] && COMMON_FLAGS+=("$KHAD_FLAG")
        [[ -n "$CONTEXT_SHIFT_FLAG" ]] && COMMON_FLAGS+=("$CONTEXT_SHIFT_FLAG")
        COMMON_FLAGS+=(--defrag-thold 0.1)
        if (( IS_MOE == 1 && IS_IK_LLAMA == 1 )); then
            COMMON_FLAGS+=(-muge -ger); echo "  MoE model → -muge -ger enabled"
        fi
        (( GPU_COUNT > 0 && IS_IK_LLAMA == 1 )) && COMMON_FLAGS+=(-mqkv)
        # Prompt cache budget for multi-GPU
        if (( GPU_COUNT > 1 )); then
            MODEL_ON_GPU_MB=$(( TOTAL_SIZE_MB * VRAM_OVERHEAD_PERCENT / 100 ))
            (( MODEL_ON_GPU_MB > TOTAL_VRAM_MB )) && MODEL_ON_GPU_MB=$TOTAL_VRAM_MB
            VRAM_HEADROOM=$(( TOTAL_VRAM_MB - MODEL_ON_GPU_MB - KV_TOTAL_MB - COMPUTE_PER_GPU_MB * GPU_COUNT ))
            (( VRAM_HEADROOM < 0 )) && VRAM_HEADROOM=0
            CACHE_RAM_MB=$(( VRAM_HEADROOM / 2 )); (( CACHE_RAM_MB > 4096 )) && CACHE_RAM_MB=4096
            if (( CACHE_RAM_MB < 256 )); then
                CACHE_RAM_MB=0; MAX_CHECKPOINTS=0; COMMON_FLAGS+=(-cram 0 --ctx-checkpoints 0)
            else
                MAX_CHECKPOINTS=$(( CACHE_RAM_MB / 200 )); (( MAX_CHECKPOINTS < 2 )) && MAX_CHECKPOINTS=2
                (( MAX_CHECKPOINTS > 16 )) && MAX_CHECKPOINTS=16
                COMMON_FLAGS+=(-cram "$CACHE_RAM_MB" --ctx-checkpoints "$MAX_CHECKPOINTS")
            fi
            echo "  Prompt cache: ${CACHE_RAM_MB}MB (VRAM headroom: ${VRAM_HEADROOM}MB), checkpoints: ${MAX_CHECKPOINTS}"
        else
            (( CRAM_MB > 0 )) && COMMON_FLAGS+=(-cram "$CRAM_MB")
        fi
    else
        [[ -n "$CONTEXT_SHIFT_FLAG" ]] && COMMON_FLAGS+=("$CONTEXT_SHIFT_FLAG")
    fi

    # GPU offloading
    (( GPU_COUNT > 0 )) && COMMON_FLAGS+=(-ngl 999 -mg "${GPU_INDEX[${GPU_ORDER[0]}]}")

    # Speculative decoding
    if [[ -n "$SPEC_TYPE" && "$SPEC_TYPE" != "none" ]]; then
        COMMON_FLAGS+=(--spec-type "$SPEC_TYPE"); echo "  Speculative decoding: type=$SPEC_TYPE"
        [[ -n "$SPEC_DRAFT_N_MAX" ]] && COMMON_FLAGS+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
    fi

    # User passthrough flags
    if (( ${#EXTRA_LLAMA_FLAGS[@]} > 0 )); then
        echo "  User flags (passthrough): ${EXTRA_LLAMA_FLAGS[*]}"
    fi

    # ── Vision/mmproj resolution (needs KV_TOTAL_MB from above) ────
    _resolve_vision
    # ── Strategy selection (needs KV_TOTAL_MB + VRAM_NEEDED_MB) ───
    _select_strategy
}

# ── Resolve vision/mmproj (called after flag building) ─────────
_resolve_vision() {
    if [[ "$MMPROJ_PATH" == "auto" ]]; then
        MMPROJ_PATH=$(find_local_mmproj)
        [[ -z "$MMPROJ_PATH" ]] && MMPROJ_PATH=$(download_mmproj) || true
        if [[ -z "$MMPROJ_PATH" ]]; then
            echo "Error: Vision enabled but no compatible mmproj found." >&2
            echo "Place matching mmproj GGUF next to the model, or use --mmproj /path/to/mmproj.gguf." >&2
            exit 1
        fi
    fi
    VRAM_NEEDED_MB=$(( TOTAL_SIZE_MB * VRAM_OVERHEAD_PERCENT / 100 + KV_TOTAL_MB + COMPUTE_PER_GPU_MB ))
    if [[ -n "$MMPROJ_PATH" ]]; then
        if [[ -f "$MMPROJ_PATH" ]]; then
            if ! mmproj_matches_model "$MMPROJ_PATH"; then
                echo "Error: mmproj does not match model: $MMPROJ_PATH ($(mmproj_label "$MMPROJ_PATH"))" >&2
                exit 1
            fi
            MMPROJ_SIZE_MB=$(du -sm "$MMPROJ_PATH" 2>/dev/null | awk '{print $1}' || echo 0)
            VRAM_NEEDED_MB=$(( VRAM_NEEDED_MB + MMPROJ_SIZE_MB ))
            COMMON_FLAGS+=(--mmproj "$MMPROJ_PATH")
            echo "  Vision: mmproj loaded from $MMPROJ_PATH"
        else
            echo "Error: Vision enabled but mmproj file not found."; exit 1
        fi
    fi
    if (( IS_MOE == 0 )); then
        TOTAL_POOL_MB=$(( TOTAL_VRAM_MB + RAM_AVAIL_MB - SYSTEM_HEADROOM_MB ))
        check_memory_or_die "$TOTAL_POOL_MB" "total GPU+RAM"
    fi
}

# ── Strategy selection (called after vision + flag building) ────
_select_strategy() {
    choose_strategy() {
        (( GPU_COUNT == 0 )) && { echo "cpu_only"; return; }
        local best=${GPU_ORDER[0]}
        local single_gpu_usable=$(( GPU_VRAM_TOTAL[best] - 1024 ))
        (( single_gpu_usable < 0 )) && single_gpu_usable=0
        local single_gpu_needed=$(( TOTAL_SIZE_MB * 110 / 100 + KV_TOTAL_MB + COMPUTE_PER_GPU_MB ))
        (( single_gpu_needed <= single_gpu_usable )) && { echo "single_gpu"; return; }
        (( IS_MOE == 0 && VRAM_NEEDED_MB <= TOTAL_VRAM_MB )) && { echo "multi_gpu_dense"; return; }
        (( IS_MOE == 1 )) && { echo "moe_offload"; return; }
        echo "dense_cpu_offload"
    }
    STRATEGY=$(choose_strategy)
    if [[ -n "$MMPROJ_PATH" && "$STRATEGY" == "single_gpu" && GPU_COUNT -gt 1 ]]; then
        if (( IS_MOE )); then
            STRATEGY="moe_offload"; echo "  Vision: forcing moe_offload (mmproj needs extra VRAM)"
        else
            STRATEGY="multi_gpu_dense"; echo "  Vision: forcing multi_gpu_dense (mmproj needs extra VRAM)"
        fi
    fi
    echo ""; echo "Strategy: $STRATEGY"
}

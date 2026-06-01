#!/bin/bash
# lib/10-moe.sh — MoE expert placement & auto-tuning.
# Contains: load_placement_cache_file, _compute_gpu_kv_reserves, probe cache functions,
# system probe, phase-1 placement, hard ceilings, layer distribution, auto-tuning loop,
# bidirectional sweep recovery.

# Only MoE offload reaches here — all other strategies call run_with_restart (never returns)

# All MoE placement logic is wrapped in _run_moe_placement() —
# called from _main_launch when STRATEGY == "moe_offload".

# ── Helper functions ──────────────────────────────────────────
load_placement_cache_file() {
    local file="$1" line key value
    unset CACHED_GPU_ASSIGNMENTS CACHED_NCPUMOE CACHED_BATCH CACHED_UBATCH CACHED_PARALLEL CACHED_KVUNIFIED CACHED_MMAP CACHED_NO_PINNED
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        key="${line%%=*}"; value="${line#*=}"
        [[ "$value" == \"*\" ]] || return 1; value="${value:1:${#value}-2}"
        case "$key" in
            CACHED_GPU_ASSIGNMENTS) [[ "$value" =~ ^[0-9:\ ]*$ ]] || return 1; CACHED_GPU_ASSIGNMENTS="$value" ;;
            CACHED_NCPUMOE) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_NCPUMOE="$value" ;;
            CACHED_BATCH) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_BATCH="$value" ;;
            CACHED_UBATCH) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_UBATCH="$value" ;;
            CACHED_PARALLEL) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_PARALLEL="$value" ;;
            CACHED_KVUNIFIED) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_KVUNIFIED="$value" ;;
            CACHED_MMAP) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_MMAP="$value" ;;
            CACHED_NO_PINNED) [[ "$value" =~ ^[0-9]+$ ]] || return 1; CACHED_NO_PINNED="$value" ;;
            *) return 1 ;;
        esac
    done < "$file"
}

_compute_gpu_kv_reserves() {
    local mode="$1" total_free=0 i gi share pct
    GPU_KV_RESERVED_TOTAL_MB=0; GPU_KV_MAX_PCT=0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        gi=${GPU_ORDER[$i]}; GPU_KV_RESERVE_MB[$gi]=0; total_free=$(( total_free + GPU_VRAM_FREE[$gi] ))
    done
    [[ "$mode" == "cpu" || "$mode" == "off" ]] && return 0
    (( KV_TOTAL_MB > 0 && total_free > 0 && GPU_COUNT > 0 )) || return 0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        gi=${GPU_ORDER[$i]}
        share=$(( (KV_TOTAL_MB * GPU_VRAM_FREE[$gi] + total_free - 1) / total_free ))
        if (( KV_PER_LAYER_MB > 0 )); then share=$(( ((share + KV_PER_LAYER_MB - 1) / KV_PER_LAYER_MB) * KV_PER_LAYER_MB ))
        fi
        GPU_KV_RESERVE_MB[$gi]=$share
        GPU_KV_RESERVED_TOTAL_MB=$(( GPU_KV_RESERVED_TOTAL_MB + share ))
        pct=0; (( GPU_VRAM_TOTAL[$gi] > 0 )) && pct=$(( share * 100 / GPU_VRAM_TOTAL[$gi] ))
        (( pct > GPU_KV_MAX_PCT )) && GPU_KV_MAX_PCT=$pct
    done
}

probe_cache_key() {
    printf '%s\n' "model=$MODEL_NAME" "layers=${LAYER_COUNT:-0}" "experts=${EXPERT_COUNT:-0}" \
        "embd=${EMBEDDING_LENGTH:-0}" "ff=${FEED_FORWARD_LENGTH:-0}" "ctx=$CTX_SIZE" \
        "ubatch=${UBATCH:-512}" "kv=$KV_QUALITY" | md5sum | awk '{print $1}'
}

probe_cache_file() { echo "$PROBE_CACHE_DIR/$(probe_cache_key).probe"; }

get_gpu_driver_version() {
    if [[ "$GPU_BACKEND" == "cuda" ]] && command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
    elif [[ "$GPU_BACKEND" == "Vulkan" ]] && command -v amd-smi &>/dev/null; then
        local ver; ver=$(amd-smi version 2>/dev/null | grep -oE 'amdgpu version:[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        [[ -n "$ver" ]] && echo "$ver" || { ver=$(amd-smi version 2>/dev/null | grep -oE 'AMDSMI Tool: *[0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) && echo "amdsmi-$ver" || echo ""; }
    else echo ""; fi
}

system_probe_signature() {
    local sig=""; for i in $(seq 0 $(( GPU_COUNT - 1 ))); do sig+="${GPU_NAME[$i]}|"; done
    local drv; drv=$(get_gpu_driver_version); sig+="drv=${drv:-unknown}"
    printf '%s' "$sig" | md5sum | awk '{print $1}'
}

system_probe_file() { echo "$PROBE_CACHE_DIR/system_$(system_probe_signature).cache"; }

load_system_probe() {
    local file; file=$(system_probe_file); SYS_CUDA_OVERHEAD_MB=""
    [[ -f "$file" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        key="${line%%=*}"; value="${line#*=}"
        case "$key" in SYS_CUDA_OVERHEAD_MB) [[ "$value" =~ ^[0-9]+$ ]] || return 1; SYS_CUDA_OVERHEAD_MB="$value" ;; *) ;; esac
    done < "$file"
    [[ -n "${SYS_CUDA_OVERHEAD_MB}" ]]
}

load_probe_cache() {
    local file; file=$(probe_cache_file); PROBED_COMPUTE_BUF_MB=""; PROBED_KV_PER_LAYER_MB=""
    [[ -f "$file" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        key="${line%%=*}"; value="${line#*=}"
        case "$key" in
            PROBED_COMPUTE_BUF_MB) [[ "$value" =~ ^[0-9]+$ ]] || return 1; PROBED_COMPUTE_BUF_MB="$value" ;;
            PROBED_KV_PER_LAYER_MB) [[ "$value" =~ ^[0-9]+$ ]] || return 1; PROBED_KV_PER_LAYER_MB="$value" ;;
            *) return 1 ;;
        esac
    done < "$file"
    [[ -n "${PROBED_COMPUTE_BUF_MB}" && -n "${PROBED_KV_PER_LAYER_MB}" ]] || return 1
    local _total_sys_mem=0
    if [[ -n "${TOTAL_VRAM_MB:-}" && -n "${RAM_AVAIL_MB:-}" ]]; then _total_sys_mem=$(( TOTAL_VRAM_MB + RAM_AVAIL_MB )); fi
    if (( _total_sys_mem > 0 && LAYER_COUNT > 0 )); then
        local _kv_total=$(( PROBED_KV_PER_LAYER_MB * LAYER_COUNT ))
        if (( _kv_total > _total_sys_mem )); then
            echo "  Note: probe reports KV ${PROBED_KV_PER_LAYER_MB}MB/layer × ${LAYER_COUNT} layers = ${_kv_total}MB > total system memory ${_total_sys_mem}MB — rejecting as corrupted"; PROBED_COMPUTE_BUF_MB=""; PROBED_KV_PER_LAYER_MB=""; return 1
        fi
    fi
    return 0
}

write_probe_cache() {
    local log_path="$1" offset="${2:-0}"
    [[ -f "$log_path" && -r "$log_path" ]] || return 0
    local recent compute_buf kv_total gpu_layers
    recent=$(tail -c +"$((offset+1))" "$log_path" 2>/dev/null | iconv -f utf-8 -t utf-8 -c 2>/dev/null || true) || return 0
    [[ -n "$recent" ]] || return 0
    compute_buf=$(printf '%s\n' "$recent" | grep -oE "CUDA[0-9]+ compute buffer size = *[0-9]+\.[0-9]+ MiB" | grep -oE "[0-9]+\.[0-9]+" | sort -gr | head -1 || true)
    kv_total=$(printf '%s\n' "$recent" | grep -oE "CUDA[0-9]+ KV buffer size = *[0-9]+\.[0-9]+ MiB" | grep -oE "[0-9]+\.[0-9]+" | awk '{sum+=$1} END {if (NR>0) printf "%.0f", sum}' || true)
    [[ -n "$compute_buf" && -n "$kv_total" ]] || return 0
    gpu_layers=$(( LAYER_COUNT - LAYERS_CPU )); (( gpu_layers > 0 )) || return 0
    local compute_buf_mb kv_per_layer_mb
    compute_buf_mb=$(printf "%.0f" "$compute_buf"); kv_per_layer_mb=$(awk -v t="$kv_total" -v n="$gpu_layers" 'BEGIN{printf "%.0f", t/n}')
    mkdir -p "$PROBE_CACHE_DIR" 2>/dev/null || return 0
    local out; out=$(probe_cache_file)
    cat > "$out" <<EOF
# Probe cache for $MODEL_NAME (ctx=$CTX_SIZE ubatch=${UBATCH:-512} kv=$KV_QUALITY)
# Generated: $(date)
PROBED_COMPUTE_BUF_MB=$compute_buf_mb
PROBED_KV_PER_LAYER_MB=$kv_per_layer_mb
EOF
    echo "  Probe written: compute_buf=${compute_buf_mb}MB, kv/layer=${kv_per_layer_mb}MB → $out"
}

_recompute_gpu_layer_caps() {
    local i gi overhead usable cap
    MAX_GPU_LAYERS=0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        gi=${GPU_ORDER[$i]}; overhead=$(( DET_FIXED_PER_GPU + ${GPU_KV_RESERVE_MB[$gi]:-0} ))
        (( i == 0 )) && overhead=$(( overhead + MAIN_GPU_GLOBALS_MB ))
        usable=$(( GPU_VRAM_FREE[$gi] - overhead )); (( usable < 0 )) && usable=0
        cap=0; (( COST_PER_LAYER_MB > 0 )) && cap=$(( usable / COST_PER_LAYER_MB ))
        (( cap > LAYER_COUNT )) && cap=$LAYER_COUNT
        MAX_GPU_LAYERS_PER[$gi]=$cap; MAX_GPU_LAYERS=$(( MAX_GPU_LAYERS + cap ))
    done
    (( MAX_GPU_LAYERS > LAYER_COUNT )) && MAX_GPU_LAYERS=$LAYER_COUNT
}

_recompute_cpu_layer_caps() {
    local cpu_kv_ram_mb=0
    [[ "$KV_PLACEMENT_EFFECTIVE" == "cpu" ]] && cpu_kv_ram_mb=$KV_TOTAL_MB
    _CPU_BUDGET_STRICT=$(( RAM_AVAIL_MB - RAM_OVERHEAD_PRE_MB - cpu_kv_ram_mb ))
    (( _CPU_BUDGET_STRICT < 0 )) && _CPU_BUDGET_STRICT=0
    MAX_CPU_LAYERS_STRICT=0; (( EXPERT_PER_LAYER_MB > 0 )) && MAX_CPU_LAYERS_STRICT=$(( _CPU_BUDGET_STRICT / EXPERT_PER_LAYER_MB ))
    (( MAX_CPU_LAYERS_STRICT > LAYER_COUNT )) && MAX_CPU_LAYERS_STRICT=$LAYER_COUNT
    _PRE_WORKING_SET_FLOOR=$(( RAM_OVERHEAD_PRE_MB + cpu_kv_ram_mb + 8 * EXPERT_PER_LAYER_MB ))
    MAX_CPU_LAYERS_MMAP=0
    (( RAM_AVAIL_MB >= _PRE_WORKING_SET_FLOOR )) && MAX_CPU_LAYERS_MMAP=$LAYER_COUNT
    MAX_CPU_LAYERS=$MAX_CPU_LAYERS_STRICT
    CEIL_CPU_LABEL="strict --no-mmap"
    if (( MAX_GPU_LAYERS + MAX_CPU_LAYERS_STRICT < LAYER_COUNT )) && (( USER_NO_MMAP == 0 )) && (( MAX_CPU_LAYERS_MMAP > MAX_CPU_LAYERS_STRICT )); then
        MAX_CPU_LAYERS=$MAX_CPU_LAYERS_MMAP; CEIL_CPU_LABEL="mmap (page-cache)"
    fi
}

# ── MoE placement (called from main when STRATEGY == moe_offload) ──
_run_moe_placement() {

# Short-circuit if user specified --n-cpu-moe
USER_NCPUMOE=""
_prev=""
for _f in "${EXTRA_LLAMA_FLAGS[@]:-}"; do
    if [[ "$_prev" == "--n-cpu-moe" || "$_prev" == "-ncmoe" ]]; then USER_NCPUMOE="$_f"; break; fi
    _prev="$_f"
done
if [[ -n "$USER_NCPUMOE" ]]; then
    echo ""; echo "User-supplied --n-cpu-moe ${USER_NCPUMOE} detected — skipping MoE placement."
    if (( DRY_RUN )); then if (( BENCHMARK )); then print_cmd "${BASE_FLAGS[@]}"; else print_cmd "${COMMON_FLAGS[@]}"; fi; exit 0; fi
    if (( BENCHMARK )); then run_two_phase_benchmark; BENCHMARK=0; fi
    if (( DO_AI_TUNE )); then TUNE_EXTRA_FIXED=(); ai_tune; fi
    run_with_restart "${COMMON_FLAGS[@]}"
fi

mkdir -p "$CACHE_DIR"
CACHE_KEY_INPUT="$(placement_cache_key_input)"
CACHE_KEY=$(printf '%s\n' "$CACHE_KEY_INPUT" | md5sum | awk '{print $1}')
CACHE_FILE="$CACHE_DIR/${CACHE_KEY}.conf"


# Per-layer expert size
if (( EXPERT_BYTES > 0 )); then
    EXPERT_TOTAL_MB=$(( EXPERT_BYTES / 1048576 ))
    NON_EXPERT_TOTAL_MB=$(( NON_EXPERT_BYTES / 1048576 ))
    EXPERT_PER_LAYER_MB=$(( EXPERT_TOTAL_MB / LAYER_COUNT ))
    EXPERT_PCT=$(( EXPERT_TOTAL_MB * 100 / TOTAL_SIZE_MB ))
    echo "Expert size per layer: ${EXPERT_PER_LAYER_MB}MB (${EXPERT_PCT}% expert, from GGUF tensors)"
else
    EXPERT_TOTAL_MB=$(awk "BEGIN {printf \"%.0f\", $TOTAL_SIZE_MB * 0.90}")
    EXPERT_PER_LAYER_MB=$(awk "BEGIN {printf \"%.0f\", $EXPERT_TOTAL_MB / $LAYER_COUNT}")
    echo "Expert size per layer: ~${EXPERT_PER_LAYER_MB}MB (estimated 90%, no tensor data)"
fi

KV_TOTAL_MB=${KV_TOTAL_MB:-0}
KV_PER_LAYER_MB=$(( KV_TOTAL_MB / LAYER_COUNT ))
(( KV_PER_LAYER_MB < 1 && KV_TOTAL_MB > 0 )) && KV_PER_LAYER_MB=1
declare -a GPU_KV_RESERVE_MB
GPU_KV_RESERVED_TOTAL_MB=0
GPU_KV_MAX_PCT=0


_compute_gpu_kv_reserves "$KV_PLACEMENT_EFFECTIVE"

# Check cached config
if [[ -f "$CACHE_FILE" && "$RETUNE" == "0" ]]; then
    echo ""; echo "Using cached config: $CACHE_FILE"
    if grep -qvE '^(#.*|CACHED_GPU_ASSIGNMENTS="[0-9: ]*"|CACHED_(NCPUMOE|BATCH|UBATCH|PARALLEL|KVUNIFIED|MMAP|NO_PINNED)="[0-9]+"|[[:space:]]*)$' "$CACHE_FILE"; then
        echo "WARNING: Cache file contains unexpected content, ignoring."; rm -f "$CACHE_FILE"
    elif ! load_placement_cache_file "$CACHE_FILE"; then
        echo "WARNING: Cache file failed validation, ignoring."; rm -f "$CACHE_FILE"
    fi
fi

# Recovered --n-cpu-moe cache
if [[ -f "$CACHE_FILE" && "$RETUNE" == "0" && -n "${CACHED_NCPUMOE:-}" ]]; then
    echo "  Recovered config: --n-cpu-moe ${CACHED_NCPUMOE} -b ${CACHED_BATCH:-1024} -ub ${CACHED_UBATCH:-512} --parallel ${CACHED_PARALLEL:-2}"
    FILTERED=(); skip_next=0
    for f in "${COMMON_FLAGS[@]}"; do
        if (( skip_next )); then skip_next=0; continue; fi
        case "$f" in -b|-ub|--parallel) skip_next=1; continue ;; esac
        FILTERED+=("$f")
    done
    COMMON_FLAGS=("${FILTERED[@]}" -b "${CACHED_BATCH:-1024}" -ub "${CACHED_UBATCH:-512}" --parallel "${CACHED_PARALLEL:-2}" --n-cpu-moe "$CACHED_NCPUMOE")
    [[ "${CACHED_KVUNIFIED:-0}" == "1" ]] && COMMON_FLAGS+=(--kv-unified)
    if [[ "${CACHED_NO_PINNED:-0}" == "1" ]]; then
        if [[ "$GPU_BACKEND" != "Vulkan" ]] && [[ ! " ${LAUNCH_ENV_PREFIX[*]:-} " =~ " GGML_CUDA_NO_PINNED=1 " ]]; then
            LAUNCH_ENV_PREFIX+=(GGML_CUDA_NO_PINNED=1)
            echo "  Cached: GGML_CUDA_NO_PINNED=1 (recovered previously due to pinned-host alloc)"
        fi
    fi
    if (( DRY_RUN )); then if (( BENCHMARK )); then print_cmd "${BASE_FLAGS[@]}"; else print_cmd "${COMMON_FLAGS[@]}"; fi; exit 0; fi
    run_with_restart "${COMMON_FLAGS[@]}"
fi

# Cached GPU assignments
if [[ -f "$CACHE_FILE" && "$RETUNE" == "0" && -n "${CACHED_GPU_ASSIGNMENTS:-}" ]]; then
    CACHE_VALID=1; OT_ARGS=()
    for assignment in $CACHED_GPU_ASSIGNMENTS; do
        IFS=: read -r ci cs cc <<< "$assignment"
        OT_ARGS+=("$ci" "$cs" "$cc")
        layer_mb=$(( cc * EXPERT_PER_LAYER_MB ))
        for gi in $(seq 0 $(( GPU_COUNT - 1 ))); do
            if [[ "${GPU_INDEX[$gi]}" == "$ci" ]]; then
                kv_share_mb=$(( ${GPU_KV_RESERVE_MB[$gi]:-0} + COMPUTE_PER_GPU_MB ))
                needed=$(( layer_mb + kv_share_mb ))
                if (( needed > GPU_VRAM_FREE[$gi] )); then
                    echo "  WARNING: GPU${ci} needs ${needed}MB but only ${GPU_VRAM_FREE[$gi]}MB free"
                    CACHE_VALID=0
                fi; break
            fi
        done
        echo "  GPU${ci}: ${cc} layers (starting at block ${cs})"
    done
    if (( CACHE_VALID )); then
        CACHED_MMAP_FLAGS=(--no-mmap)
        [[ "${CACHED_MMAP:-0}" == "1" ]] && { CACHED_MMAP_FLAGS=(); echo "  Cache says: mmap path (Phase 2 opt-in remembered)"; }
        OT_STRING=$(build_ot_string "${OT_ARGS[@]}")
        LAUNCH_FLAGS=("${COMMON_FLAGS[@]}" "${CACHED_MMAP_FLAGS[@]}" -ot "$OT_STRING")
        if (( DRY_RUN )); then if (( BENCHMARK )); then print_cmd "${BASE_FLAGS[@]}"; else print_cmd "${LAUNCH_FLAGS[@]}"; fi; exit 0; fi
        if (( BENCHMARK )); then run_two_phase_benchmark; BENCHMARK=0; fi
        if (( DO_AI_TUNE )); then
            COMMON_FLAGS+=("${CACHED_MMAP_FLAGS[@]}" -ot "$OT_STRING")
            TUNE_EXTRA_FIXED=("${CACHED_MMAP_FLAGS[@]}" -ot "$OT_STRING")
            ai_tune; kill_server "$RUNNING_PID"
        fi
        run_with_restart "${LAUNCH_FLAGS[@]}"
    else
        echo "  Cached config invalid for current VRAM — retuning..."; rm -f "$CACHE_FILE"
    fi
fi

# Probe cache helpers






# ── Phase-1 placement ────────────────────────────────────────
NON_EXPERT_RAM_MB=${NON_EXPERT_TOTAL_MB:-$(( TOTAL_SIZE_MB - EXPERT_TOTAL_MB ))}
NON_EXPERT_PER_LAYER_MB=$(( NON_EXPERT_RAM_MB / LAYER_COUNT ))
MAIN_GPU_GLOBALS_MB=$(( NON_EXPERT_RAM_MB - NON_EXPERT_PER_LAYER_MB * LAYER_COUNT / 2 ))
(( MAIN_GPU_GLOBALS_MB < 0 )) && MAIN_GPU_GLOBALS_MB=0

SYS_CUDA_OVERHEAD_MB=""
load_system_probe || true
if [[ -z "$SYS_CUDA_OVERHEAD_MB" ]]; then SYS_CUDA_OVERHEAD_MB=0; SYS_PROBE_FIRST_RUN=1
else SYS_PROBE_FIRST_RUN=0; fi

PROBE_HIT=0; PLACEMENT_LABEL="cold-start"
load_probe_cache && PROBE_HIT=1

if (( PROBE_HIT )); then
    PLACEMENT_LABEL="probe-hit"
    DET_FIXED_PER_GPU=$(( SYS_CUDA_OVERHEAD_MB + PROBED_COMPUTE_BUF_MB ))
    COST_PER_LAYER_MB=$(( EXPERT_PER_LAYER_MB + NON_EXPERT_PER_LAYER_MB ))
    echo "  Per-GPU reserve: ${SYS_CUDA_OVERHEAD_MB}MB cuda_overhead (measured) + ${PROBED_COMPUTE_BUF_MB}MB compute (probed) = ${DET_FIXED_PER_GPU}MB"
    echo "  Per-layer cost: ${EXPERT_PER_LAYER_MB}MB experts + ${NON_EXPERT_PER_LAYER_MB}MB attn/norm = ${COST_PER_LAYER_MB}MB"
else
    DET_FIXED_PER_GPU=$(( SYS_CUDA_OVERHEAD_MB + COMPUTE_FLOOR_MB ))
    COST_PER_LAYER_MB=$(( EXPERT_PER_LAYER_MB + NON_EXPERT_PER_LAYER_MB ))
    (( SYS_PROBE_FIRST_RUN )) && echo "  Note: first launch on this system — using cited 1024MB compute floor for safety. Subsequent launches will use measured cuda_overhead."
    echo "  Per-GPU reserve: ${SYS_CUDA_OVERHEAD_MB}MB cuda_overhead (measured) + ${COMPUTE_FLOOR_MB}MB compute floor (cited llama.cpp) = ${DET_FIXED_PER_GPU}MB"
    echo "  Per-layer cost: ${EXPERT_PER_LAYER_MB}MB experts + ${NON_EXPERT_PER_LAYER_MB}MB attn/norm = ${COST_PER_LAYER_MB}MB"
fi

[[ "$KV_PLACEMENT_EFFECTIVE" == "gpu" && "$KV_PLACEMENT" != "cpu" ]] && (( GPU_KV_RESERVED_TOTAL_MB > 0 )) && \
    echo "  GPU KV reserve first: ${GPU_KV_RESERVED_TOTAL_MB}MB total (${KV_TYPE}, max ${GPU_KV_MAX_PCT}% of a GPU)"

# Hard ceilings

_RE_CUDA_HOST_MB=1024; _PRE_GRAPH_SCRATCH_MB=2048; _PRE_MMAP_PT_MB=$(( TOTAL_SIZE_MB / 500 ))
if (( EMBEDDING_LENGTH > 0 )); then
    if (( EXPERT_COUNT > 0 && EXPERT_USED_COUNT > 0 && EXPERT_FF > 0 )); then _PRE_ACT_FFN=$(( EXPERT_USED_COUNT * EXPERT_FF )); (( EXPERT_SHARED_FF > 0 )) && _PRE_ACT_FFN=$(( _PRE_ACT_FFN + EXPERT_SHARED_FF ))
    else _PRE_ACT_FFN=$FEED_FORWARD_LENGTH; fi
    (( KV_LORA_RANK > 0 )) && _PRE_ACT_FFN=$(( _PRE_ACT_FFN + KV_LORA_RANK + Q_LORA_RANK ))
    _PRE_CPU_ACT_MB=$(( UBATCH * (EMBEDDING_LENGTH + _PRE_ACT_FFN) * 4 * 2 / 1048576 ))
    (( _PRE_CPU_ACT_MB < 64 )) && _PRE_CPU_ACT_MB=64
else _PRE_CPU_ACT_MB=512; fi
RAM_OVERHEAD_PRE_MB=$(( _PRE_CUDA_HOST_MB + _PRE_GRAPH_SCRATCH_MB + _PRE_MMAP_PT_MB + _PRE_CPU_ACT_MB ))


_recompute_gpu_layer_caps; _recompute_cpu_layer_caps

MAX_GPU_LAYERS_NO_GPU_KV=0
for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
    gi=${GPU_ORDER[$i]}; _no_kv_overhead=$DET_FIXED_PER_GPU; (( i == 0 )) && _no_kv_overhead=$(( _no_kv_overhead + MAIN_GPU_GLOBALS_MB ))
    _no_kv_usable=$(( GPU_VRAM_FREE[$gi] - _no_kv_overhead )); (( _no_kv_usable < 0 )) && _no_kv_usable=0
    _no_kv_cap=0; (( COST_PER_LAYER_MB > 0 )) && _no_kv_cap=$(( _no_kv_usable / COST_PER_LAYER_MB ))
    (( _no_kv_cap > LAYER_COUNT )) && _no_kv_cap=$LAYER_COUNT
    MAX_GPU_LAYERS_NO_GPU_KV=$(( MAX_GPU_LAYERS_NO_GPU_KV + _no_kv_cap ))
done
(( MAX_GPU_LAYERS_NO_GPU_KV > LAYER_COUNT )) && MAX_GPU_LAYERS_NO_GPU_KV=$LAYER_COUNT
GPU_KV_LAYER_LOSS=$(( MAX_GPU_LAYERS_NO_GPU_KV - MAX_GPU_LAYERS )); (( GPU_KV_LAYER_LOSS < 0 )) && GPU_KV_LAYER_LOSS=0

if [[ "$KV_PLACEMENT" == "auto" && "$KV_PLACEMENT_EFFECTIVE" == "gpu" ]] && (( GPU_KV_RESERVED_TOTAL_MB > 0 )); then
    _kv_prompt_reason=""
    if (( MAX_GPU_LAYERS + MAX_CPU_LAYERS < LAYER_COUNT )); then _kv_prompt_reason="GPU KV does not fit with available CPU/GPU layer capacity"
    elif (( GPU_KV_MAX_PCT >= 80 )); then _kv_prompt_reason="GPU KV would consume ${GPU_KV_MAX_PCT}% of one GPU"
    elif (( GPU_KV_LAYER_LOSS >= 8 )); then _kv_prompt_reason="GPU KV reduces expert layers on GPU by ${GPU_KV_LAYER_LOSS}"
    fi
    if [[ -n "$_kv_prompt_reason" ]]; then
        echo ""; echo "KV placement decision:"; echo "  $_kv_prompt_reason."
        echo "  [g] GPU KV: faster long-context attention, fewer expert layers on GPU."
        echo "  [c] CPU KV: same ${KV_TYPE} KV quality, frees VRAM for experts, slower long-context attention."
        if [[ -t 0 && "${LLM_ASSUME_YES:-0}" != "1" && "$DRY_RUN" == "0" ]]; then
            read -r -p "  Choose KV placement [G/c]: " _kv_answer
            case "${_kv_answer,,}" in c|cpu) KV_PLACEMENT_EFFECTIVE="cpu"; _apply_kv_placement_flag cpu; _compute_gpu_kv_reserves cpu; _recompute_gpu_layer_caps; _recompute_cpu_layer_caps; echo "  Selected CPU KV (--no-kv-offload)." ;; *) echo "  Selected GPU KV." ;; esac
        else echo "  Non-interactive mode: keeping default GPU KV. Use --kv-placement cpu to move KV to RAM."; fi
    fi
fi

echo ""; echo "Hard ceilings (measured, no margins):"
for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
    gi=${GPU_ORDER[$i]}; _kv_note=""; (( ${GPU_KV_RESERVE_MB[$gi]:-0} > 0 )) && _kv_note=", KV reserve ${GPU_KV_RESERVE_MB[$gi]}MB"
    echo "  GPU${GPU_INDEX[$gi]} (${GPU_NAME[$gi]}): up to ${MAX_GPU_LAYERS_PER[$gi]} layers (${GPU_VRAM_FREE[$gi]}MB free${_kv_note})"
done
echo "  GPU total:      ${MAX_GPU_LAYERS} layers"
echo "  CPU [${CEIL_CPU_LABEL}]: ${MAX_CPU_LAYERS} layers (avail ${RAM_AVAIL_MB}MB − overhead ${RAM_OVERHEAD_PRE_MB}MB, expert ${EXPERT_PER_LAYER_MB}MB/layer)"
echo "  Total capacity: $(( MAX_GPU_LAYERS + MAX_CPU_LAYERS ))/${LAYER_COUNT} layers required"

if (( MAX_GPU_LAYERS + MAX_CPU_LAYERS < LAYER_COUNT )); then
    GAP=$(( LAYER_COUNT - MAX_GPU_LAYERS - MAX_CPU_LAYERS ))
    GAP_VRAM_MB=$(( GAP * COST_PER_LAYER_MB )); GAP_RAM_MB=$(( GAP * EXPERT_PER_LAYER_MB ))
    echo ""; echo "ERROR: Model does not fit on this system."
    echo "  Required:    ${LAYER_COUNT} layers"; echo "  GPU cap:     ${MAX_GPU_LAYERS} layers across ${GPU_COUNT} GPU(s)"
    echo "  CPU cap:     ${MAX_CPU_LAYERS} layers (${CEIL_CPU_LABEL})"
    echo "  Gap:         ${GAP} layers — need ~${GAP_VRAM_MB}MB more free VRAM or ~${GAP_RAM_MB}MB more RAM"
    echo ""; echo "  Options:"
    echo "    1. Free VRAM (close other GPU workloads, --gpus to add a card)"
    if (( USER_NO_MMAP )); then echo "    2. Drop --no-mmap so kernel can page experts on demand"
    else echo "    2. Add system RAM"; fi
    echo "    3. Use a smaller quantization or smaller model"; exit 1
fi

REDIST_BUMPS=()
declare -a LAYERS_PER_GPU
NEXT_LAYER=0; TOTAL_GPU_LAYERS=0
LAYERS_CPU=$(( LAYER_COUNT - TOTAL_GPU_LAYERS ))

for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
    gi=${GPU_ORDER[$i]}
    local_overhead=$(( DET_FIXED_PER_GPU + ${GPU_KV_RESERVE_MB[$gi]:-0} ))
    (( i == 0 )) && local_overhead=$(( local_overhead + MAIN_GPU_GLOBALS_MB + ${GPU_KV_RESERVE_MB[$gi]:-0} ))
    usable_mb=$(( GPU_VRAM_FREE[$gi] - local_overhead )); (( usable_mb < 0 )) && usable_mb=0
    layers=0; (( COST_PER_LAYER_MB > 0 )) && layers=$(( usable_mb / COST_PER_LAYER_MB ))
    remain=$(( LAYER_COUNT - NEXT_LAYER )); (( layers > remain )) && layers=$remain
    LAYERS_PER_GPU[$gi]=$layers; NEXT_LAYER=$(( NEXT_LAYER + layers )); TOTAL_GPU_LAYERS=$(( TOTAL_GPU_LAYERS + layers ))
done
LAYERS_CPU=$(( LAYER_COUNT - TOTAL_GPU_LAYERS ))

CPU_EXPERT_MB=$(( LAYERS_CPU * EXPERT_PER_LAYER_MB ))
CUDA_HOST_MB=1024; GRAPH_SCRATCH_MB=2048; MMAP_PT_MB=$(( TOTAL_SIZE_MB / 500 ))
if (( EMBEDDING_LENGTH > 0 )); then
    if (( EXPERT_COUNT > 0 && EXPERT_USED_COUNT > 0 && EXPERT_FF > 0 )); then _act_ffn=$(( EXPERT_USED_COUNT * EXPERT_FF )); (( EXPERT_SHARED_FF > 0 )) && _act_ffn=$(( _act_ffn + EXPERT_SHARED_FF ))
    else _act_ffn=$FEED_FORWARD_LENGTH; fi
    (( KV_LORA_RANK > 0 )) && _act_ffn=$(( _act_ffn + KV_LORA_RANK + Q_LORA_RANK ))
    CPU_ACT_MB=$(( UBATCH * (EMBEDDING_LENGTH + _act_ffn) * 4 * 2 / 1048576 )); (( CPU_ACT_MB < 64 )) && CPU_ACT_MB=64
else CPU_ACT_MB=512; fi
RAM_OVERHEAD_MB=$(( CUDA_HOST_MB + GRAPH_SCRATCH_MB + MMAP_PT_MB + CPU_ACT_MB ))
CPU_KV_MB=0; [[ "$KV_PLACEMENT_EFFECTIVE" == "cpu" ]] && CPU_KV_MB=$KV_TOTAL_MB
RAM_NEEDED=$(( CPU_EXPERT_MB + CPU_KV_MB + RAM_OVERHEAD_MB ))

# Phase 1 redistribution
if (( RAM_NEEDED > RAM_AVAIL_MB && LAYERS_CPU > 0 && GPU_COUNT > 0 )); then
    echo "WARNING: ${PLACEMENT_LABEL} placement needs ${RAM_NEEDED}MB RAM but only ${RAM_AVAIL_MB}MB available"
    echo "  Redistributing layers to GPUs to fit in RAM..."
    for bump in "${REDIST_BUMPS[@]}"; do
        NEXT_LAYER=0; TOTAL_GPU_LAYERS=0
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; pct=$(( bump - i * 5 )); (( pct < 30 )) && pct=30
            local_overhead=$COMPUTE_PER_GPU_MB; (( i == 0 )) && local_overhead=$(( COMPUTE_PER_GPU_MB + MAIN_GPU_GLOBALS_MB ))
            available_mb=$(( GPU_VRAM_FREE[$gi] - local_overhead )); (( available_mb < 0 )) && available_mb=0
            budget_mb=$(( available_mb * pct / 100 )); layers=0
            if (( COST_PER_LAYER_MB > 0 )); then
                layers=$(( budget_mb / COST_PER_LAYER_MB ))
                if (( layers == 0 && bump == ${REDIST_BUMPS[-1]} )); then layers=$(( available_mb / COST_PER_LAYER_MB )); fi
            fi
            remain=$(( LAYER_COUNT - NEXT_LAYER )); (( layers > remain )) && layers=$remain
            LAYERS_PER_GPU[$gi]=$layers; NEXT_LAYER=$(( NEXT_LAYER + layers )); TOTAL_GPU_LAYERS=$(( TOTAL_GPU_LAYERS + layers ))
        done
        LAYERS_CPU=$(( LAYER_COUNT - TOTAL_GPU_LAYERS ))
        CPU_EXPERT_MB=$(( LAYERS_CPU * EXPERT_PER_LAYER_MB ))
        RAM_NEEDED=$(( CPU_EXPERT_MB + CPU_KV_MB + RAM_OVERHEAD_MB ))
        if (( RAM_NEEDED <= RAM_AVAIL_MB )); then
            echo "  Adjusted GPU budget to ${bump}% — ${LAYERS_CPU} CPU layers (${RAM_NEEDED}MB RAM)"; break
        fi
    done
fi

MMAP_PATH=0; MOE_MMAP_FLAGS=(--no-mmap)
if (( TOTAL_SIZE_MB > RAM_AVAIL_MB )); then
    echo "  Note: model file (${TOTAL_SIZE_MB}MB) > RAM (${RAM_AVAIL_MB}MB) — forcing mmap path"
    if (( USER_NO_MMAP )); then
        echo "ERROR: --no-mmap requested, but model file exceeds available RAM."; exit 1
    fi
    MMAP_PATH=1; MOE_MMAP_FLAGS=()
fi

if (( RAM_NEEDED > RAM_AVAIL_MB )); then
    WORKING_SET_FLOOR=$(( RAM_OVERHEAD_MB + CPU_KV_MB + 8 * EXPERT_PER_LAYER_MB ))
    SHORTFALL=$(( RAM_NEEDED - RAM_AVAIL_MB ))
    echo ""; echo "NOTE: Full resident MoE size would exceed available RAM."
    echo "      (On the mmap path this is informational — the kernel pages cold experts in/out.)"
    echo ""; echo "  Breakdown:"
    echo "    CPU expert layers:     ${LAYERS_CPU} layers (${CPU_EXPERT_MB}MB)"
    (( CPU_KV_MB > 0 )) && echo "    CPU KV cache:          ${CPU_KV_MB}MB (${KV_TYPE})"
    echo "    ggml compute scratch:  ${GRAPH_SCRATCH_MB}MB"; echo "    CUDA host buffers:     ${CUDA_HOST_MB}MB"
    echo "    mmap page tables:      ${MMAP_PT_MB}MB"; echo "    CPU activation buffer: ${CPU_ACT_MB}MB"
    echo "    ───────────────────────────────────"; echo "    Full resident size:    ${RAM_NEEDED}MB"
    echo "    RAM available:         ${RAM_AVAIL_MB}MB"; echo "    Shortfall (resident):  ${SHORTFALL}MB"; echo ""
    if (( RAM_AVAIL_MB >= WORKING_SET_FLOOR )); then
        echo "  Working set on mmap path: ~${WORKING_SET_FLOOR}MB (compute + 8 hot layers)"
        echo "  — that fits in your ${RAM_AVAIL_MB}MB of available RAM."; echo ""
        if [[ -t 0 ]] && (( ${LLM_ASSUME_YES:-0} == 0 )); then
            read -r -t 20 -p "  Continue and let mmap page cold experts on demand? [y/N] " _reply || _reply="n"
            if [[ ! "$_reply" =~ ^[Yy] ]]; then echo "  Aborted by user."; exit 1; fi
        else echo "  Non-interactive (or LLM_ASSUME_YES=1) — continuing."; fi
        echo "  Proceeding with mmap paging enabled (no --no-mmap)."; MMAP_PATH=1; MOE_MMAP_FLAGS=()
    else
        echo "  Working set on mmap path would be ~${WORKING_SET_FLOOR}MB, still above ${RAM_AVAIL_MB}MB available."
        echo "  1. Add more system RAM  2. Use a smaller quantization  3. Use a smaller model"; exit 1
    fi
fi

# Build -ot string
OT_ARGS=(); NEXT_LAYER=0
echo ""; echo "Expert placement (${PLACEMENT_LABEL}):"
for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
    gi=${GPU_ORDER[$i]}; layers=${LAYERS_PER_GPU[$gi]}
    if (( layers > 0 )); then OT_ARGS+=("${GPU_INDEX[$gi]}" "$NEXT_LAYER" "$layers")
        echo "  GPU${GPU_INDEX[$gi]} (${GPU_NAME[$gi]}): ${layers} layers (blk ${NEXT_LAYER}-$((NEXT_LAYER + layers - 1)))"
        NEXT_LAYER=$(( NEXT_LAYER + layers ))
    fi
done
echo "  CPU (RAM): ${LAYERS_CPU} layers (~${CPU_EXPERT_MB}MB)"
OT_STRING=$(build_ot_string "${OT_ARGS[@]}")

if (( DRY_RUN )); then if (( BENCHMARK )); then print_cmd "${BASE_FLAGS[@]}"; else print_cmd "${COMMON_FLAGS[@]}" "${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"; fi; exit 0; fi
if (( BENCHMARK )); then run_two_phase_benchmark; BENCHMARK=0; fi

# ── Auto-tuning loop ────────────────────────────────────────
echo ""; echo "Starting server (conservative)..."
if try_start "${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"; then
    echo "  Server healthy (PID $RUNNING_PID)"
    (( ! PROBE_HIT )) && write_probe_cache "$SERVER_LOG" "${LAUNCH_LOG_OFFSET:-0}"

    OPT_ITER=0; OPT_MAX_ITER=4
    declare -a SAVED_LAYERS_PER_GPU
    SAVED_LAYERS_CPU=$LAYERS_CPU; SAVED_OT_STRING="$OT_STRING"

    while (( OPT_ITER < OPT_MAX_ITER )); do
        OPT_ITER=$(( OPT_ITER + 1 ))
        (( OPT_ITER > 1 )) && echo "" && echo "── Iteration $OPT_ITER: re-measuring with new placement ──"

        # Stress probe
        declare -a PEAK_USED; SAMPLE_FILE=$(mktemp)
        if (( GPU_COUNT > 0 )) && command -v nvidia-smi >/dev/null 2>&1; then
            echo "  Stress probe: measuring peak VRAM during generation..."
            ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> "$SAMPLE_FILE"; sleep 0.1; done ) &
            SAMPLER_PID=$!
        fi

        local prompt_stress="Describe in detail how this component fits into the system. "
        for _i in $(seq 1 30); do prompt_stress+="Describe in detail how this component fits into the system. "; done
        curl -sf --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt_stress\"}],\"max_tokens\":64,\"temperature\":0}" >/dev/null 2>&1
        kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true

        echo "  Peak VRAM during load:"
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; idx=${GPU_INDEX[$gi]}
            peak=$(awk -F, -v idx="$idx" '$1+0==idx && $2+0>max {max=$2+0} END {print max+0}' "$SAMPLE_FILE" 2>/dev/null)
            [[ "$peak" =~ ^[0-9]+$ ]] || peak=0
            PEAK_USED[$gi]=$peak
            echo "    GPU${idx} (${GPU_NAME[$gi]}): ${PEAK_USED[$gi]}MB peak / ${GPU_VRAM_TOTAL[$gi]}MB total"
        done
        rm -f "$SAMPLE_FILE"

        # Measure actual VRAM
        echo "  Measuring VRAM headroom..."
        declare -a ACTUAL_FREE ACTUAL_USED
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; raw_free=""
            if command -v nvidia-smi &>/dev/null; then
                raw_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX[$gi]}" 2>/dev/null | tr -d ' ' || true)
            fi
            if [[ ! "$raw_free" =~ ^[0-9]+$ ]]; then
                estimated_used=$(( LAYERS_PER_GPU[$gi] * COST_PER_LAYER_MB ))
                raw_free=$(( GPU_VRAM_FREE[$gi] - estimated_used )); (( raw_free < 0 )) && raw_free=0
            fi
            if [[ -n "${PEAK_USED[$gi]:-}" ]] && (( PEAK_USED[$gi] > 0 )); then
                ACTUAL_USED[$gi]=${PEAK_USED[$gi]}; ACTUAL_FREE[$gi]=$(( GPU_VRAM_TOTAL[$gi] - PEAK_USED[$gi] ))
            else
                ACTUAL_FREE[$gi]=$raw_free; ACTUAL_USED[$gi]=$raw_free
            fi
            (( ACTUAL_FREE[$gi] < 0 )) && ACTUAL_FREE[$gi]=0
        done

        # Derive cuda_overhead
        if [[ -z "${SYS_CUDA_OVERHEAD_MB:-}" ]] || [[ ! -f "$(system_probe_file)" ]]; then
            primary_gi=${GPU_ORDER[0]}; primary_used=${ACTUAL_USED[$primary_gi]:-0}
            if (( primary_used > 0 )); then
                launch_log_tail=$(tail -c +"$((${LAUNCH_LOG_OFFSET:-0}+1))" "$SERVER_LOG" 2>/dev/null | iconv -f utf-8 -t utf-8 -c 2>/dev/null || true)
                primary_idx=${GPU_INDEX[$primary_gi]}
                sum_buffers_mb() {
                    local tag="$1" pattern
                    if [[ "$tag" == "model" ]]; then pattern="CUDA${primary_idx}( model)? buffer size = *[0-9]+\.[0-9]+ MiB"
                    else pattern="CUDA${primary_idx} ${tag} buffer size = *[0-9]+\.[0-9]+ MiB"; fi
                    { printf '%s\n' "$launch_log_tail" | grep -oE "$pattern" | grep -oE "[0-9]+\.[0-9]+" | awk '{sum+=$1} END {if (NR>0) printf "%.0f", sum; else print 0}'; } || true
                }
                mb_model=$(sum_buffers_mb model); mb_kv=$(sum_buffers_mb KV); mb_compute=$(sum_buffers_mb compute)
                accounted=$(( mb_model + mb_kv + mb_compute ))
                measured_overhead=$(( primary_used - accounted ))
                if (( measured_overhead > 0 && measured_overhead < primary_used )); then
                    mkdir -p "$PROBE_CACHE_DIR" 2>/dev/null || true
                    local drv_version; drv_version=$(get_gpu_driver_version)
                    cat > "$(system_probe_file)" <<EOF
# System probe (post-launch measurement) for ${GPU_NAME[$primary_gi]}
# Driver: ${drv_version}; Backend: $GPU_BACKEND
# Generated: $(date); SYS_CUDA_OVERHEAD_MB=$measured_overhead
EOF
                    echo "  System probe written: cuda_overhead=${measured_overhead}MB (measured from launch)"
                fi
            fi
        fi

        # Calculate extra layers
        SAFETY_MB=$COMPUTE_PER_GPU_MB; TOTAL_EXTRA=0
        declare -a EXTRA_PER_GPU
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; extra=0
            current_kv=$(( LAYERS_PER_GPU[$gi] * KV_PER_LAYER_MB ))
            usable_free=$(( ACTUAL_FREE[$gi] - current_kv - SAFETY_MB )); (( usable_free < 0 )) && usable_free=0
            if (( COST_PER_LAYER_MB > 0 )); then extra=$(( usable_free / COST_PER_LAYER_MB ))
            fi
            (( extra < 0 )) && extra=0; EXTRA_PER_GPU[$gi]=$extra
            TOTAL_EXTRA=$(( TOTAL_EXTRA + extra ))
        done

        if (( TOTAL_EXTRA == 0 || LAYERS_CPU == 0 )); then
            if (( OPT_ITER == 1 )); then echo "  Conservative config is already optimal (GPUs full)"
            else echo "  Iteration $OPT_ITER: converged — measured headroom cannot fit another layer"; fi
            break
        fi

        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do gi=${GPU_ORDER[$i]}; SAVED_LAYERS_PER_GPU[$gi]=${LAYERS_PER_GPU[$gi]}; done
        SAVED_LAYERS_CPU=$LAYERS_CPU; SAVED_OT_STRING="$OT_STRING"

        remaining_cpu=$LAYERS_CPU; applied_total=0
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; add=$(( EXTRA_PER_GPU[$gi] < remaining_cpu ? EXTRA_PER_GPU[$gi] : remaining_cpu ))
            LAYERS_PER_GPU[$gi]=$(( LAYERS_PER_GPU[$gi] + add )); applied_total=$(( applied_total + add ))
            remaining_cpu=$(( remaining_cpu - add )); (( remaining_cpu <= 0 )) && break
        done
        LAYERS_CPU=$remaining_cpu; TOTAL_GPU_LAYERS=$(( LAYER_COUNT - LAYERS_CPU ))

        OT_ARGS=(); NEXT_LAYER=0
        echo ""; echo "Optimized placement (iter $OPT_ITER, +${applied_total} layers):"
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; layers=${LAYERS_PER_GPU[$gi]}
            if (( layers > 0 )); then OT_ARGS+=("${GPU_INDEX[$gi]}" "$NEXT_LAYER" "$layers")
                echo "  GPU${GPU_INDEX[$gi]} (${GPU_NAME[$gi]}): ${layers} layers (blk ${NEXT_LAYER}-$((NEXT_LAYER + layers - 1)))"
                NEXT_LAYER=$(( NEXT_LAYER + layers ))
            fi
        done
        echo "  CPU (RAM): ${LAYERS_CPU} layers"
        OT_STRING=$(build_ot_string "${OT_ARGS[@]}")

        kill_server "$RUNNING_PID"; echo ""; echo "Restarting with optimized config (iter $OPT_ITER)..."
        MLOCK_ARGS=()
        (( MMAP_PATH )) && log "  mmap path active — skipping --mlock"
        if (( ! MMAP_PATH && ! IS_WSL2 )); then MLOCK_ARGS=(--mlock); fi
        if ! try_start "${MOE_MMAP_FLAGS[@]}" "${MLOCK_ARGS[@]}" -ot "$OT_STRING"; then
            echo "  Iteration $OPT_ITER restart failed (with mlock) — retrying without mlock..."
            kill_server ""
            if ! try_start "${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"; then
                echo "  Iteration $OPT_ITER: new placement won't load — rolling back to last known good"
                kill_server ""
                for i in $(seq 0 $(( GPU_COUNT - 1 ))); do gi=${GPU_ORDER[$i]}; LAYERS_PER_GPU[$gi]=${SAVED_LAYERS_PER_GPU[$gi]}; done
                LAYERS_CPU=$SAVED_LAYERS_CPU; OT_STRING="$SAVED_OT_STRING"
                TOTAL_GPU_LAYERS=$(( LAYER_COUNT - LAYERS_CPU ))
                if ! try_start "${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"; then echo "  FAILED: Could not even restart last-known-good. Check $SERVER_LOG"; exit 1; fi
                echo "  Rolled back to: $(( LAYER_COUNT - LAYERS_CPU )) GPU layers, $LAYERS_CPU CPU layers"; break
            fi
        fi
    done

    # Save to cache
    CACHED_GPU_ASSIGNMENTS=""; NEXT_LAYER=0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        gi=${GPU_ORDER[$i]}; layers=${LAYERS_PER_GPU[$gi]}
        if (( layers > 0 )); then CACHED_GPU_ASSIGNMENTS+="${GPU_INDEX[$gi]}:${NEXT_LAYER}:${layers} "; NEXT_LAYER=$(( NEXT_LAYER + layers )); fi
    done
    cat > "$CACHE_FILE" <<CEOF
# Auto-tuned config for $MODEL_NAME
# Generated: $(date); Total layers: $LAYER_COUNT, Experts: $EXPERT_COUNT
CACHED_GPU_ASSIGNMENTS="$CACHED_GPU_ASSIGNMENTS"
CACHED_MMAP="${MMAP_PATH}"
CEOF

    TOTAL_GPU_LAYERS=$(( LAYER_COUNT - LAYERS_CPU ))
    echo ""; echo "═══════════════════════════════════════"
    echo "  Model: $MODEL_NAME"; echo "  Size: ${TOTAL_SIZE_GB}GB | Layers: $LAYER_COUNT | Experts: $EXPERT_COUNT"
    echo "  GPU layers: $TOTAL_GPU_LAYERS | CPU layers: $LAYERS_CPU"
    echo "  Context: $CTX_SIZE | Port: $PORT"; echo "  Config cached: $CACHE_FILE"
    echo "═══════════════════════════════════════"

    (( DO_AI_TUNE )) && { echo ""; echo "MoE placement complete. Running AI Tune with placement locked..."; COMMON_FLAGS+=("${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"); TUNE_EXTRA_FIXED=("${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"); ai_tune; }
    kill_server "$RUNNING_PID"
    if (( DO_AI_TUNE )); then run_with_restart "${COMMON_FLAGS[@]}"
    else run_with_restart "${COMMON_FLAGS[@]}" "${MOE_MMAP_FLAGS[@]}" -ot "$OT_STRING"; fi
else
    echo ""; echo "  Conservative -ot placement failed — trying tighter recovery config..."
    tail -5 "$SERVER_LOG" 2>/dev/null | sed 's/^/    /'; kill_server ""

    NCPUMOE_SUPPORTED=0; KVUNIFIED_SUPPORTED=0
    "$LLAMA_SERVER" --help 2>&1 | grep -qE "(^| )--n-cpu-moe[ =]" && NCPUMOE_SUPPORTED=1
    "$LLAMA_SERVER" --help 2>&1 | grep -qE "^ *-kvu.*--kv-unified\b" && KVUNIFIED_SUPPORTED=1

    RECOVERED=0; boundary_hit=0
    if (( NCPUMOE_SUPPORTED )); then
        RECOVERY_COMMON=(); skip_next=0
        for f in "${COMMON_FLAGS[@]}"; do
            if (( skip_next )); then skip_next=0; continue; fi
            case "$f" in -b|-ub) skip_next=1; continue ;; -ot|--override-tensor) skip_next=1; continue ;; --parallel) skip_next=1; continue ;; esac
            RECOVERY_COMMON+=("$f")
        done
        RECOVERY_COMMON+=(-b 1024 -ub 512 --parallel 2)
        (( KVUNIFIED_SUPPORTED )) && RECOVERY_COMMON+=(--kv-unified)

        SAVED_COMMON_FLAGS=("${COMMON_FLAGS[@]}"); COMMON_FLAGS=("${RECOVERY_COMMON[@]}")
        ot_failure_mode=$(_parse_load_failure)
        case "$ot_failure_mode" in pinned_hang|pinned_fail)
            if [[ "$GPU_BACKEND" != "Vulkan" ]] && [[ ! " ${LAUNCH_ENV_PREFIX[*]:-} " =~ " GGML_CUDA_NO_PINNED=1 " ]]; then
                echo "  -ot failure was pinned-host alloc — disabling pinned buffers (GGML_CUDA_NO_PINNED=1)"
                LAUNCH_ENV_PREFIX+=(GGML_CUDA_NO_PINNED=1)
            fi ;; esac

        START_N=$LAYERS_CPU; (( START_N < 1 )) && START_N=1; (( START_N > LAYER_COUNT - 1 )) && START_N=$(( LAYER_COUNT - 1 ))
        MIN_N=$(( LAYER_COUNT - MAX_GPU_LAYERS )); (( MIN_N < 0 )) && MIN_N=0
        MAX_N=$(( LAYER_COUNT - 1 )); (( MAX_CPU_LAYERS < MAX_N )) && MAX_N=$MAX_CPU_LAYERS
        N=$START_N; (( N < MIN_N )) && N=$MIN_N; (( N > MAX_N )) && N=$MAX_N
        attempts=0; max_attempts=$(( LAYER_COUNT + 2 ))
        no_pinned_tried=0
        [[ " ${LAUNCH_ENV_PREFIX[*]:-} " =~ " GGML_CUDA_NO_PINNED=1 " ]] && no_pinned_tried=1
        [[ "$GPU_BACKEND" == "Vulkan" ]] && no_pinned_tried=1
        last_direction=""; boundary_hit=0

        while (( attempts < max_attempts )); do
            attempts=$(( attempts + 1 ))
            echo "  Recovery attempt $attempts: --n-cpu-moe $N (range [$MIN_N..$MAX_N], b=1024 ub=512 parallel=2$( (( KVUNIFIED_SUPPORTED )) && echo ' kv-unified')$( (( no_pinned_tried )) && [[ "$GPU_BACKEND" != "Vulkan" ]] && echo ' no-pinned'))"
            if try_start --n-cpu-moe "$N"; then
                RECOVERED=1; echo "  Server healthy with --n-cpu-moe $N (attempts: $attempts)"
                COMMON_FLAGS+=(--n-cpu-moe "$N"); OT_STRING=""; LAYERS_CPU=$N
                cached_no_pinned=0; [[ " ${LAUNCH_ENV_PREFIX[*]:-} " =~ " GGML_CUDA_NO_PINNED=1 " ]] && cached_no_pinned=1
                cat > "$CACHE_FILE" <<CEOF
# Auto-tuned config for $MODEL_NAME (recovered via --n-cpu-moe)
# Generated: $(date); CACHED_NCPUMOE="$N" CACHED_BATCH="1024" CACHED_UBATCH="512" CACHED_PARALLEL="2" CACHED_KVUNIFIED="$KVUNIFIED_SUPPORTED" CACHED_NO_PINNED="$cached_no_pinned"
CEOF
                break
            fi
            mode=$(_parse_load_failure); direction=$(_failure_direction "$mode"); kill_server ""
            if [[ -n "$last_direction" && "$direction" != "indeterminate" ]]; then
                if [[ "$last_direction" == "up" && "$direction" == "more_gpu" ]] || [[ "$last_direction" == "down" && "$direction" == "more_cpu" ]]; then
                    echo "  Direction flip: prior attempt failed wanting '${last_direction}', now wants '${direction}' — boundary boxed in."; boundary_hit=1; break
                fi
            fi
            case "$mode" in
                oom\ *) read -r _tag dev size_mb <<< "$mode"; free_mb=$(_free_vram_for_device "$dev")
                    if [[ -n "$free_mb" && "$size_mb" -gt 0 && "$EXPERT_PER_LAYER_MB" -gt 0 ]]; then
                        overshoot=$(( size_mb - free_mb )); (( overshoot < 0 )) && overshoot=0
                        bump=$(( (overshoot + EXPERT_PER_LAYER_MB - 1) / EXPERT_PER_LAYER_MB )); (( bump < 1 )) && bump=1
                        echo "  OOM on CUDA${dev}: tried ${size_mb}MB, free ${free_mb}MB → overshoot ${overshoot}MB → +${bump} layers to CPU"
                        N=$(( N + bump ))
                    else echo "  OOM on CUDA${dev:-?} — log lacked size/free, bumping N by 1"; N=$(( N + 1 )); fi
                    last_direction="up" ;;
                pinned_fail|pinned_cap_exceeded|pinned_hang)
                    if [[ "$GPU_BACKEND" == "Vulkan" ]]; then echo "  Pinned alloc failure on Vulkan — stepping DOWN"; N=$(( N - 1 )); last_direction="down"
                    elif (( ! no_pinned_tried )); then echo "  Host pinned alloc failure (${mode}) — setting GGML_CUDA_NO_PINNED=1"; LAUNCH_ENV_PREFIX+=(GGML_CUDA_NO_PINNED=1); no_pinned_tried=1
                    else echo "  Host alloc still failing with NO_PINNED (${mode}) — moving 1 layer onto GPU"; N=$(( N - 1 )); last_direction="down"; fi
                    ;;
                ram_oom) echo "  Host RAM exhausted (ram_oom) — moving 1 layer onto GPU"; N=$(( N - 1 )); last_direction="down" ;;
                *) echo "  Load failed (reason not classifiable) — bumping N by 1"; tail -3 "$SERVER_LOG" 2>/dev/null | sed 's/^/    /'
                    if [[ "$last_direction" == "up" ]]; then N=$(( N - 1 )); last_direction="down"; else N=$(( N + 1 )); last_direction="up"; fi ;;
            esac
            if (( N > MAX_N )); then echo "  Reached N=$N > MAX_N=$MAX_N — out of upward moves"; boundary_hit=1; break; fi
            if (( N < MIN_N )); then echo "  Reached N=$N < MIN_N=$MIN_N — out of downward moves"; boundary_hit=1; break; fi
        done
    fi

    if (( ! RECOVERED )); then
        _rejected_flag=""
        if [[ -f "$SERVER_LOG" ]]; then
            _rejected_flag=$(grep -oE "error: invalid argument: [^[:space:]]+" "$SERVER_LOG" 2>/dev/null | tail -1 | sed 's/^error: invalid argument: //')
        fi
        if [[ -n "$_rejected_flag" ]]; then
            echo ""; echo "═══ Launch failed: binary rejected a flag ═══"
            echo "  Flag rejected: ${_rejected_flag}"; echo "  Backend:       $( (( IS_IK_LLAMA )) && echo 'ik_llama.cpp' || echo 'llama.cpp (mainline)' )"
            echo "  Binary:        $LLAMA_SERVER"; echo ""
            echo "  This is a configuration mismatch, not a memory-fit problem."
            echo ""; echo "  Fixes:"
            echo "    1. Switch backend with --server-bin <correct binary>, OR"
            echo "    2. Re-tune for the current backend: --ai-tune --retune"
            if [[ "$_rejected_flag" =~ merge-qkv|merge-up-gate|grouped-expert|run-time-repack|cache-ram|defrag-thold|ctx-checkpoints ]]; then
                echo "    3. The rejected flag is ik_llama-only — switch to ik_llama.cpp"
            fi; exit 1
        fi
        echo ""; echo "FAILED: Bidirectional recovery exhausted."
        tail -20 "$SERVER_LOG" 2>/dev/null; echo ""
        echo "═══ Model does not fit on this system ═══"
        echo "  Model:           ${MODEL_NAME} — ${LAYER_COUNT} layers, ${EXPERT_COUNT} experts"
        echo "  File size:       ${TOTAL_SIZE_MB}MB"; echo "  Per-layer cost:  ${COST_PER_LAYER_MB}MB GPU / ${EXPERT_PER_LAYER_MB}MB CPU"
        echo ""; echo "  Measured ceilings (no margins):"
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}; echo "    GPU${GPU_INDEX[$gi]} (${GPU_NAME[$gi]}): ${MAX_GPU_LAYERS_PER[$gi]} layers max (${GPU_VRAM_FREE[$gi]}MB free)"
        done
        echo "    GPU total:      ${MAX_GPU_LAYERS} / ${LAYER_COUNT} layers"
        echo "    CPU [${CEIL_CPU_LABEL}]: ${MAX_CPU_LAYERS} / ${LAYER_COUNT} layers (${RAM_AVAIL_MB}MB RAM avail − ${RAM_OVERHEAD_PRE_MB}MB overhead)"
        if (( boundary_hit )); then echo "  Sweep result:    bidirectional recovery boxed in the boundary."; fi
        gap=$(( LAYER_COUNT - MAX_GPU_LAYERS - MAX_CPU_LAYERS ))
        if (( gap > 0 )); then gap_vram=$(( gap * COST_PER_LAYER_MB )); gap_ram=$(( gap * EXPERT_PER_LAYER_MB )); echo "  Static gap:      ${gap} layers — ~${gap_vram}MB more VRAM or ~${gap_ram}MB more RAM"; fi
        echo ""; echo "  Actionable next steps:"
        echo "    1. Free VRAM: close GPU workloads, or pass --gpus to add another card"
        if (( USER_NO_MMAP )); then echo "    2. Drop --no-mmap so the kernel can page experts on demand"
        else echo "    2. Add system RAM (cheapest upgrade for MoE expert offload)"; fi
        echo "    3. Use a smaller quantization (Q4 → IQ3 → IQ2) or smaller model"
        echo "    4. Reduce context: --ctx-size 16384 or --ctx-size 8192 (KV-bound cases)"; echo ""; exit 1
    fi
    kill_server "$RUNNING_PID"; run_with_restart "${COMMON_FLAGS[@]}"
fi

}

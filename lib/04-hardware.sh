#!/bin/bash
# lib/04-hardware.sh — GPU/CPU/RAM detection.
# Detects NVIDIA GPUs (nvidia-smi), AMD GPUs (amd-smi), and Vulkan fallback.
# Also detects CPU cores and system RAM.
#
# All detection is wrapped in _detect_hardware() — called from main entry
# after GPU_COUNT, CPU_ONLY, RAM_BUDGET_MB, GPUS_FILTER are known.

# ── Global GPU arrays (must be top-level — declare inside a function makes them local) ──
declare -a GPU_INDEX=() GPU_NAME=() GPU_VRAM_TOTAL=() GPU_VRAM_FREE=() GPU_PCIE_WIDTH=() GPU_PCIE_GEN=() GPU_BANDWIDTH=() GPU_COMPUTE_CAP=() GPU_ORDER=()

# ── Update check constants ─────────────────────────────────────
LLM_SERVER_REPO="${LLM_SERVER_REPO:-$HOME/llm-server}"
UPDATE_DISMISS_FILE="$CACHE_DIR/update_dismissed"
UPDATE_DISMISS_DAYS=7

# ── Placement cache key input ──────────────────────────────────
placement_cache_key_input() {
    local parts=(
        "version=$VERSION"
        "model=$MODEL_NAME"
        "bytes=$TOTAL_SIZE_BYTES"
        "ctx=$CTX_SIZE"
        "kv=$KV_QUALITY"
        "kv_placement=$KV_PLACEMENT"
        "backend=$BACKEND_TAG"
        "llama_server=$LLAMA_SERVER"
        "gpu_filter=${GPUS_FILTER:-all}"
        "ram_avail=$RAM_AVAIL_MB"
        "ram_budget=$RAM_BUDGET_MB"
        "gpu_count=$GPU_COUNT"
    )
    local i free_bucket total_bucket
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        free_bucket=$(( (GPU_VRAM_FREE[$i] / 1024) * 1024 ))
        total_bucket=$(( (GPU_VRAM_TOTAL[$i] / 1024) * 1024 ))
        parts+=("gpu${i}_index=${GPU_INDEX[$i]}")
        parts+=("gpu${i}_name=${GPU_NAME[$i]}")
        parts+=("gpu${i}_free_bucket=${free_bucket}")
        parts+=("gpu${i}_total_bucket=${total_bucket}")
        parts+=("gpu${i}_pcie=x${GPU_PCIE_WIDTH[$i]}g${GPU_PCIE_GEN[$i]}")
        parts+=("gpu${i}_bandwidth=${GPU_BANDWIDTH[$i]}")
        parts+=("gpu${i}_compute=${GPU_COMPUTE_CAP[$i]}")
    done
    printf '%s\n' "${parts[@]}"
}

# ── Hardware detection ─────────────────────────────────────────
_detect_hardware() {
    # Detect physical CPU cores
    PHYSICAL_CORES=$(lscpu 2>/dev/null | awk '/^Core\(s\) per socket:/ {cores=$NF} /^Socket\(s\):/ {socks=$NF} END {print cores * socks}' || true)
    PHYSICAL_CORES=${PHYSICAL_CORES:-4}

    # Detect RAM
    RAM_AVAIL_MB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo)
    RAM_TOTAL_MB=$(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo)

    echo "CPU: ${PHYSICAL_CORES} physical cores"

    # Apply --ram-budget cap
    if (( RAM_BUDGET_MB > 0 && RAM_BUDGET_MB < RAM_AVAIL_MB )); then
        RAM_AVAIL_MB=$RAM_BUDGET_MB
        echo "RAM: ${RAM_AVAIL_MB}MB available (capped by --ram-budget) / ${RAM_TOTAL_MB}MB total"
    else
        echo "RAM: ${RAM_AVAIL_MB}MB available / ${RAM_TOTAL_MB}MB total"
    fi

    # Detect GPUs
    GPU_COUNT=0
    if (( CPU_ONLY )); then
        echo "GPUs: skipped (--cpu flag)"
    elif command -v nvidia-smi &>/dev/null; then
        while IFS= read -r line; do
            IFS=',' read -ra fields <<< "$line"
            [[ ${#fields[@]} -lt 6 ]] && continue
            idx=$(echo "${fields[0]}" | tr -d ' ')
            name=$(echo "${fields[1]}" | sed 's/^ //')
            vram_total=$(echo "${fields[2]}" | tr -d ' ')
            vram_free=$(echo "${fields[3]}" | tr -d ' ')
            pcie_width=$(echo "${fields[4]}" | tr -d ' ')
            pcie_gen=$(echo "${fields[5]}" | tr -d ' ')
            compute_cap=$(echo "${fields[6]:-0.0}" | tr -d ' ')
            [[ -z "$compute_cap" || "$compute_cap" == "[N/A]" ]] && compute_cap="0.0"
            if (( vram_free < 500 )); then
                log "Skipping GPU $idx ($name): only ${vram_free}MB free"
                continue
            fi
            GPU_INDEX+=("$idx")
            GPU_NAME+=("$name")
            GPU_VRAM_TOTAL+=("$vram_total")
            GPU_VRAM_FREE+=("$vram_free")
            GPU_PCIE_WIDTH+=("$pcie_width")
            GPU_PCIE_GEN+=("$pcie_gen")
            GPU_COMPUTE_CAP+=("$compute_cap")
            GPU_BANDWIDTH+=( $(( pcie_width * pcie_gen )) )
            GPU_COUNT=$(( GPU_COUNT + 1 ))
        done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.width.current,pcie.link.gen.current,compute_cap --format=csv,noheader,nounits 2>/dev/null)
    fi
    GPU_BACKEND="cuda"

    # AMD-SMI GPU detection (ROCm / radeonsi-tools on Linux)
    if (( GPU_COUNT == 0 )) && ! (( CPU_ONLY )); then
        if command -v amd-smi &>/dev/null; then
            amd_names_output=$(amd-smi 2>/dev/null || true)
            declare -A amd_gpu_names
            if [[ -n "$amd_names_output" ]]; then
                prev_line=""
                while IFS= read -r cur_line; do
                    if [[ "$cur_line" == "|"* ]] && [[ "$cur_line" =~ ^\|[[:space:]]*([0-9]+)[[:space:]] ]]; then
                        idx="${BASH_REMATCH[1]}"
                        if [[ "$prev_line" == "|"* ]]; then
                            stripped="${prev_line#|}"
                            stripped="${stripped# }"
                            if [[ "$stripped" =~ ^[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+\.[0-9]+[[:space:]]+([^|]+) ]]; then
                                name="${BASH_REMATCH[1]}"
                                name="$(echo "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                                amd_gpu_names["$idx"]="$name"
                            fi
                        fi
                    fi
                    prev_line="$cur_line"
                done <<< "$amd_names_output"
            fi

            amd_output=$(amd-smi metric --mem 2>/dev/null || true)
            if [[ -n "$amd_output" ]]; then
                current_gpu=""
                current_total_gtt=0
                current_free_gtt=0
                current_gpu_name="Unknown AMD GPU"
                while IFS= read -r line; do
                    if [[ "$line" =~ ^[[:space:]]*GPU:[[:space:]]*([0-9]+) ]]; then
                        if [[ -n "$current_gpu" ]] && (( current_free_gtt >= 500 )); then
                            GPU_INDEX+=("$current_gpu")
                            GPU_NAME+=("$current_gpu_name")
                            GPU_VRAM_TOTAL+=("$current_total_gtt")
                            GPU_VRAM_FREE+=("$current_free_gtt")
                            GPU_PCIE_WIDTH+=(0)
                            GPU_PCIE_GEN+=(0)
                            GPU_COMPUTE_CAP+=(0.0)
                            GPU_BANDWIDTH+=(0)
                            GPU_COUNT=$(( GPU_COUNT + 1 ))
                            GPU_BACKEND="Vulkan"
                            log "  GPU ${current_gpu}: ${current_gpu_name} — ${current_total_gtt}MB GTT total, ${current_free_gtt}MB free"
                        fi
                        current_gpu="${BASH_REMATCH[1]}"
                        current_total_gtt=0
                        current_free_gtt=0
                        if [[ -n "${amd_gpu_names[$current_gpu]:-}" ]]; then
                            current_gpu_name="${amd_gpu_names[$current_gpu]}"
                        else
                            current_gpu_name="Unknown AMD GPU"
                        fi
                    elif [[ "$line" =~ TOTAL_GTT:[[:space:]]*([0-9]+) ]]; then
                        current_total_gtt=${BASH_REMATCH[1]}
                    elif [[ "$line" =~ FREE_GTT:[[:space:]]*([0-9]+) ]]; then
                        current_free_gtt=${BASH_REMATCH[1]}
                    fi
                done <<< "$amd_output"
                if [[ -n "$current_gpu" ]] && (( current_free_gtt >= 500 )); then
                    GPU_INDEX+=("$current_gpu")
                    GPU_NAME+=("$current_gpu_name")
                    GPU_VRAM_TOTAL+=("$current_total_gtt")
                    GPU_VRAM_FREE+=("$current_free_gtt")
                    GPU_PCIE_WIDTH+=(0)
                    GPU_PCIE_GEN+=(0)
                    GPU_COMPUTE_CAP+=(0.0)
                    GPU_BANDWIDTH+=(0)
                    GPU_COUNT=$(( GPU_COUNT + 1 ))
                    GPU_BACKEND="Vulkan"
                    log "  GPU ${current_gpu}: ${current_gpu_name} — ${current_total_gtt}MB GTT total, ${current_free_gtt}MB free"
                fi
                unset amd_gpu_names
            fi
        fi
    fi

    # Fallback: detect Vulkan/AMD/Intel GPUs when nvidia-smi and amd-smi are unavailable
    if (( GPU_COUNT == 0 )) && ! (( CPU_ONLY )); then
        if command -v vulkaninfo &>/dev/null; then
            vulkan_out=$(vulkaninfo 2>&1)
            gpu_name=""
            gpu_vram_mb=0
            in_gpu0=0
            pending_size=0
            while IFS= read -r line; do
                if [[ "$line" =~ ^GPU0: ]]; then in_gpu0=1; continue; fi
                if (( in_gpu0 )) && [[ "$line" =~ ^GPU[1-9] ]]; then break; fi
                if (( in_gpu0 )); then
                    if [[ "$line" =~ deviceName[[:space:]]*=[[:space:]]*(.*) ]]; then gpu_name="${BASH_REMATCH[1]}"; fi
                    if [[ "$line" =~ memoryHeaps\[ ]]; then pending_size=0; fi
                    if [[ "$line" =~ size[[:space:]]*=[[:space:]]*([0-9]+) ]]; then pending_size=${BASH_REMATCH[1]}; fi
                    if [[ "$line" =~ MEMORY_HEAP_DEVICE_LOCAL_BIT ]] && (( pending_size > 0 )); then
                        gpu_vram_mb=$(( gpu_vram_mb + pending_size / 1024 / 1024 ))
                        pending_size=0
                    fi
                fi
            done <<< "$vulkan_out"

            awk_total=$(echo "$vulkan_out" | awk '
                /^GPU0:/ { in_gpu0=1 }
                /^GPU[1-9]/ { in_gpu0=0 }
                !in_gpu0 { next }
                /memoryHeaps\[/ { in_heap=1; pending_size=0; has_device_local=0; next }
                /size[[:space:]]*=/ && in_heap { gsub(/[^0-9]/, "", $3); pending_size = $3 / 1024 / 1024 }
                /MEMORY_HEAP_DEVICE_LOCAL_BIT/ && in_heap && pending_size > 0 { total += pending_size; pending_size = 0 }
                /flags:/ && in_heap { in_flags=1; next }
                /None/ && in_flags && pending_size > 0 { total += pending_size; pending_size = 0 }
                /MEMORY_HEAP_/ && in_flags { in_flags=0 }
                END { print total+0 }
            ')
            gpu_vram_mb=$awk_total

            if [[ -n "$gpu_name" ]] && (( gpu_vram_mb > 500 )); then
                GPU_INDEX+=(0)
                GPU_NAME+=("$gpu_name")
                GPU_VRAM_TOTAL+=("$gpu_vram_mb")
                GPU_VRAM_FREE+=("$gpu_vram_mb")
                GPU_PCIE_WIDTH+=(0)
                GPU_PCIE_GEN+=(0)
                GPU_BANDWIDTH+=(0)
                GPU_COMPUTE_CAP+=(0.0)
                GPU_COUNT=$(( GPU_COUNT + 1 ))
                GPU_BACKEND="Vulkan"
                echo "GPUs: $GPU_COUNT detected (Vulkan fallback: $gpu_name, ${gpu_vram_mb}MB)"
            fi
        fi
    fi

    # Apply --gpus filter
    if [[ -n "$GPUS_FILTER" && $GPU_COUNT -gt 0 ]]; then
        IFS=',' read -ra ALLOWED_GPUS <<< "$GPUS_FILTER"
        declare -a NEW_GPU_INDEX NEW_GPU_NAME NEW_GPU_VRAM_TOTAL NEW_GPU_VRAM_FREE NEW_GPU_PCIE_WIDTH NEW_GPU_PCIE_GEN NEW_GPU_BANDWIDTH NEW_GPU_COMPUTE_CAP
        NEW_GPU_COUNT=0
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            for allowed in "${ALLOWED_GPUS[@]}"; do
                if [[ "${GPU_INDEX[$i]}" == "$allowed" ]]; then
                    NEW_GPU_INDEX+=("${GPU_INDEX[$i]}")
                    NEW_GPU_NAME+=("${GPU_NAME[$i]}")
                    NEW_GPU_VRAM_TOTAL+=("${GPU_VRAM_TOTAL[$i]}")
                    NEW_GPU_VRAM_FREE+=("${GPU_VRAM_FREE[$i]}")
                    NEW_GPU_PCIE_WIDTH+=("${GPU_PCIE_WIDTH[$i]}")
                    NEW_GPU_PCIE_GEN+=("${GPU_PCIE_GEN[$i]}")
                    NEW_GPU_BANDWIDTH+=("${GPU_BANDWIDTH[$i]}")
                    NEW_GPU_COMPUTE_CAP+=("${GPU_COMPUTE_CAP[$i]}")
                    (( NEW_GPU_COUNT++ )) || true
                    break
                fi
            done
        done
        if (( NEW_GPU_COUNT == 0 )); then
            echo "Error: --gpus filter '$GPUS_FILTER' matched no detected GPUs"
            exit 1
        fi
        GPU_INDEX=("${NEW_GPU_INDEX[@]}")
        GPU_NAME=("${NEW_GPU_NAME[@]}")
        GPU_VRAM_TOTAL=("${NEW_GPU_VRAM_TOTAL[@]}")
        GPU_VRAM_FREE=("${NEW_GPU_VRAM_FREE[@]}")
        GPU_PCIE_WIDTH=("${NEW_GPU_PCIE_WIDTH[@]}")
        GPU_PCIE_GEN=("${NEW_GPU_PCIE_GEN[@]}")
        GPU_BANDWIDTH=("${NEW_GPU_BANDWIDTH[@]}")
        GPU_COMPUTE_CAP=("${NEW_GPU_COMPUTE_CAP[@]}")
        GPU_COUNT=$NEW_GPU_COUNT
        echo "GPU filter: using only GPU(s) $GPUS_FILTER ($GPU_COUNT matched)"
    fi

    if (( GPU_COUNT == 0 )); then
        echo "GPUs: none detected (CPU-only mode)"
    else
        echo "GPUs: $GPU_COUNT detected"
        GPU_ORDER=($(for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
                cc_int=$(awk "BEGIN{printf \"%d\", ${GPU_COMPUTE_CAP[$i]:-0}*10}")
                printf "%d %d %d %d\n" "${GPU_BANDWIDTH[$i]}" "${GPU_VRAM_TOTAL[$i]}" "$cc_int" "$i"
            done | sort -k1,1rn -k2,2rn -k3,3rn | awk '{print $NF}'))
        for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
            gi=${GPU_ORDER[$i]}
            echo "  GPU${GPU_INDEX[$gi]}: ${GPU_NAME[$gi]} ${GPU_VRAM_FREE[$gi]}MB free / ${GPU_VRAM_TOTAL[$gi]}MB total (PCIe x${GPU_PCIE_WIDTH[$gi]} gen${GPU_PCIE_GEN[$gi]})"
        done
    fi

    # Calculate total VRAM
    TOTAL_VRAM_MB=0
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        TOTAL_VRAM_MB=$(( TOTAL_VRAM_MB + GPU_VRAM_TOTAL[$i] ))
    done
}

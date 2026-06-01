#!/bin/bash
# lib/12-show-configs.sh — show_configs_table and show_configs_detail.

_show_backend_header() {
    echo "Backends (last fetched state; run --update to refresh):"
    for pair in "ik_llama.cpp:$HOME/ik_llama.cpp" "llama.cpp:$HOME/llama.cpp" "llama.cpp-vulkan:$HOME/llama.cpp"; do
        local label=${pair%%:*} dir=${pair##*:}
        local extra=""
        [[ "$label" == "llama.cpp-vulkan" ]] && { [[ -d "$dir/build-vulkan" ]] || continue; extra=" (build-vulkan)"; }
        printf "  %-18s %s%s\n" "$label" "$(_backend_status "$dir")" "$extra"
    done
    echo
}

_tune_backend_tag() {
    local f=$1
    case "$f" in *_ik.json) echo "ik_llama" ;; *_llama.json) echo "llama" ;; *_vulkan.json) echo "llama-vk" ;; *) echo "llama" ;; esac
}

_human_age() {
    local s=$1
    if (( s < 60 )); then echo "${s}s ago"
    elif (( s < 3600 )); then echo "$(( s/60 ))m ago"
    elif (( s < 86400 )); then echo "$(( s/3600 ))h ago"
    else echo "$(( s/86400 ))d ago"
    fi
}

show_configs_table() {
    _show_backend_header
    if [[ ! -d "$CACHE_DIR" ]]; then echo "No cache dir yet ($CACHE_DIR). Run: llm-server <model> [--ai-tune]"; return; fi
    shopt -s nullglob
    local tunes=("$CACHE_DIR"/tune_*.json) confs=("$CACHE_DIR"/*.conf)
    shopt -u nullglob
    if (( ${#tunes[@]} == 0 && ${#confs[@]} == 0 )); then echo "No cached configs. Run: llm-server <model> [--ai-tune]"; return; fi
    printf "%-40s %-6s %-5s %-8s %-7s %-12s %-9s\n" "MODEL" "TYPE" "TUNE" "GAIN" "TOK/S" "CACHED" "BACKEND"
    printf "%-40s %-6s %-5s %-8s %-7s %-12s %-9s\n" "----------------------------------------" "------" "-----" "--------" "-------" "------------" "---------"
    declare -A TUNE_FILE TUNE_GAIN TUNE_TPS TUNE_ROUNDS TUNE_AGE TUNE_BACKEND TUNE_TYPE
    local now; now=$(date +%s)
    for f in "${tunes[@]}"; do
        [[ -f "$f" ]] || continue
        local info
        info=$(python3 - <<PY 2>/dev/null
import json
try:
    d = json.load(open("$f"))
    model = d.get("model","?")
    best = d.get("best_config",{}) or {}
    tps = best.get("gen_tps") or 0
    base = d.get("baseline_gen_tps") or 0
    gain = ((tps-base)/base*100) if (base>0 and tps>0) else 0
    rounds = d.get("rounds", 0)
    import datetime
    ta = d.get("tuned_at","")
    try: dt = datetime.datetime.fromisoformat(ta.replace("Z","+00:00"))
    except: dt = None
    age = int(dt.timestamp()) if dt else 0
    print(f"{model}\t{tps:.1f}\t{gain:+.2f}\t{rounds}\t{age}")
except: pass
PY
)
        [[ -z "$info" ]] && continue
        local model tps gain rounds age
        IFS=$'\t' read -r model tps gain rounds age <<< "$info"
        local bck; bck=$(_tune_backend_tag "$(basename "$f")")
        local key="${model}__${bck}"
        if [[ -z "${TUNE_FILE[$key]:-}" ]] || (( age > ${TUNE_AGE[$key]:-0} )); then
            TUNE_FILE[$key]=$f; TUNE_TPS[$key]=$tps; TUNE_GAIN[$key]=$gain
            TUNE_ROUNDS[$key]=$rounds; TUNE_AGE[$key]=$age; TUNE_BACKEND[$key]=$bck
        fi
    done
    _model_type() { case "${1,,}" in *a3b*|*a10b*|*a24b*|*-moe-*|*_moe_*|*minimax*|*qwen3.5-122b*) echo "MoE" ;; *) echo "dense" ;; esac; }
    for key in "${!TUNE_FILE[@]}"; do
        local model=${key%__*} bck=${TUNE_BACKEND[$key]} short="$model" type
        type=$(_model_type "$model"); (( ${#short} > 40 )) && short="${short:0:37}..."
        local cached; cached=$(_human_age $(( now - TUNE_AGE[$key] )))
        printf "%-40s %-6s %-5s %-8s %-7s %-12s %-9s\n" "$short" "$type" "${TUNE_ROUNDS[$key]}rnd" "${TUNE_GAIN[$key]}%" "${TUNE_TPS[$key]}" "$cached" "$bck"
    done
    declare -A SEEN_MODEL PLACEMENT_MTIME
    for f in "${confs[@]}"; do
        [[ -f "$f" ]] || continue
        local model
        model=$(grep -m1 -iE "^# *Auto[- ]tuned config for" "$f" 2>/dev/null | sed -E 's/.*for +//; s/[[:space:]]+$//; s/\.gguf$//' || true)
        [[ -z "$model" ]] && continue
        local hit=0
        for key in "${!TUNE_FILE[@]}"; do [[ "${key%__*}" == "$model.gguf" || "${key%__*}" == "$model" ]] && { hit=1; break; }; done
        (( hit )) && continue
        local mt; mt=$(stat -c %Y "$f" 2>/dev/null || echo "$now")
        if [[ -z "${PLACEMENT_MTIME[$model]:-}" ]] || (( mt > ${PLACEMENT_MTIME[$model]} )); then PLACEMENT_MTIME[$model]=$mt; fi
    done
    for model in "${!PLACEMENT_MTIME[@]}"; do
        local short="$model" type; type=$(_model_type "$model")
        (( ${#short} > 40 )) && short="${short:0:37}..."
        printf "%-40s %-6s %-5s %-8s %-7s %-12s %-9s\n" "$short" "$type" "-" "-" "-" "$(_human_age $(( now - PLACEMENT_MTIME[$model] )))" "placement"
    done
}

show_configs_detail() {
    local model_arg=$1
    _show_backend_header
    local model_path="$model_arg"
    [[ -f "$model_path" ]] || model_path="${LLM_MODEL_DIR:-$HOME/ai_models}/$model_arg"
    [[ -f "$model_path" ]] || { echo "Model not found: $model_arg"; exit 1; }
    local model_name; model_name=$(basename "$model_path")
    echo "Model: $model_name"
    echo "  Path: $model_path"
    echo "  Size: $(du -h "$model_path" 2>/dev/null | awk '{print $1}' || true)"
    echo
    shopt -s nullglob
    local tunes=("$CACHE_DIR"/tune_"$model_name"_*.json)
    shopt -u nullglob
    if (( ${#tunes[@]} == 0 )); then echo "No tune cache for this model. Run: llm-server \"$model_arg\" --ai-tune"; fi
    local now; now=$(date +%s)
    for f in "${tunes[@]}"; do
        local bck; bck=$(_tune_backend_tag "$(basename "$f")")
        echo "Tune cache ($bck): $f"
        python3 - <<PY 2>/dev/null
import json, datetime
d = json.load(open("$f"))
best = d.get("best_config", {}) or {}
base = d.get("baseline_gen_tps") or 0
tps = best.get("gen_tps") or 0
gain = ((tps-base)/base*100) if (base>0 and tps>0) else 0
print(f"  Config:     {best.get('name','?')}")
print(f"  Baseline:   {base:.2f} tok/s gen, {d.get('baseline_pp_tps',0):.2f} tok/s pp")
print(f"  Best:       {tps:.2f} tok/s gen, {best.get('pp_tps',0):.2f} tok/s pp  ({gain:+.2f}%)")
print(f"  Rounds:     {d.get('rounds', 0)}")
print(f"  Tuned at:   {d.get('tuned_at','?')}")
flags = best.get("flags", {})
if flags:
    print(f"  Tuned flags: " + " ".join(f"{k} {v}" if v else k for k,v in flags.items()))
PY
        echo
    done
    echo "Resolved launch command:"
    SHOW_CONFIGS=0 LLM_ASSUME_YES=1 LLM_SERVER_UPDATE_CHECKED=1 \
        "$0" --server-bin "$LLAMA_SERVER" --ctx-size "$CTX_SIZE" "$model_arg" --dry-run 2>&1 \
        | sed -n '/^Command:/,$p' | sed 's/^/  /'
}

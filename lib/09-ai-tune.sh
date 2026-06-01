#!/bin/bash
# lib/09-ai-tune.sh — AI-driven flag optimization loop.
# Contains: append_tune_history, load_tune_history, build_hw/model profiles,
# bench_gen_tps, query_llm_chat, build_fixed_flags, parse_tune_overrides,
# _tune_fmt_duration, _tune_progress_line, diff_flags, show_comparison,
# ai_tune(), load_tune_cache, maybe_ai_tune_launch_flags, maybe_prompt_first_run_ai_tune.

# ── AI Tune constants (set during _init_ai_tune) ──────────────
# These are set by _init_ai_tune() after DO_UNLIMITED is known.
AI_TUNE_ROUNDS=12
AI_TUNE_MAX_CRASHES=4
TUNE_HISTORY_FILE="$CACHE_DIR/tune_history.jsonl"
TUNE_LONG_TIMEOUT=0

# ── Append tune history ───────────────────────────────────────
append_tune_history() {
    local round="$1" gen_tps="$2" pp_tps="$3" status="$4" flags_json="$5" config_name="$6"
    mkdir -p "$CACHE_DIR"
    echo "$config_name" > "$tune_msg_dir/hist_name.txt"
    python3 -c "
import json; from datetime import datetime, timezone
entry = {
    'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'model': '${MODEL_NAME}', 'hw_hash': '${hw_hash}', 'round': ${round},
    'name': open('${tune_msg_dir}/hist_name.txt').read().strip(),
    'gen_tps': ${gen_tps}, 'pp_tps': ${pp_tps}, 'status': '${status}',
    'flags': json.loads(open('/dev/stdin').read())
}
with open('${TUNE_HISTORY_FILE}', 'a') as f: f.write(json.dumps(entry) + '\n')
" <<< "$flags_json"
}

# ── Load tune history ─────────────────────────────────────────
load_tune_history() {
    local hw_hash="$1"
    if [[ ! -f "$TUNE_HISTORY_FILE" ]]; then echo "(No previous tuning data)"; return; fi
    python3 -c "
import json, sys
hw_hash = '${hw_hash}'; history = []
try:
    for line in open('${TUNE_HISTORY_FILE}'):
        line = line.strip()
        if not line: continue
        try: history.append(json.loads(line))
        except: pass
except: pass
if not history: print('(No previous tuning data)'); sys.exit(0)
by_model = {}; same_hw = [e for e in history if e.get('hw_hash') == hw_hash]
diff_hw = [e for e in history if e.get('hw_hash') != hw_hash]
lines = []
if same_hw:
    lines.append('## Results on THIS hardware:')
    models_seen = {}
    for e in same_hw:
        m = e.get('model', '?')
        if m not in models_seen: models_seen[m] = []
        models_seen[m].append(e)
    for m, entries in models_seen.items():
        ok = [e for e in entries if e.get('status') == 'ok' and e.get('gen_tps', 0) > 0]
        crashed = [e for e in entries if e.get('status') == 'crashed']
        if ok:
            best = max(ok, key=lambda e: e.get('gen_tps', 0))
            lines.append(f'  {m}: best={best[\"gen_tps\"]} tok/s ({best[\"name\"]})')
            flags = best.get('flags', {})
            if flags: lines.append(f'    flags: {\" \".join(f\"{k} {v}\" if v else k for k, v in flags.items())}')
        if crashed:
            crash_names = [e.get('name', '?') for e in crashed[:3]]
            lines.append(f'  {m}: crashed configs: {\", \".join(crash_names)}')
if diff_hw:
    lines.append(''); lines.append('## Results on OTHER hardware (for reference):')
    models_seen = {}
    for e in diff_hw:
        m = e.get('model', '?')
        if m not in models_seen: models_seen[m] = []
        models_seen[m].append(e)
    for m, entries in list(models_seen.items())[:5]:
        ok = [e for e in entries if e.get('status') == 'ok' and e.get('gen_tps', 0) > 0]
        if ok:
            best = max(ok, key=lambda e: e.get('gen_tps', 0))
            lines.append(f'  {m}: best={best[\"gen_tps\"]} tok/s ({best[\"name\"]})')
output = '\n'.join(lines[:50])
print(output if output.strip() else '(No previous tuning data)')
" 2>/dev/null || echo "(No previous tuning data)"
}

# ── HW/Model profiles ─────────────────────────────────────────
build_hw_profile() {
    local gpu_json="["
    local i
    for i in $(seq 0 $(( GPU_COUNT - 1 ))); do
        (( i > 0 )) && gpu_json+=","
        gpu_json+="{\"index\":${GPU_INDEX[$i]:-0},\"name\":\"${GPU_NAME[$i]:-}\",\"vram_free_mb\":${GPU_VRAM_FREE[$i]:-0},\"vram_total_mb\":${GPU_VRAM_TOTAL[$i]:-0},\"pcie_width\":${GPU_PCIE_WIDTH[$i]:-0},\"pcie_gen\":${GPU_PCIE_GEN[$i]:-0}}"
    done
    gpu_json+="]"
    cat <<HWEOF
{
  "gpu_count": ${GPU_COUNT}, "gpus": ${gpu_json},
  "ram_available_mb": ${RAM_AVAIL_MB:-0}, "physical_cores": ${PHYSICAL_CORES:-4}
}
HWEOF
}

build_model_profile() {
    local kv_label="unknown"
    case "${KV_QUALITY:-}" in f16) kv_label="f16 (${KV_TOTAL_MB:-0}MB)" ;; q8_0) kv_label="q8_0 (${KV_TOTAL_MB:-0}MB)" ;; q4_0) kv_label="q4_0 (${KV_TOTAL_MB:-0}MB)" ;; *) kv_label="${KV_QUALITY:-?} (${KV_TOTAL_MB:-0}MB)" ;; esac
    cat <<MPEOF
{
  "name": "${MODEL_NAME}", "architecture": "${MODEL_ARCH}", "layers": ${LAYER_COUNT},
  "experts": ${EXPERT_COUNT}, "size_mb": ${TOTAL_SIZE_MB}, "kv_heads": ${HEAD_COUNT_KV},
  "is_moe": ${IS_MOE:-0}, "has_fused": ${HAS_FUSED},
  "backend": "$(if (( IS_IK_LLAMA )); then echo "ik_llama.cpp"; else echo "llama.cpp (mainline)"; fi)",
  "kv_cache": "${kv_label}", "total_vram_mb": ${TOTAL_VRAM_MB:-0},
  "total_ram_mb": ${RAM_AVAIL_MB:-0}, "strategy": "${STRATEGY}"
}
MPEOF
}

# ── Benchmark for AI Tune ─────────────────────────────────────
bench_gen_tps() {
    local url="http://127.0.0.1:${PORT}"
    local prompt="You are an expert systems architect and software engineer. Provide a comprehensive technical analysis covering these topics in detail: 1. Distributed system architecture patterns. 2. Performance optimization techniques. 3. Database design. 4. Concurrency primitives. 5. Network protocols. For each topic, provide real-world examples, performance characteristics, and common pitfalls to avoid."
    local total_gen=0 total_pp=0 request_count=0
    for request in 1 2 3; do
        local result
        result=$(curl -sf "$url/v1/chat/completions" \
            -H "Content-Type: application/json" --max-time 180 \
            -d "{\"model\":\"test\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant. /no_think\"},{\"role\":\"user\",\"content\":$(printf '%s' "$prompt" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))")}],\"max_tokens\":1500,\"temperature\":0.3}" 2>/dev/null)
        if [[ -z "$result" ]]; then continue; fi
        local gen pp
        gen=$(python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); g=t.get('predicted_per_second',0); print(f'{g:.2f}' if g>0 else '0')" <<< "$result" 2>/dev/null || echo "0")
        pp=$(python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); p=t.get('prompt_per_second',0); print(f'{p:.2f}' if p>0 else '0')" <<< "$result" 2>/dev/null || echo "0")
        if [[ "$gen" != "0" && "$pp" != "0" ]]; then
            total_gen=$(python3 -c "print(f'{float('$total_gen') + float('$gen'):.2f}')" 2>/dev/null || echo "0")
            total_pp=$(python3 -c "print(f'{float('$total_pp') + float('$pp'):.2f}')" 2>/dev/null || echo "0")
            (( ++request_count ))
        fi
    done
    if (( request_count > 0 )); then
        local avg_gen avg_pp
        avg_gen=$(python3 -c "print(f'{float('$total_gen') / $request_count:.2f}')" 2>/dev/null || echo "0")
        avg_pp=$(python3 -c "print(f'{float('$total_pp') / $request_count:.2f}')" 2>/dev/null || echo "0")
        echo "$avg_gen $avg_pp"
    else
        echo "0 0"
    fi
}

# ── Query LLM chat ────────────────────────────────────────────
query_llm_chat() {
    local messages_json="$1"
    local url="http://127.0.0.1:${PORT}"
    local response
    response=$(curl -sf "$url/v1/chat/completions" \
        -H "Content-Type: application/json" --max-time 1800 \
        -d "{\"messages\":${messages_json},\"max_tokens\":16384,\"temperature\":0.3,\"chat_template_kwargs\":{\"enable_thinking\":false}}" 2>/dev/null) || true
    if [[ -z "$response" ]]; then echo "ERROR"; return 0; fi
    python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']
    content = c.get('content', '') or ''
    reasoning = c.get('reasoning_content', '') or ''
    text = content if '{' in content else (reasoning if '{' in reasoning else content)
    print(text)
except: print('ERROR')
" <<< "$response"
}

# ── Build fixed flags ─────────────────────────────────────────
build_fixed_flags() {
    local -a flags_arr=(-m "$MODEL_PATH" --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" --reasoning --mmproj)
    if (( !DO_UNLIMITED )); then
        flags_arr+=(--device --tensor-split --split-mode -mg -ngl -ot --n-cpu-moe)
    fi
    if (( ${#TUNE_EXTRA_FIXED[@]} > 0 )); then
        local f
        for f in "${TUNE_EXTRA_FIXED[@]}"; do
            local key="$f"
            case "$key" in --*=*) key="${key%%=*}" ;; --*) key="$key" ;; -*[!-]*) key="$key" ;; *) key="--$key" ;; esac
            local already=0 e
            for e in "${flags_arr[@]}"; do
                local ekey="$e"
                case "$ekey" in --*=*) ekey="${ekey%%=*}" ;; --*) ekey="$ekey" ;; -*[!-]*) ekey="$ekey" ;; *) ekey="--$ekey" ;; esac
                [[ "$ekey" == "$key" ]] && { already=1; break; }
            done
            (( already )) || flags_arr+=("$f")
        done
    fi
    if (( ${#TUNE_USER_LOCKED_KEYS[@]} > 0 )); then
        local f
        for f in "${TUNE_USER_LOCKED_KEYS[@]}"; do
            local key="$f"
            case "$key" in --*=*) key="${key%%=*}" ;; --*) key="$key" ;; -*[!-]*) key="$key" ;; *) key="--$key" ;; esac
            local already=0 e
            for e in "${flags_arr[@]}"; do
                local ekey="$e"
                case "$ekey" in --*=*) ekey="${ekey%%=*}" ;; --*) ekey="$ekey" ;; -*[!-]*) ekey="$ekey" ;; *) ekey="--$ekey" ;; esac
                [[ "$ekey" == "$key" ]] && { already=1; break; }
            done
            (( already )) || flags_arr+=("$f")
        done
    fi
    local IFS=' '
    echo "${flags_arr[*]}"
}

# ── Try start with overrides ──────────────────────────────────
declare -a TUNE_OVERRIDE_FLAGS=()
declare -a TUNE_OVERRIDE_KEYS=()

try_start_with_overrides() {
    local deduped=() skip_next=0
    for (( i=0; i<${#COMMON_FLAGS[@]}; i++ )); do
        if (( skip_next )); then skip_next=0; continue; fi
        local flag="${COMMON_FLAGS[$i]}"
        local dominated=0
        for key in "${TUNE_OVERRIDE_KEYS[@]}"; do
            if [[ "$flag" == "$key" ]]; then dominated=1
                if (( i + 1 < ${#COMMON_FLAGS[@]} )) && [[ "${COMMON_FLAGS[$((i+1))]}" != -* ]]; then skip_next=1; fi
                break
            fi
        done
        (( dominated )) || deduped+=("$flag")
    done
    local mode="${TRY_START_MODE:-tee}"
    launch_backend "$mode" "${deduped[@]}" "${TUNE_OVERRIDE_FLAGS[@]}"
    local pid="$RUNNING_PID"
    wait_for_ready "$pid"
}

parse_tune_overrides() {
    local overrides_json="$1"
    TUNE_OVERRIDE_FLAGS=(); TUNE_OVERRIDE_KEYS=()
    local kv_locked="$KV_QUALITY_EXPLICIT"
    local spec_enabled=0
    [[ -n "$SPEC_TYPE" && "$SPEC_TYPE" != "none" ]] && spec_enabled=1
    local locked_keys_env
    locked_keys_env=$(printf '%s\n' "${TUNE_USER_LOCKED_KEYS[@]:-}")
    local moe_detected=0 multi_gpu=0
    (( EXPERT_COUNT > 0 )) && moe_detected=1
    (( GPU_COUNT >= 2 )) && multi_gpu=1
    while IFS= read -r -d '' item; do
        case "$item" in F$'\t'*) TUNE_OVERRIDE_FLAGS+=("${item#F$'\t'}") ;; K$'\t'*) TUNE_OVERRIDE_KEYS+=("${item#K$'\t'}") ;; esac
    done < <(KV_LOCKED="$kv_locked" WSL2="$IS_WSL2" SPEC_ENABLED="$spec_enabled" \
             MOE_DETECTED="$moe_detected" MULTI_GPU="$multi_gpu" \
             TUNE_USER_LOCKED_KEYS="$locked_keys_env" python3 -c "
import json, os, sys
try: overrides = json.loads(sys.stdin.read())
except json.JSONDecodeError: overrides = {}
if not isinstance(overrides, dict): overrides = {}
protected = {'-m', '--host', '--port', '--ctx-size', '--mmproj', '-ot', '--reasoning'}
protected |= {k for k in os.environ.get('TUNE_USER_LOCKED_KEYS', '').splitlines() if k}
moe = os.environ.get('MOE_DETECTED') == '1'; multi_gpu = os.environ.get('MULTI_GPU') == '1'
if moe or not multi_gpu: protected |= {'--device', '--tensor-split', '--split-mode', '-mg', '-ngl', '-ot', '--n-cpu-moe'}
if os.environ.get('KV_LOCKED') == '1': protected |= {'--cache-type-k', '--cache-type-v'}
if os.environ.get('WSL2') == '1': protected |= {'--mlock', '-mlk'}
if os.environ.get('SPEC_ENABLED') != '1': protected |= {'--spec-type', '--spec-draft-n-max'}
aliases = {'--mg':'-mg','--main-gpu':'-mg','--batch-size':'-b','--ubatch-size':'-ub','--ngl':'-ngl','--n-gpu-layers':'-ngl','--n-cpu-moe':'--n-cpu-moe','-ncmoe':'--n-cpu-moe','--override-tensor':'-ot','--threads':'--threads','-t':'--threads','--threads-batch':'--threads-batch','-tb':'--threads-batch','--parallel':'--parallel','-np':'--parallel','--flash-attn':'--flash-attn','-fa':'--flash-attn','--reasoning':'--reasoning','-rea':'--reasoning'}
normalized = {}
for k, v in overrides.items():
    if not isinstance(k, str) or not k.startswith('-'): continue
    k = aliases.get(k, k); normalized[k] = v
records = []
for k, v in normalized.items():
    if k in protected: continue
    if v is False: continue
    records.append(('K', k)); records.append(('F', k))
    if v is not True and v != '' and v is not None: records.append(('F', str(v)))
for kind, value in records: sys.stdout.write(kind + '\t' + value + '\0')
" <<< "$overrides_json" || true)
}

# ── Tune formatting helpers ────────────────────────────────────
_tune_fmt_duration() {
    local s="$1"; (( s < 0 )) && s=0
    if (( s < 60 )); then printf '%ds' "$s"
    elif (( s < 3600 )); then printf '%dm%02ds' $(( s / 60 )) $(( s % 60 ))
    else printf '%dh%02dm' $(( s / 3600 )) $(( (s % 3600) / 60 ))
    fi
}

_tune_progress_line() {
    local done="$1" total="$2" start="$3" bgen="$4" bname="$5" base="$6"
    local now elapsed remain avg_per eta_rem improvement color reset dim
    now=$(date +%s); elapsed=$(( now - start )); remain=$(( total - done ))
    (( remain < 0 )) && remain=0
    local slots_done=$(( done + 1 ))
    if (( slots_done > 0 && elapsed > 0 )); then avg_per=$(( elapsed / slots_done )); else avg_per=0; fi
    eta_rem=$(( avg_per * remain ))
    improvement=""
    if [[ -n "$base" && "$base" != "0" && "$base" != "0.00" ]]; then
        improvement=$(python3 -c "b=float('$base'); g=float('$bgen'); print(f'{(g-b)/b*100:+.2f}%')" 2>/dev/null || echo "")
    fi
    local use_color=0
    if [[ -z "${NO_COLOR:-}" ]]; then
        if [[ -n "${FORCE_COLOR:-}" || -n "${CLICOLOR_FORCE:-}" ]]; then use_color=1
        elif [[ -t 1 || -t 2 ]]; then use_color=1
        elif [[ -w /dev/tty && "${TERM:-}" != "dumb" ]]; then use_color=1; fi
    fi
    if (( use_color )); then color=$'\033[1;36m'; dim=$'\033[2m'; reset=$'\033[0m'
    else color=""; dim=""; reset=""; fi
    local el_s eta_s pct bar filled empty i
    el_s=$(_tune_fmt_duration "$elapsed"); eta_s=$(_tune_fmt_duration "$eta_rem")
    pct=0; (( total > 0 )) && pct=$(( done * 100 / total )); (( pct > 100 )) && pct=100
    filled=$(( pct / 5 )); empty=$(( 20 - filled )); bar=""
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=0; i<empty;  i++)); do bar+="-"; done
    local block
    block=$(printf '\n%s================ AI Tune progress ================%s\n\r' "$color" "$reset")
    block+=$(printf '%s  round %d/%d  [%s]  %d%%%s\n\r' "$color" "$done" "$total" "$bar" "$pct" "$reset")
    block+=$(printf '%s  elapsed %s | ETA %s | best: %s @ %s tok/s %s%s%s\n\r' \
        "$color" "$el_s" "$eta_s" "$bname" "$bgen" "$dim" "$improvement" "$reset")
    block+=$(printf '%s==================================================%s\n\r' "$color" "$reset")
    printf '%s' "$block"
    { printf '%s' "$block" > /dev/tty; } 2>/dev/null || true
}

# ── Flag diff helper ───────────────────────────────────────────
diff_flags() {
    local prev_json="$1" curr_json="$2" prev_gen="$3" prev_pp="$4"
    python3 -c "
import json, sys
def fmt_val(v):
    if v is True: return 'true'
    if v is False: return 'false'
    return str(v)
try: prev = json.loads(sys.argv[1]) if sys.argv[1] else {}
except: prev = {}
try: curr = json.loads(sys.argv[2]) if sys.argv[2] else {}
except: curr = {}
all_keys = sorted(set(list(prev.keys()) + list(curr.keys())))
changed=[]; added=[]; removed=[]
for k in all_keys:
    in_prev = k in prev; in_curr = k in curr
    if in_prev and in_curr:
        if str(prev[k]) != str(curr[k]): changed.append((k, prev[k], curr[k]))
    elif in_curr and not in_prev: added.append((k, curr[k]))
    elif in_prev and not in_curr: removed.append((k, prev[k]))
max_key_len = max((len(k) for k in all_keys), default=0); max_val_len = 10
for flag, old_v, new_v in changed:
    ov = f'[{fmt_val(old_v)}]'.ljust(max_val_len); nv = f'[{fmt_val(new_v)}]'.ljust(max_val_len)
    print(f'    - {flag}: {ov} → {nv}')
for flag, val in added: print(f'    + {flag}: [new] [{fmt_val(val)}]')
for flag, val in removed: print(f'    - {flag}: [{fmt_val(val)}] → [removed]')
" "$prev_json" "$curr_json" 2>/dev/null || true
}

show_comparison() {
    local round_gen="$1" round_pp="$2" prev_gen="$3" prev_pp="$4" best_gen="$5" best_pp="$6" best_name="$7" best_round="$8"
    python3 -c "
import sys
def tps(v):
    try: return f'{float(v):.2f}'
    except: return '0.00'
rg = float('$round_gen') if '$round_gen' else 0.0; rp = float('$round_pp')  if '$round_pp'  else 0.0
pg = float('$prev_gen')  if '$prev_gen'  else None; pp = float('$prev_pp')   if '$prev_pp'   else None
bg = float('$best_gen')  if '$best_gen'  else None; bp = float('$best_pp')   if '$best_pp'   else None
bname = '$best_name'; br    = int('$best_round') if '$best_round' else 0
has_prev = pg is not None and pg > 0; has_best = bg is not None and bg > 0
if has_prev:
    print('  Δ Config → Baseline Comparison:')
    print(f'    Baseline (prev round):  {tps(pg)} gen tok/s  (pp: {tps(pp)})')
    print(f'    This config:            {tps(rg)} gen tok/s  (pp: {tps(rp)})')
    dg = rg - pg; dp = rp - pp; pg_pct = dg / pg * 100 if pg > 0 else 0.0; pp_pct = dp / pp * 100 if pp > 0 else 0.0
    print(f'    Δ: {dg:+.2f} gen tok/s ({pg_pct:+.2f}%) | PP: {dp:+.2f} tok/s ({pp_pct:+.2f}%)')
if has_best and br > 0:
    print(f'  Best so far: {bname} @ {tps(bg)} gen tok/s (round {br}, pp: {tps(bp)})')
    if rg > 0:
        dg2 = rg - bg; pg2 = dg2 / bg * 100 if bg > 0 else 0.0
        print(f'    Δ vs Best: {dg2:+.2f} gen tok/s ({pg2:+.2f}%)' + (' — NEW BEST!' if rg >= bg else ''))
" 2>/dev/null || true
}

# ── Main AI Tune function ──────────────────────────────────────
ai_tune() {
    mkdir -p "$CACHE_DIR"
    local vision_suffix=""
    [[ -n "$MMPROJ_PATH" ]] && vision_suffix="_v"
    local tune_cache_key="${MODEL_NAME}_${TOTAL_SIZE_BYTES}_hw$(echo "${GPU_COUNT}_$(for i in $(seq 0 $((GPU_COUNT-1))); do echo -n "${GPU_NAME[$i]}_"; done)" | md5sum | cut -c1-8)${vision_suffix}_${BACKEND_TAG}"
    local tune_cache_suffix=""
    (( DO_UNLIMITED )) && tune_cache_suffix="_unlimited"
    local tune_cache="$CACHE_DIR/tune_${tune_cache_key}${tune_cache_suffix}.json"

    # Check existing cache
    if [[ -f "$tune_cache" && "$RETUNE" == "0" && "$DO_AI_TUNE" == "0" ]]; then
        echo ""; echo "═══ AI Tune: cached result found ═══"
        python3 -c "import json; d=json.load(open('$tune_cache')); b=d.get('best_config',{}); print(f'  Winner: {b.get(\"name\", \"?\")}')"; true 2>/dev/null
        python3 -c "import json; d=json.load(open('$tune_cache')); b=d.get('best_config',{}); print(f'  Gen: {b.get(\"gen_tps\", \"?\")} tok/s  PP: {b.get(\"pp_tps\", \"?\")} tok/s'); print(f'  Tuned: {d.get(\"tuned_at\", \"?\")}')" 2>/dev/null
        echo "  Run --ai-tune --retune to re-optimize"; echo ""; return 0
    fi

    # Warn if model is very large relative to system memory
    local total_ram_mb
    total_ram_mb=$(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null) || total_ram_mb=${RAM_AVAIL_MB}
    local system_mem_mb=$(( TOTAL_VRAM_MB + total_ram_mb ))
    local model_pct=$(( TOTAL_SIZE_MB * 100 / system_mem_mb ))
    if (( model_pct > 70 )); then
        echo ""; echo "WARNING: Model uses ${model_pct}% of system memory (${TOTAL_SIZE_MB}MB / ${system_mem_mb}MB)."
        echo "  AI Tune reloads the model each round — this will be very slow (~${AI_TUNE_ROUNDS}x load time)"
        echo "  and may trigger OOM on memory-constrained systems."
        echo ""; read -r -t 20 -p "Continue with AI Tune? [y/N] " answer < /dev/tty || answer="n"
        if [[ "${answer,,}" != "y" ]]; then echo "AI Tune cancelled."; return 0; fi
    fi

    local _ai_moe=0 _ai_multi_gpu=0
    (( EXPERT_COUNT > 0 )) && _ai_moe=1; (( GPU_COUNT >= 2 )) && _ai_multi_gpu=1
    local _ai_scope_label="perf flags only (placement owned by Phase 1 / iterative optimization)"
    if (( ! _ai_moe && _ai_multi_gpu )); then
        _ai_scope_label="perf + placement-routing (--tensor-split, --split-mode unlocked for dense multi-GPU)"
    fi

    echo ""; echo "═══════════════════════════════════════════════════"
    echo "  AI Tune — iterative LLM-driven optimization"
    echo "  Model: ${MODEL_NAME}"; echo "  Rounds: ${AI_TUNE_ROUNDS}"; echo "  Scope: ${_ai_scope_label}"
    if (( ! VERBOSE )); then echo "  (quiet mode — run with -v / --verbose to see full logs)"; echo "  Server logs: $SERVER_LOG"; fi
    echo ""

    # Quiet mode setup
    if (( ! VERBOSE )); then
        exec 7>&1 8>&2
        trap '{ exec 1>&7 2>&8; exec 7>&- 8>&-; } 2>/dev/null; trap - RETURN' RETURN
        exec >> "$SERVER_LOG" 2>&1
    fi
    local tune_launch_mode="tee"
    (( ! VERBOSE )) && tune_launch_mode="log_only"

    local hw_hash
    hw_hash=$(echo "${GPU_COUNT}_$(for i in $(seq 0 $((GPU_COUNT-1))); do echo -n "${GPU_NAME[$i]}_"; done)" | md5sum | cut -c1-8)
    local tune_history; tune_history=$(load_tune_history "$hw_hash")
    local help_text; help_text=$("$LLAMA_SERVER" --help 2>&1 || true)
    local hw_profile; hw_profile=$(build_hw_profile)
    local model_profile; model_profile=$(build_model_profile)
    local fixed_flags; fixed_flags=$(build_fixed_flags)

    local tune_msg_dir; tune_msg_dir=$(mktemp -d /tmp/llm-tune-msgs.XXXXXX)
    local baseline_flags_desc=""
    for f in "${COMMON_FLAGS[@]}"; do baseline_flags_desc+="$f "; done

    # ── Round 0: Benchmark baseline ──
    local tune_start_ts; tune_start_ts=$(date +%s)
    echo "Round 0/${AI_TUNE_ROUNDS}: Benchmarking baseline (heuristic config)..."
    kill_server ""
    if ! TRY_START_MODE="$tune_launch_mode" try_start; then
        echo "ERROR: Baseline config failed to start"; rm -rf "$tune_msg_dir"; return 1
    fi
    local bench_result baseline_gen baseline_pp
    bench_result=$(bench_gen_tps)
    baseline_gen=$(echo "$bench_result" | awk '{print $1}' || true)
    baseline_pp=$(echo "$bench_result" | awk '{print $2}' || true)
    echo "  Baseline: gen=${baseline_gen} tok/s  pp=${baseline_pp} tok/s"
    _tune_progress_line 0 "$AI_TUNE_ROUNDS" "$tune_start_ts" "$baseline_gen" "baseline" "$baseline_gen"

    if [[ "$baseline_gen" == "0" || "$baseline_gen" == "0.00" || -z "$baseline_gen" ]]; then
        echo "ERROR: Could not measure baseline"; kill_server "$RUNNING_PID"; rm -rf "$tune_msg_dir"; return 1
    fi

    append_tune_history 0 "$baseline_gen" "$baseline_pp" "ok" "{}" "baseline"

    # Build system prompt
    local system_prompt unlimited_section=""
    system_prompt=$(cat <<'SYSEOF'
You are an expert performance tuner for llama.cpp inference servers. Your goal: maximize GENERATION tok/s while preserving output quality. Speed is the primary metric.

# How this works
- I give you hardware info, model info, the full --help output, and the current config + benchmark results
- You propose ONE config per round
- I benchmark it and tell you the results (or report a crash)
- You learn from each result and propose a better config next round
- We do 8 rounds. Make each one count — never repeat a failed or worse config.

# Optimization priority (in order)
1. GPU split strategy: single GPU > graph split > row split > layer split
2. Flash attention: enable it — faster and unlocks KV quantization
3. KV cache quantization: q8_0 is the sweet spot (2x VRAM savings, negligible quality loss). q4_0 saves 4x VRAM but degrades long-context quality — only use if VRAM-starved.
4. Batch size tuning: 2048-4096 is usually optimal. Above 4096 = diminishing returns.
5. Thread count: generation threads = physical CPU cores. Hyperthreads don't help.
6. Memory flags: --no-mmap when model spans GPU + RAM. --mlock pins RAM (less swapping).

# Quality guardrails
- Prefer q8_0 KV over q4_0 — the speed difference is small but quality difference is real
- Quantized KV cache REQUIRES --flash-attn — crashes without it
- Don't reduce context size to gain speed — it cripples usability
- --parallel 1 unless multi-user is explicitly needed (extra slots waste VRAM)

# Crash recovery
- If a config crashed, your next proposal must be MORE conservative on the likely cause
- OOM = reduce batch, increase KV quantization, or spread across more GPUs
- Immediate crash = you used an unsupported or invalid flag — remove it
- Never repeat a config that crashed. Build on what worked instead.
{unlimited_section}
# Rules
1. The baseline config flags are ALREADY applied. You only propose CHANGES.
2. Your "flags" JSON should contain ONLY flags you want to ADD or OVERRIDE.
   Example: {"--batch-size": "4096", "--flash-attn": "on"} — these replace the baseline values.
3. Boolean flags: {"--no-mmap": true, "--mlock": true}
4. To remove a baseline flag: {"--some-flag": false}
5. These flags are FIXED (never include them): {fixed_flags}
6. Your response MUST contain a JSON object with these keys:
   {"name": "short description", "flags": {...}, "reasoning": "why this should be faster"}
7. Think carefully about VRAM budget — total VRAM minus model size = room for KV + batch buffers
8. ONLY use flags from the --help output above. Unknown flags crash the server.
SYSEOF
)
    system_prompt="${system_prompt//\{fixed_flags\}/$fixed_flags}"
    if (( ! _ai_moe && _ai_multi_gpu )); then
        unlimited_section=$'\n# Placement-routing unlocked (dense multi-GPU)\nThis run is dense multi-GPU, so the following are tunable in addition to perf flags:\n- --tensor-split R0,R1,R2,... (ratios across GPUs; baseline is bandwidth-weighted; try small skews first)\n- --split-mode (graph | row | layer) — test alternatives on multi-GPU\n- --device / --main-gpu / -ngl when no user GPU filter was provided\nFlags the user explicitly chose are rejected even if you propose them. Start small (one perturbation per round).\n'
    fi
    system_prompt="${system_prompt//\{unlimited_section\}/$unlimited_section}"
    if (( IS_WSL2 )); then
        local wsl2_note=$'\n# WSL2 environment — extra caution required\n- NEVER propose --mlock (causes host freeze; script will strip it if you try).\n- Host OOM killer is unreliable — prefer conservative batch/parallel values.\n- Headroom is tighter: stay at least 3GB below the system total.\n'
        system_prompt="${system_prompt}${wsl2_note}"
    fi

    # Build initial context
    local initial_context
    initial_context=$(cat <<CTXEOF
# Hardware
${hw_profile}
# Model
${model_profile}
# Server --help output (FULL — every available flag)
${help_text}
# Current baseline config
Flags: ${baseline_flags_desc}
Benchmark: gen=${baseline_gen} tok/s, pp=${baseline_pp} tok/s
# VRAM/RAM safety budget
- Model size: ${TOTAL_SIZE_MB}MB | KV cache: ${KV_TOTAL_MB:-0}MB (${KV_QUALITY:-?})
- Total VRAM: ${TOTAL_VRAM_MB:-0}MB | Total RAM: ${RAM_AVAIL_MB:-0}MB
- KV cache sizes at ctx=${CTX_SIZE}: q4_0≈${KV_TOTAL_MB:-0}MB, q8_0≈$((KV_TOTAL_MB * 2))MB, f16≈$((KV_TOTAL_MB * 4))MB
- CRITICAL: If model+KV+batch already uses >90% of VRAM+RAM, do NOT increase KV cache size
- Increasing batch size also uses more VRAM. Stay within budget.
- If baseline uses q4_0 KV, it's likely because VRAM is tight — respect that choice unless headroom exists.
# Performance + quality guide
- Single GPU > Multi GPU (eliminates inter-GPU transfer overhead)
- graph split > row split > layer split (when multi-GPU is needed)
- --flash-attn: almost always faster, required for quantized KV cache
- KV cache q8_0: best speed/quality tradeoff (2x VRAM savings vs f16, negligible quality loss)
- KV cache q4_0: 4x VRAM savings vs f16, slight quality loss on long contexts — use when VRAM is tight
- batch 2048-4096 usually optimal, above 4096 = diminishing returns
- threads for gen = physical CPU cores, hyperthreads don't help
- --no-mmap when model is split across GPU + system RAM
- Quantized KV (q4_0/q8_0) REQUIRES --flash-attn — crashes without it
- ONLY use flags from the --help output above. Unknown flags crash the server.
- Check the "backend" field in Model info to know which server binary you're tuning.
# Previous tuning history (learn from past results!)
${tune_history}
Propose your first config to try. Think about what might beat the baseline. Use the history above to avoid repeating mistakes and build on what worked.
CTXEOF
)

    echo "$system_prompt" > "$tune_msg_dir/system.txt"
    echo "$initial_context" > "$tune_msg_dir/context.txt"
    local messages
    messages=$(python3 -c "
import json
msgs = [
    {'role': 'system', 'content': open('${tune_msg_dir}/system.txt').read()},
    {'role': 'user', 'content': open('${tune_msg_dir}/context.txt').read()}
]
print(json.dumps(msgs))
")

    local all_results="[]"
    local best_gen="$baseline_gen"
    local best_pp="$baseline_pp"
    local best_name="baseline"
    local best_flags_json="{}"
    local best_round=0
    local prev_flags_json="{}"
    local prev_gen_tps="$baseline_gen"
    local prev_pp_tps="$baseline_pp"
    local round=0 crash_retries=0
    local consecutive_no_improve=0

    # ── Iterative tuning rounds ──
    while (( round < AI_TUNE_ROUNDS )); do
        (( ++round ))
        echo ""; echo "Round ${round}/${AI_TUNE_ROUNDS}: Querying model for config proposal..."
        local llm_response
        llm_response=$(query_llm_chat "$messages")
        if [[ -z "$llm_response" || "$llm_response" == "ERROR" ]]; then echo "  WARNING: No response from model, skipping round"; continue; fi

        # Parse config from LLM response
        local parsed
        parsed=$(python3 - "$llm_response" <<'PY'
import json, re, sys
text = sys.argv[1].strip() if len(sys.argv) > 1 else ''
def try_obj(c):
    try: return json.loads(c) if isinstance(json.loads(c), dict) else None
    except: return None
def emit(o): print(json.dumps(o)); sys.exit(0)
if not text: print('PARSE_ERROR'); sys.exit(0)
obj = try_obj(text)
if obj and 'flags' in obj: emit(obj)
for block in re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE):
    obj = try_obj(block.strip())
    if obj and 'flags' in obj: emit(obj)
decoder = json.JSONDecoder()
for start, ch in enumerate(text):
    if ch != '{': continue
    try: obj, end = decoder.raw_decode(text[start:]); break
    except: continue
    if isinstance(obj, dict) and 'flags' in obj: emit(obj)
for block in re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE):
    try: obj = try_obj(block.strip()); obj and emit(obj)
    except: pass
for start, ch in enumerate(text):
    if ch != '{': continue
    try: obj, end = decoder.raw_decode(text[start:])
    except: continue
    if isinstance(obj, dict): emit(obj)
print('PARSE_ERROR')
PY
)

        if [[ "$parsed" == "PARSE_ERROR" || -z "$parsed" ]]; then
            echo "  WARNING: Could not parse config from response, skipping"
            echo "$llm_response" > "$tune_msg_dir/resp_${round}.txt"
            echo "  Raw response saved to: $tune_msg_dir/resp_${round}.txt"
            messages=$(python3 -c "
import json, sys
msgs = json.loads(sys.stdin.read())
msgs.append({'role': 'assistant', 'content': open('${tune_msg_dir}/resp_${round}.txt').read()})
msgs.append({'role': 'user', 'content': 'ERROR: Could not parse your response as JSON. Please respond with a JSON object containing \"name\", \"flags\" (object), and \"reasoning\" keys.'})
print(json.dumps(msgs))
" <<< "$messages")
            continue
        fi

        local config_name config_reasoning config_flags_json
        config_name=$(echo "$parsed" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name','config_${round}'))" 2>/dev/null || echo "config_${round}")
        config_reasoning=$(echo "$parsed" | python3 -c "import sys,json; print(json.load(sys.stdin).get('reasoning',''))" 2>/dev/null || echo "")
        config_flags_json=$(echo "$parsed" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin).get('flags',{})))" 2>/dev/null || echo "{}")

        echo "  Config: ${config_name}"; echo "  Reason: ${config_reasoning}"
        parse_tune_overrides "$config_flags_json"
        echo "  Overrides: ${TUNE_OVERRIDE_FLAGS[*]:-none}"

        echo "$llm_response" > "$tune_msg_dir/resp_${round}.txt"
        messages=$(python3 -c "
import json, sys
msgs = json.loads(sys.stdin.read())
msgs.append({'role': 'assistant', 'content': open('${tune_msg_dir}/resp_${round}.txt').read()})
print(json.dumps(msgs))
" <<< "$messages")

        # RAM/VRAM safety check
        local est_kv_mb=${KV_TOTAL_MB:-0}
        local proposed_kv_k proposed_batch proposed_ubatch proposed_parallel
        proposed_kv_k=$(echo "$config_flags_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('--cache-type-k',''))" 2>/dev/null) || true
        proposed_batch=$(echo "$config_flags_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('-b', d.get('--batch-size','')))" 2>/dev/null) || true
        proposed_parallel=$(echo "$config_flags_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('--parallel', d.get('-np','')))" 2>/dev/null) || true
        local kv_mult=1
        case "${proposed_kv_k:-${KV_QUALITY:-q4_0}}" in f16) kv_mult=4 ;; q8_0) kv_mult=2 ;; q4_0) kv_mult=1 ;; esac
        est_kv_mb=$(( KV_TOTAL_MB * kv_mult ))
        local parallel_slots=${proposed_parallel:-${PARALLEL_SLOTS:-4}}
        [[ ! "$parallel_slots" =~ ^[0-9]+$ ]] && parallel_slots=4; (( parallel_slots < 1 )) && parallel_slots=1
        est_kv_mb=$(( est_kv_mb * parallel_slots ))
        local batch_sz=${proposed_batch:-${BATCH:-2048}}
        [[ ! "$batch_sz" =~ ^[0-9]+$ ]] && batch_sz=2048
        local est_compute_mb=$(( batch_sz / 2 )); (( est_compute_mb < 512 )) && est_compute_mb=512
        local est_total_mb=$(( TOTAL_SIZE_MB + est_kv_mb + est_compute_mb ))
        local total_ram_mb system_total_mb
        total_ram_mb=$(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null) || total_ram_mb=${RAM_AVAIL_MB}
        system_total_mb=$(( TOTAL_VRAM_MB + total_ram_mb - 5120 ))
        (( IS_WSL2 )) && system_total_mb=$(( system_total_mb - 3072 ))

        if (( est_total_mb > system_total_mb )); then
            echo "  SKIPPED — estimated ${est_total_mb}MB > system capacity ${system_total_mb}MB (OOM protection)"
            echo "Result for '${config_name}': SKIPPED — needs ~${est_total_mb}MB but system has ${system_total_mb}MB. Do NOT increase KV cache, batch size, or parallel. Try other flags." > "$tune_msg_dir/feedback_${round}.txt"
            messages=$(python3 -c "
import json, sys
msgs = json.loads(sys.stdin.read())
msgs.append({'role': 'user', 'content': open('${tune_msg_dir}/feedback_${round}.txt').read()})
print(json.dumps(msgs))
" <<< "$messages")
            _tune_progress_line "$round" "$AI_TUNE_ROUNDS" "$tune_start_ts" "$best_gen" "$best_name" "$baseline_gen"
            continue
        fi

        kill_server "$RUNNING_PID"; sleep 1
        local round_gen="0" round_pp="0" round_status="ok"

        if ! TRY_START_MODE="$tune_launch_mode" try_start_with_overrides; then
            echo "  CRASHED — config failed to start"; kill_server ""; round_status="crashed"
            (( ++crash_retries ))
            if (( crash_retries <= AI_TUNE_MAX_CRASHES )); then
                echo "  (free retry ${crash_retries}/${AI_TUNE_MAX_CRASHES} — crash doesn't count as a round)"
                (( round-- ))
            fi
            echo "  Restarting baseline for next query..."
            TRY_START_MODE="$tune_launch_mode" try_start || { echo "ERROR: Can't restart baseline"; break; }
        else
            echo "  Benchmarking..."
            bench_result=$(bench_gen_tps)
            round_gen=$(echo "$bench_result" | awk '{print $1}' || true)
            round_pp=$(echo "$bench_result" | awk '{print $2}' || true)
            local detail_block
            detail_block=$(printf '  Δ Config Changes:\n\r')
            detail_block+=$(diff_flags "$prev_flags_json" "$config_flags_json" "$prev_gen_tps" "$prev_pp_tps")
            detail_block+=$(printf '\n\r')
            detail_block+=$(show_comparison "$round_gen" "$round_pp" "$prev_gen_tps" "$prev_pp_tps" "$best_gen" "$best_pp" "$best_name" "$best_round")
            if python3 -c "exit(0 if ${round_gen} > ${best_gen} else 1)"; then
                detail_block+=$(printf '  ★ NEW BEST: gen=%s tok/s (%s%%)  pp=%s tok/s\n' "$round_gen" "$improvement" "$round_pp")
                best_gen="$round_gen"; best_pp="$round_pp"; best_name="$config_name"
                best_flags_json="$config_flags_json"; best_round=$round
            else
                detail_block+=$(printf '  Result: gen=%s tok/s (%s%%)  pp=%s tok/s\n' "$round_gen" "$improvement" "$round_pp")
            fi
            printf '%s\n\r' "$detail_block"
            { printf '%s\n\r' "$detail_block" > /dev/tty; } 2>/dev/null || true
            kill_server "$RUNNING_PID"; sleep 1
            echo "  Restarting baseline for next query..."
            TRY_START_MODE="$tune_launch_mode" try_start || { echo "ERROR: Can't restart baseline after benchmark"; break; }
        fi

        # Record result
        echo "$config_name" > "$tune_msg_dir/name_${round}.txt"
        all_results=$(CONFIG_FLAGS_JSON="$config_flags_json" python3 -c "
import json, os, sys
results = json.loads(sys.stdin.read())
try: flags_obj = json.loads(os.environ.get('CONFIG_FLAGS_JSON', '{}'))
except json.JSONDecodeError: flags_obj = {}
results.append({'round': ${round}, 'name': open('${tune_msg_dir}/name_${round}.txt').read().strip(), 'gen_tps': ${round_gen}, 'pp_tps': ${round_pp}, 'flags': flags_obj, 'status': '${round_status}'})
print(json.dumps(results))
" <<< "$all_results")
        append_tune_history "$round" "$round_gen" "$round_pp" "$round_status" "$config_flags_json" "$config_name"
        _tune_progress_line "$round" "$AI_TUNE_ROUNDS" "$tune_start_ts" "$best_gen" "$best_name" "$baseline_gen" \
            "$config_name" "$round_gen" "$round_pp" "$prev_gen_tps" "$prev_pp_tps" "$prev_flags_json" "$config_flags_json"

        if [[ "$round_status" == "ok" ]]; then
            prev_flags_json="$config_flags_json"; prev_gen_tps="$round_gen"; prev_pp_tps="$round_pp"
        fi

        # Convergence check
        if (( DO_CONVERGE )); then
            if python3 -c "exit(0 if ${round_gen:-0} <= ${best_gen} else 1)" 2>/dev/null; then
                (( ++consecutive_no_improve )); echo "  No improvement (consecutive: ${consecutive_no_improve}/2)"
                if (( consecutive_no_improve >= 2 )); then
                    echo ""; echo "  ★ Convergence reached — no improvement for 2 rounds. Stopping."
                    echo "  Best: ${best_name} @ ${best_gen} tok/s"; break
                fi
            else
                consecutive_no_improve=0
            fi
        fi

        # Feed result back to LLM
        local feedback
        if [[ "$round_status" == "crashed" ]]; then
            feedback="Result for '${config_name}': CRASHED (server failed to start — likely OOM or invalid flags). Current best: gen=${best_gen} tok/s. Baseline: gen=${baseline_gen} tok/s. IMPORTANT: Do NOT increase memory usage. Try changing non-memory flags instead (threads, split mode, micro-batch)."
        else
            feedback="Result for '${config_name}': gen=${round_gen} tok/s, pp=${round_pp} tok/s (baseline: gen=${baseline_gen}, best so far: gen=${best_gen}). Propose your next config."
        fi
        echo "$feedback" > "$tune_msg_dir/feedback_${round}.txt"
        messages=$(python3 -c "
import json, sys
msgs = json.loads(sys.stdin.read())
msgs.append({'role': 'user', 'content': open('${tune_msg_dir}/feedback_${round}.txt').read()})
print(json.dumps(msgs))
" <<< "$messages")
    done

    # ── Save results ──
    kill_server "$RUNNING_PID"; kill_server ""
    local tune_total_elapsed=$(( $(date +%s) - tune_start_ts ))

    if (( ! VERBOSE )); then
        exec 1>&7 2>&8; exec 7>&- 8>&-
    fi

    echo ""; echo "═══════════════════════════════════════════════════"
    if [[ "$best_name" == "baseline" ]]; then
        echo "  AI Tune complete: baseline wins"
        echo "  The heuristic config is already optimal (${baseline_gen} tok/s)"
        echo "  Total time: $(_tune_fmt_duration "$tune_total_elapsed")"
        echo "═══════════════════════════════════════════════════"
        python3 -c "
import json; from datetime import datetime, timezone
cache = {
    'model': '${MODEL_NAME}', 'tuned_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'provider': 'self', 'baseline_gen_tps': ${baseline_gen}, 'baseline_pp_tps': ${baseline_pp},
    'baseline_wins': True,
    'best_config': {'name': 'baseline', 'flags': {}, 'gen_tps': ${baseline_gen}, 'pp_tps': ${baseline_pp}},
    'rounds': ${AI_TUNE_ROUNDS}, 'all_results': json.loads(open('/dev/stdin').read())
}
with open('${tune_cache}', 'w') as f: json.dump(cache, f, indent=2)
print(f'  Saved marker to: ${tune_cache}')
" <<< "$all_results"
    else
        local total_improvement
        total_improvement=$(python3 -c "g=${baseline_gen}; print(f'{(${best_gen}-g)/g*100:.2f}' if g>0 else '0')")
        echo "  AI Tune complete: ${best_name} wins!"
        echo "  Baseline: ${baseline_gen} tok/s → Best: ${best_gen} tok/s (+${total_improvement}%)"
        echo "  Total time: $(_tune_fmt_duration "$tune_total_elapsed")"
        echo "═══════════════════════════════════════════════════"
        echo "$best_name" > "$tune_msg_dir/best_name.txt"
        BEST_FLAGS_JSON="$best_flags_json" python3 -c "
import json, os; from datetime import datetime, timezone
try: best_flags = json.loads(os.environ.get('BEST_FLAGS_JSON', '{}'))
except json.JSONDecodeError: best_flags = {}
cache = {
    'model': '${MODEL_NAME}', 'tuned_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'provider': 'self', 'baseline_gen_tps': ${baseline_gen}, 'baseline_pp_tps': ${baseline_pp},
    'best_config': {'name': open('${tune_msg_dir}/best_name.txt').read().strip(), 'flags': best_flags, 'gen_tps': ${best_gen}, 'pp_tps': ${best_pp}},
    'rounds': ${AI_TUNE_ROUNDS}, 'all_results': json.loads(open('/dev/stdin').read())
}
with open('${tune_cache}', 'w') as f: json.dump(cache, f, indent=2)
print(f'  Saved to: ${tune_cache}')
" <<< "$all_results"
    fi
    rm -rf "$tune_msg_dir"
    echo ""; return 0
}

# ── Load tuned cache ───────────────────────────────────────────
load_tune_cache() {
    mkdir -p "$CACHE_DIR"
    local vision_suffix=""
    [[ -n "$MMPROJ_PATH" ]] && vision_suffix="_v"
    local tune_cache_key="${MODEL_NAME}_${TOTAL_SIZE_BYTES}_hw$(echo "${GPU_COUNT}_$(for i in $(seq 0 $((GPU_COUNT-1))); do echo -n "${GPU_NAME[$i]}_"; done)" | md5sum | cut -c1-8)${vision_suffix}_${BACKEND_TAG}"
    local tune_cache_suffix=""
    (( DO_UNLIMITED )) && tune_cache_suffix="_unlimited"
    local tune_cache="$CACHE_DIR/tune_${tune_cache_key}${tune_cache_suffix}.json"
    local explicit_cache=0
    if [[ -n "$EXPLICIT_TUNE_CACHE" ]]; then
        explicit_cache=1; tune_cache="$EXPLICIT_TUNE_CACHE"
        if [[ ! -f "$tune_cache" ]]; then echo "Error: --tune-cache not found: $tune_cache" >&2; exit 1; fi
    elif (( DO_UNLIMITED )) && [[ ! -f "$tune_cache" && -f "$CACHE_DIR/tune_${tune_cache_key}.json" ]]; then
        tune_cache="$CACHE_DIR/tune_${tune_cache_key}.json"
    fi
    if [[ -f "$tune_cache" ]]; then
        echo ""; (( explicit_cache )) && echo "Using selected AI-tuned config: $(basename "$tune_cache")" || echo "Using AI-tuned config (run --ai-tune --retune to re-optimize)"
        if (( DO_AI_TUNE )); then return 1; fi  # Let ai_tune() handle it
        return 0
    fi
    return 1
}

maybe_ai_tune_launch_flags() {
    (( DO_AI_TUNE )) || return 1
    COMMON_FLAGS=("$@")
    TUNE_EXTRA_FIXED=()
    ai_tune
    return 0
}

maybe_prompt_first_run_ai_tune() {
    (( DO_AI_TUNE || DRY_RUN || BENCHMARK || RETUNE )) && return 0
    (( TUNE_LOADED )) && return 0
    [[ -n "$EXPLICIT_TUNE_CACHE" ]] && return 0
    [[ -t 0 && -r /dev/tty ]] || return 0
    (( ${LLM_ASSUME_YES:-0} == 0 )) || return 0
    local answer
    echo ""; echo "No AI-tuned config found for this model/hardware/backend."
    echo "AI Tune can create a better config now by loading the model and benchmarking several rounds."
    echo "This usually takes a few minutes; large models or full placement mode can take longer."
    read -r -t 20 -p "Run AI Tune before launching? [y/N] " answer < /dev/tty || answer="n"
    case "${answer,,}" in y|yes) DO_AI_TUNE=1; echo "AI Tune enabled for this launch." ;; *) echo "Continuing with heuristic config." ;; esac
}

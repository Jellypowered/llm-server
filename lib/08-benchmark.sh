#!/bin/bash
# lib/08-benchmark.sh — run_benchmark and run_two_phase_benchmark.

run_benchmark() {
    local label="$1"; shift
    local url="http://127.0.0.1:${PORT}"
    local bench_log="$BENCHMARK_LOG"
    exec 3>&1 4>&2
    exec >> "$bench_log" 2>&1
    echo ""; echo "═══ Benchmark: ${label} ═══"
    echo "  Flags: $*"
    local pp_prompt="Explain the theory of relativity in simple terms. Cover special and general relativity, time dilation, and gravitational effects."
    local result
    result=$(curl -sf "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$pp_prompt\"}],\"max_tokens\":200,\"temperature\":0.1}" 2>/dev/null)
    exec 1>&3 2>&4 3>&- 4>&-
    if [[ -z "$result" ]]; then echo "Benchmark failed — server not responding" >&2; return 1; fi
    local pp_tokens tg_tokens response_rates pp_rate tg_rate
    pp_tokens=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); print(u.get('prompt_tokens',0))" 2>/dev/null || echo 0)
    tg_tokens=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); print(u.get('completion_tokens',0))" 2>/dev/null || echo 0)
    response_rates=$(echo "$result" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin); t=d.get('timings', {}); pp=t.get('prompt_per_second') or 0; tg=t.get('predicted_per_second') or 0
    print(f'{pp:.1f}\t{tg:.1f}' if pp and tg else '')
except: print('')
" 2>/dev/null || true)
    if [[ -n "$response_rates" ]]; then
        pp_rate="${response_rates%%$'\t'*}"; tg_rate="${response_rates#*$'\t'}"
        echo "  Prompt processing: ${pp_tokens:-?} tokens @ ${pp_rate:-?} tok/s" >&2
        echo "  Generation:        ${tg_tokens:-?} tokens @ ${tg_rate:-?} tok/s" >&2
    else
        local timings
        timings=$(curl -sf "$url/slots" 2>/dev/null)
        if [[ -n "$timings" ]]; then
            pp_rate=$(echo "$timings" | python3 -c "
import sys,json
d=json.load(sys.stdin); s=d[0] if isinstance(d,list) else d
t_pp = s.get('t_prompt_processing', 0); n_pp = s.get('n_prompt_tokens_processed', 1)
if t_pp > 0: print(f\"{n_pp / (t_pp / 1000):.1f}\")
else: print('?')
" 2>/dev/null)
            tg_rate=$(echo "$timings" | python3 -c "
import sys,json
d=json.load(sys.stdin); s=d[0] if isinstance(d,list) else d
t_gen = s.get('t_token_generation', 0); n_gen = s.get('n_decoded', 1)
if t_gen > 0: print(f\"{n_gen / (t_gen / 1000):.1f}\")
else: print('?')
" 2>/dev/null)
            echo "  Prompt processing: ${pp_tokens:-?} tokens @ ${pp_rate:-?} tok/s" >&2
            echo "  Generation:        ${tg_tokens:-?} tokens @ ${tg_rate:-?} tok/s" >&2
        else
            echo "  Prompt tokens: ${pp_tokens:-?}" >&2
            echo "  Generated tokens: ${tg_tokens:-?}" >&2
        fi
    fi
    { echo ""; echo "═══ Benchmark: ${label} ═══"; echo "  Flags: $*"
      if [[ -n "$tg_rate" && -n "$pp_rate" ]]; then
          echo "  Prompt processing: ${pp_tokens:-?} tokens @ ${pp_rate:-?} tok/s"
          echo "  Generation:        ${tg_tokens:-?} tokens @ ${tg_rate:-?} tok/s"
      fi
    } >> "$bench_log" 2>&1
    echo "${tg_rate:-0} ${pp_rate:-0}"
}

run_two_phase_benchmark() {
    local p1_gen="" p1_pp="" p2_gen="" p2_pp="" p3_gen="" p3_pp=""
    echo ""; echo "═══ Phase 1: True Baseline (llama.cpp defaults) ═══"
    echo "  Running with BASE_FLAGS only — no optimizations applied."
    kill_server ""
    if ! TRY_START_MODE=log_only try_start "${BASE_FLAGS[@]}"; then
        echo "  Phase 1 failed — server could not start with base flags"
    else
        local result; result=$(run_benchmark "True Baseline" "${BASE_FLAGS[@]}")
        p1_gen=$(echo "$result" | awk '{print $1}'); p1_pp=$(echo "$result" | awk '{print $2}')
        kill_server "$RUNNING_PID"
    fi
    echo ""; echo "═══ Phase 2: Heuristic Config (llm-server optimized) ═══"
    echo "  Running with COMMON_FLAGS — GPU placement, flash-attn, KV quantization, etc."
    if ! TRY_START_MODE=log_only try_start; then
        echo "  Phase 2 failed — server could not start with heuristic flags"
    else
        local result; result=$(run_benchmark "Heuristic Config" "${COMMON_FLAGS[@]}")
        p2_gen=$(echo "$result" | awk '{print $1}'); p2_pp=$(echo "$result" | awk '{print $2}')
        kill_server "$RUNNING_PID"
    fi
    if [[ -n "$SPEC_TYPE" && "$SPEC_TYPE" != "none" ]]; then
        echo ""; echo "═══ Phase 3: Heuristic Config + Speculative Decoding ═══"
        local spec_flags=("${COMMON_FLAGS[@]}")
        spec_flags+=(--spec-type "$SPEC_TYPE")
        [[ -n "$SPEC_DRAFT_N_MAX" ]] && spec_flags+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
        if ! TRY_START_MODE=log_only try_start "${spec_flags[@]}"; then
            echo "  Phase 3 failed — server could not start with speculative decoding flags"
        else
            local result; result=$(run_benchmark "Heuristic + Spec ($SPEC_TYPE)" "${spec_flags[@]}")
            p3_gen=$(echo "$result" | awk '{print $1}'); p3_pp=$(echo "$result" | awk '{print $2}')
            kill_server "$RUNNING_PID"
        fi
    fi
    echo ""; echo "═══ Benchmark Summary ═══"
    if [[ -n "$p1_gen" ]]; then
        echo "  Phase 1 (Baseline):       gen=${p1_gen} tok/s  pp=${p1_pp} tok/s"
    fi
    if [[ -n "$p2_gen" ]]; then
        local p2_gain; p2_gain=$(python3 -c "g=${p1_gen:-0}; print(f'{(${p2_gen:-0}-g)/g*100:.2f}%' if g>0 else 'N/A')" 2>/dev/null || echo "N/A")
        echo "  Phase 2 (Heuristic):      gen=${p2_gen} tok/s  pp=${p2_pp} tok/s  (${p2_gain} vs baseline)"
    fi
    if [[ -n "$p3_gen" ]]; then
        local p3_gain; p3_gain=$(python3 -c "g=${p1_gen:-0}; print(f'{(${p3_gen:-0}-g)/g*100:.2f}%' if g>0 else 'N/A')" 2>/dev/null || echo "N/A")
        echo "  Phase 3 (Heuristic+Spec): gen=${p3_gen} tok/s  pp=${p3_pp} tok/s  (${p3_gain} vs baseline)"
    fi
    echo "  Detailed logs: $BENCHMARK_LOG"
    echo ""
}

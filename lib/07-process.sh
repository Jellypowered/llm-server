#!/bin/bash
# lib/07-process.sh — Process management: kill_server, wait_for_ready,
# launch_backend, run_with_restart, try_start, and related helpers.

# ── Build -ot string from (cuda_device, layer_start, layer_count) pairs ─
build_ot_string() {
    local parts=()
    while [[ $# -ge 3 ]]; do
        local cuda_idx=$1 layer_start=$2 layer_count=$3
        shift 3
        if (( layer_count > 0 )); then
            local last=$(( layer_start + layer_count - 1 ))
            local re=$(seq "$layer_start" "$last" | tr '\n' '|' | sed 's/|$//')
            parts+=("blk\\.(${re})\\.(ffn_(gate_up|up_gate|gate|up|down)_exps|(gate_inp|gate|up|down)_shexp).*=CUDA${cuda_idx}")
        fi
    done
    parts+=("exps=CPU")
    local IFS=','
    echo "${parts[*]}"
}

# ── Kill server ────────────────────────────────────────────────
kill_server() {
    local pid="${1:-}"
    local own_pgid pgid=""
    own_pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        pgid=$(_pgid_of "$pid" 2>/dev/null || true)
        if [[ -n "$pgid" && "$pgid" != "$own_pgid" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
        local i=0
        while (( i < 5 )) && kill -0 "$pid" 2>/dev/null; do sleep 1; (( ++i )); done
        if kill -0 "$pid" 2>/dev/null; then
            if [[ -n "$pgid" && "$pgid" != "$own_pgid" ]]; then
                kill -KILL -- "-$pgid" 2>/dev/null || true
            else
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        wait "$pid" 2>/dev/null || true
    fi

    # Port-listener cleanup
    local port_pids p cmd
    port_pids=$(_port_listener_pids)
    if [[ -n "$port_pids" ]]; then
        for p in $port_pids; do
            if _is_llama_proc "$p"; then
                kill -TERM "$p" 2>/dev/null || true
            else
                cmd=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ' || true)
                [[ -n "$cmd" ]] && echo "  WARN: port ${PORT} held by foreign process '$cmd' (pid $p) — refusing to kill" >&2
            fi
        done
        sleep 1
        port_pids=$(_port_listener_pids)
        for p in $port_pids; do _is_llama_proc "$p" && kill -9 "$p" 2>/dev/null || true; done
    fi

    local w=0 has_ours
    while (( w < 10 )); do
        has_ours=0
        port_pids=$(_port_listener_pids)
        for p in $port_pids; do _is_llama_proc "$p" && { has_ours=1; break; }; done
        (( has_ours )) || break
        sleep 1; (( ++w ))
    done
    RUNNING_PID=""
}

# ── Wait for server health ─────────────────────────────────────
wait_for_ready() {
    local pid=$1
    local i=0
    while (( i < HEALTH_TIMEOUT )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            local smoke
            smoke=$(curl -sf --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"test","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4,"temperature":0}' 2>/dev/null)
            if [[ -n "$smoke" ]] && echo "$smoke" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices')" 2>/dev/null; then
                RUNNING_PID=$pid; return 0
            fi
            if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        fi
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        sleep 1; (( ++i ))
    done
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
    return 1
}

# ── Launch backend ─────────────────────────────────────────────
# Args: $1 = mode (tee, tee_filter, log_only), rest = server flags
launch_backend() {
    local mode="$1"; shift
    LAUNCH_LOG_OFFSET=$(stat -c %s "$SERVER_LOG" 2>/dev/null || echo 0)
    local launcher=()
    command -v setsid >/dev/null 2>&1 && launcher=(setsid)
    local env_prefix=()
    (( ${#LAUNCH_ENV_PREFIX[@]} )) && env_prefix=(env "${LAUNCH_ENV_PREFIX[@]}")
    case "$mode" in
        tee_filter) "${launcher[@]}" "${env_prefix[@]}" "$LLAMA_SERVER" "$@" \
            > >(tee -a "$SERVER_LOG" | grep -v --line-buffered "record AND rewind is invalid") 2>&1 & ;;
        log_only)   "${launcher[@]}" "${env_prefix[@]}" "$LLAMA_SERVER" "$@" >> "$SERVER_LOG" 2>&1 & ;;
        tee|*)      "${launcher[@]}" "${env_prefix[@]}" "$LLAMA_SERVER" "$@" > >(tee -a "$SERVER_LOG") 2>&1 & ;;
    esac
    local pid=$!
    echo 1000 > "/proc/$pid/oom_score_adj" 2>/dev/null || true
    RUNNING_PID="$pid"
}

# ── Strip ik_llama-only flags ─────────────────────────────────
_strip_ik_only_flags() {
    local -n _arr="$1"
    local out=() skip=0 f
    local i
    for (( i = 0; i < ${#_arr[@]}; i++ )); do
        if (( skip )); then skip=0; continue; fi
        f="${_arr[$i]}"
        case "$f" in
            --run-time-repack|-rtr|--no-context-shift|-khad|-mqkv|--merge-qkv|\
--merge-up-gate-exps|-muge|--grouped-expert-routing|-ger|-cram|\
--cache-ram|--defrag-thold|--ctx-checkpoints)
                echo "  Removing ik-only flag: $f"
                local _next_val="${_arr[$((i+1))]:-}"
                if [[ -n "$_next_val" && "$_next_val" != -* ]]; then skip=1; fi ;;
            --split-mode)
                local next_val="${_arr[$((i+1))]:-}"
                if [[ "$next_val" == "graph" ]]; then echo "  Removing ik-only flag: $f $next_val"; skip=1
                else out+=("$f"); fi ;;
            *) out+=("$f") ;;
        esac
    done
    _arr=("${out[@]}")
}

# ── Try ik_llama → mainline fallback ──────────────────────────
_try_ik_fallback() {
    local arr_name="$1"
    (( ${IS_IK_LLAMA:-0} )) || return 1
    [[ -n "${USER_SERVER_BIN:-}" ]] && return 1
    local mainline_bin="$HOME/llama.cpp/build/bin/llama-server"
    [[ -x "$mainline_bin" ]] || return 1
    local recent_log
    recent_log=$(tail -c +"$((${LAUNCH_LOG_OFFSET:-0}+1))" "$SERVER_LOG" 2>/dev/null || true)
    case "$recent_log" in
        *"unknown model architecture"*|*"unable to load model"*|*"error loading model"*) ;;
        *) return 1 ;;
    esac
    echo ""; echo "═══ ik_llama.cpp cannot load this model — falling back to mainline llama.cpp ═══"
    LLAMA_SERVER="$mainline_bin"; IS_IK_LLAMA=0
    setup_lib_hub "$LLAMA_SERVER"
    _strip_ik_only_flags "$arr_name"
    echo "  Retrying with: $LLAMA_SERVER"
    return 0
}

# ── Parse load failure from log ───────────────────────────────
_parse_load_failure() {
    local recent
    recent=$(tail -c +"$((${LAUNCH_LOG_OFFSET:-0}+1))" "$SERVER_LOG" 2>/dev/null \
        | iconv -f utf-8 -t utf-8 -c 2>/dev/null || true)
    [[ -n "$recent" ]] || { echo "unknown"; return; }

    local oom_line size_mb device
    oom_line=$(printf '%s\n' "$recent" \
        | grep -E "cudaMalloc.*out of memory|vkAllocateMemory.*failed|VK_ERROR_OUT_OF_DEVICE_MEMORY" \
        | tail -1 || true)
    if [[ -n "$oom_line" ]]; then
        size_mb=$(printf '%s' "$oom_line" | grep -oE "allocating [0-9]+\.[0-9]+ MiB" | grep -oE "[0-9]+\.[0-9]+" | awk '{printf "%.0f", $1}' || true)
        device=$(printf '%s' "$oom_line" | grep -oE "on device [0-9]+" | grep -oE "[0-9]+" || true)
        if [[ -n "$size_mb" && -n "$device" ]]; then echo "oom $device $size_mb"; return; fi
    fi

    if printf '%s\n' "$recent" | grep -qE "cudaHostAlloc.*(failed|out of memory)|pinned.*alloc.*(failed|out of memory)"; then
        if printf '%s\n' "$recent" | grep -qiE "RLIMIT_MEMLOCK|locked.memory|ulimit.*memlock|cannot allocate locked|operation not permitted"; then
            echo "pinned_cap_exceeded"; return
        else
            echo "pinned_fail"; return
        fi
    fi

    if printf '%s\n' "$recent" | grep -qE "std::bad_alloc|what\\(\\):.*bad_alloc"; then echo "ram_oom"; return; fi
    if printf '%s\n' "$recent" | grep -qE "[[:space:]][0-9]+[[:space:]]+Killed[[:space:]]|received signal:? 9|signal 9 \(SIGKILL\)|out of memory.*killed|oom-kill"; then echo "ram_oom"; return; fi
    if printf '%s\n' "$recent" | grep -qE "Cannot allocate memory|posix_memalign.*failed|mmap.*Cannot allocate"; then
        if ! printf '%s\n' "$recent" | tail -20 | grep -qE "cudaHostAlloc|CUDA_Host|vkAllocateMemory"; then echo "ram_oom"; return; fi
    fi

    if printf '%s\n' "$recent" | grep -qE "CUDA_Host model buffer size"; then
        if ! printf '%s\n' "$recent" | grep -qE "model loaded|HTTP server listening|llama_init_from_model: graph nodes"; then echo "pinned_hang"; return; fi
    fi

    echo "unknown"
}

_failure_direction() {
    case "$1" in
        oom\ *) echo "more_cpu" ;;
        ram_oom|pinned_fail|pinned_cap_exceeded|pinned_hang) echo "more_gpu" ;;
        *) echo "indeterminate" ;;
    esac
}

_free_vram_for_device() {
    local cuda_idx="$1" gi
    for gi in $(seq 0 $(( GPU_COUNT - 1 ))); do
        if [[ "${GPU_INDEX[$gi]}" == "$cuda_idx" ]]; then echo "${GPU_VRAM_FREE[$gi]}"; return; fi
    done
    echo ""
}

# ── Try start ──────────────────────────────────────────────────
try_start() {
    local extra_flags=("$@")
    RUNNING_PID=""
    local mode="${TRY_START_MODE:-tee}"
    local launch_flags=("${COMMON_FLAGS[@]}" "${extra_flags[@]}")
    launch_backend "$mode" "${launch_flags[@]}"
    local pid="$RUNNING_PID"
    if wait_for_ready "$pid"; then return 0; fi
    if _try_ik_fallback COMMON_FLAGS; then
        _strip_ik_only_flags extra_flags
        kill_server "$pid"
        launch_flags=("${COMMON_FLAGS[@]}" "${extra_flags[@]}")
        launch_backend "$mode" "${launch_flags[@]}"
        pid="$RUNNING_PID"
        wait_for_ready "$pid" && return 0
    fi
    return 1
}

# ── Print command (for --dry-run) ─────────────────────────────
print_cmd() {
    echo ""; echo "Command:"
    local prefix=""
    (( ${#LAUNCH_ENV_PREFIX[@]} )) && prefix="${LAUNCH_ENV_PREFIX[*]} "
    local line="  ${prefix}$LLAMA_SERVER"
    local args=("$@")
    for (( i=0; i<${#args[@]}; i++ )); do
        local arg="${args[$i]}"
        if [[ "$arg" == -* ]] && (( i + 1 < ${#args[@]} )) && [[ "${args[$((i+1))]}" != -* ]]; then
            local pair="$arg ${args[$((i+1))]}"; i=$(( i + 1 ))
        else
            local pair="$arg"
        fi
        if (( ${#line} + ${#pair} + 1 > 80 )); then
            echo "$line \\"; line="    $pair"
        else
            line="$line $pair"
        fi
    done
    echo "$line"
}

# ── Run with restart ──────────────────────────────────────────
run_with_restart() {
    local flags=("$@")
    local restarts=0 last_start=0
    while true; do
        last_start=$(date +%s)
        echo "Starting server..."
        launch_backend "tee_filter" "${flags[@]}"
        local pid="$RUNNING_PID"
        local log_start="${LAUNCH_LOG_OFFSET:-0}"
        echo "Server PID: $pid"
        local healthy=0
        for i in $(seq 1 $HEALTH_TIMEOUT); do
            if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                local smoke
                smoke=$(curl -sf --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
                    -H "Content-Type: application/json" \
                    -d '{"model":"test","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4,"temperature":0}' 2>/dev/null)
                if [[ -n "$smoke" ]] && echo "$smoke" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices')" 2>/dev/null; then
                    healthy=1; break
                fi
                if ! kill -0 "$pid" 2>/dev/null; then
                    if _try_ik_fallback flags; then kill_server "$pid"; restarts=0; continue 2; fi
                    break
                fi
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                if _try_ik_fallback flags; then kill_server "$pid"; restarts=0; continue 2; fi
                break
            fi
            sleep 1
        done
        if (( healthy )); then
            restarts=0
            local cuda_killed=0 log_pos=$log_start
            while kill -0 "$pid" 2>/dev/null; do
                sleep 5
                local new_log
                new_log=$(tail -c +"$log_pos" "$SERVER_LOG" 2>/dev/null) || new_log=""
                if echo "$new_log" | grep -qE "CUDA error|Vulkan error|VK_ERROR_DEVICE_LOST|VK_ERROR_OUT_OF_DEVICE_MEMORY"; then
                    echo ""; echo "═══ GPU error detected — server is braindead, killing ═══"
                    tail -5 "$SERVER_LOG" 2>/dev/null; kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
                    cuda_killed=1; break
                fi
                local img_fails
                img_fails=$(echo "$new_log" | grep -c "failed to process image" 2>/dev/null) || img_fails=0
                if (( img_fails >= 10 )); then
                    echo ""; echo "═══ Image decode loop detected ($img_fails failures) — server is stuck, killing ═══"
                    tail -5 "$SERVER_LOG" 2>/dev/null; kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
                    cuda_killed=1; break
                fi
                log_pos=$(wc -c < "$SERVER_LOG" 2>/dev/null || echo "$log_pos")
            done
            local exit_code=0
            if (( cuda_killed )); then
                echo "Restarting after CUDA error recovery..."
                kill_server ""
                local gpu_wait=0
                while (( gpu_wait < 60 )); do
                    local gpu_procs=0
                    if [[ "$GPU_BACKEND" == "cuda" ]] && command -v nvidia-smi &>/dev/null; then
                        gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]') || gpu_procs=0
                    fi
                    if (( gpu_procs == 0 )); then echo "GPU memory free after ${gpu_wait}s"; break; fi
                    sleep 2; (( gpu_wait += 2 ))
                done
                (( gpu_wait >= 60 )) && echo "Warning: GPU processes still present after 60s, attempting restart anyway"
                sleep 3; restarts=-3; continue
            fi
            wait "$pid" 2>/dev/null && exit_code=$? || exit_code=$?
        else
            echo "Server failed to become healthy."
            kill -9 "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
            local exit_code=1
        fi
        local uptime=$(( $(date +%s) - last_start ))
        if (( exit_code == 0 || exit_code == 143 )); then
            if (( ! healthy )); then echo "Server exited cleanly before becoming healthy. Not restarting."; exit 0
            else echo "Server exited cleanly after running. Restarting..."; restarts=$(( restarts + 1 )); fi
        else
            restarts=$(( restarts + 1 ))
            echo ""; echo "═══ Server crashed (exit code $exit_code, uptime ${uptime}s) ═══"
            tail -5 "$SERVER_LOG" 2>/dev/null; echo ""
        fi
        if (( restarts > MAX_RESTARTS )); then
            if (( KEEP_ALIVE )); then
                echo "Reached $MAX_RESTARTS restarts, but --keep-alive set — continuing."
                echo "Backoff extended to 60s between attempts. Ctrl-C to stop."
                restarts=0; sleep 60; kill_server ""; continue
            fi
            echo "Too many restarts ($MAX_RESTARTS). Giving up."
            echo "Use --keep-alive (or LLM_KEEP_ALIVE=1) to never give up."
            echo "Check $SERVER_LOG for details."
            exit 1
        fi
        local wait_sec=$(( 2 ** restarts ))
        (( wait_sec > 30 )) && wait_sec=30
        if (( uptime > 60 )); then
            wait_sec=2
            echo "Runtime crash (ran ${uptime}s). Restart $restarts/$MAX_RESTARTS in ${wait_sec}s..."
        else
            echo "Startup crash. Restart $restarts/$MAX_RESTARTS in ${wait_sec}s..."
            if _try_ik_fallback flags; then restarts=0; kill_server ""; continue; fi
        fi
        sleep "$wait_sec"; kill_server ""
    done
}


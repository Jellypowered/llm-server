#!/bin/bash
# lib/01-helpers.sh — Utility functions: age formatting, backend status,
# logging, port listeners, repo update checks.

# ── Human-readable age ─────────────────────────────────────────
_human_age() {
    local s=$1
    if (( s < 60 )); then echo "${s}s ago"
    elif (( s < 3600 )); then echo "$(( s/60 ))m ago"
    elif (( s < 86400 )); then echo "$(( s/3600 ))h ago"
    else echo "$(( s/86400 ))d ago"
    fi
}

# ── Backend git status ─────────────────────────────────────────
_backend_status() {
    local dir=$1
    [[ ! -d "$dir/.git" ]] && { echo "missing"; return; }
    local hash age_s age behind upstream
    hash=$(cd "$dir" && git rev-parse --short HEAD 2>/dev/null) || { echo "missing"; return; }
    age_s=$(cd "$dir" && git log -1 --format=%ct HEAD 2>/dev/null)
    age=$(_human_age $(( $(date +%s) - age_s )))
    upstream=$(cd "$dir" && git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)
    if [[ -n "$upstream" ]]; then
        behind=$(cd "$dir" && git rev-list --count "HEAD..$upstream" 2>/dev/null || echo ?)
        if [[ "$behind" == "0" ]]; then echo "$hash · $age · up to date"
        else echo "$hash · $age · $behind behind"; fi
    else
        echo "$hash · $age · no upstream"
    fi
}

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
    case "$f" in
        *_ik.json)      echo "ik_llama" ;;
        *_llama.json)   echo "llama" ;;
        *_vulkan.json)  echo "llama-vk" ;;
        *)              echo "llama" ;;
    esac
}

# ── Logging ────────────────────────────────────────────────────
log() { [[ "$VERBOSE" == "1" ]] && echo "[DEBUG] $*" >&2 || true; }

_is_llama_proc() {
    local p="${1:-}"
    [[ -z "$p" ]] && return 1
    local cmd
    cmd=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ' || true)
    case "$cmd" in
        llama-server|llama-cli|llama|ik_llama-server|ik_llama) return 0 ;;
        *) return 1 ;;
    esac
}

_port_listener_pids() {
    lsof -nP -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

_stop_llama_listeners_for_ai_tune() {
    local port_pids p cmd
    port_pids=$(_port_listener_pids)
    [[ -z "$port_pids" ]] && return 0

    echo "Checking existing listener on port ${PORT} before AI tuning..."
    for p in $port_pids; do
        if _is_llama_proc "$p"; then
            echo "  Stopping llama-compatible listener pid $p"
            kill -TERM "$p" 2>/dev/null || true
        else
            cmd=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ' || true)
            [[ -z "$cmd" ]] && cmd="unknown"
            echo "  WARN: port ${PORT} held by foreign process '$cmd' (pid $p); refusing to kill" >&2
        fi
    done

    sleep 3
    port_pids=$(_port_listener_pids)
    for p in $port_pids; do
        _is_llama_proc "$p" && kill -KILL "$p" 2>/dev/null || true
    done
}

# ── Repo update check ──────────────────────────────────────────
_check_repo_behind() {
    local repo_dir="$1" label="$2"
    [[ -d "$repo_dir/.git" ]] || return 1
    git -C "$repo_dir" remote update --prune >/dev/null 2>&1 &
    local fetch_pid=$!
    ( sleep 5; kill "$fetch_pid" 2>/dev/null ) &
    wait "$fetch_pid" 2>/dev/null || return 1
    local local_head remote_head
    local_head=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null) || return 1
    remote_head=$(git -C "$repo_dir" rev-parse '@{u}' 2>/dev/null) || return 1
    [[ "$local_head" != "$remote_head" ]] && echo "$label" && return 0
    return 1
}

_should_check_updates() {
    if [[ -f "$UPDATE_DISMISS_FILE" ]]; then
        local dismissed_at
        dismissed_at=$(cat "$UPDATE_DISMISS_FILE" 2>/dev/null)
        local now
        now=$(date +%s)
        if (( now - dismissed_at < UPDATE_DISMISS_DAYS * 86400 )); then
            return 1
        fi
    fi
    return 0
}

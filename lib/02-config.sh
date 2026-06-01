#!/bin/bash
# lib/02-config.sh — Config discovery, migration, and settings subcommand.
# Sourced early so config discovery can use _migrate_legacy_config_sh.

# ── Migrate legacy config.sh → config ──────────────────────────
_migrate_legacy_config_sh() {
    local canonical="$1"
    local legacy="${canonical}.sh"
    [[ "$canonical" == *.sh ]] && return 0
    [[ -f "$legacy" ]] || return 0
    if [[ ! -f "$canonical" ]]; then
        mv "$legacy" "$canonical" 2>/dev/null && \
            echo "Note: migrated $legacy -> $canonical" >&2
        return 0
    fi
    local merged tmp
    tmp="$(mktemp "${canonical}.merge.XXXX")" || return 1
    {
        echo "# llm-server settings (merged from config + config.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ))"
        awk -F= '
            /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
            { key = $1; sub(/^[[:space:]]*/, "", key); sub(/[[:space:]]*$/, "", key);
              val = $0; sub(/^[^=]*=/, "", val); kv[key] = val }
            END { for (k in kv) print k "=" kv[k] }
        ' "$canonical" "$legacy"
    } > "$tmp" && mv "$tmp" "$canonical" || { rm -f "$tmp"; return 1; }
    local backup="$legacy.bak.$(date +%s)"
    mv "$legacy" "$backup" 2>/dev/null && \
        echo "Note: merged $legacy into $canonical (legacy archived at $backup)" >&2
}

# ── Settings subcommand ────────────────────────────────────────
_settings_file_path() {
    if [[ -n "${LLM_CONFIG:-}" ]]; then
        echo "$LLM_CONFIG"
    elif [[ -n "${LLM_APP_HOME:-}" && -d "$LLM_APP_HOME/config" ]]; then
        echo "$LLM_APP_HOME/config/config.sh"
    else
        echo "${XDG_CONFIG_HOME:-$HOME/.config}/llm-server/config"
    fi
}

_settings_write_template() {
    local file="$1"
    mkdir -p "$(dirname "$file")"
    cat > "$file" <<'TEMPLATE'
# llm-server settings
#
# This file is sourced as bash. Each setting below is commented out with
# its built-in default. Uncomment a line and change the value to override.
#
# Precedence at runtime (highest wins):
#   1. CLI flag         e.g. --port 9000
#   2. Environment var  e.g. LLM_PORT=9000 llm-server …
#   3. This file
#   4. Built-in default
#
# Edit with:  llm-server config edit
# View with:  llm-server config show
# Reset with: llm-server config reset

# ── Server ────────────────────────────────────────────────────────────
# LLM_PORT=8081                  # OpenAI-compatible API port
# LLM_CTX_SIZE=65536             # default context size (tokens)
# LLM_MAX_RESTARTS=5             # give-up threshold for repeated startup fails
# LLM_KEEP_ALIVE=0                # 1 = restart forever (overrides MAX_RESTARTS)
# LLM_HEALTH_TIMEOUT=600         # seconds to wait for server to become healthy

# ── Paths ─────────────────────────────────────────────────────────────
# LLM_MODEL_DIR="$HOME/ai_models"      # default search path for model files
# LLM_CACHE_DIR="$HOME/.cache/llm-server"
# LLM_LOG_DIR="$HOME/.cache/llm-server/logs"

# ── Resource limits ───────────────────────────────────────────────────
# LLM_RAM_BUDGET=                # cap RAM use (MB). Empty = use all available.
# LLM_KV_PLACEMENT=auto          # auto|gpu|cpu. auto reserves GPU KV first.

# ── UI ────────────────────────────────────────────────────────────────
# LLM_ASSUME_YES=0               # 1 = auto-accept all interactive prompts

# ── Backend selection (advanced) ──────────────────────────────────────
# LLAMA_SERVER=                  # path to a specific llama-server binary
# LLM_APP_HOME=                  # portable mode: self-contained install dir
TEMPLATE
}

_settings_show() {
    local file
    file="$(_settings_file_path)"
    echo "Settings file: $file"
    if [[ -f "$file" ]]; then
        echo "Status: exists"
    else
        echo "Status: not yet created (built-in defaults in effect)"
    fi
    echo ""
    echo "Effective values (source: file/env/default):"
    local k v src env_v file_v
    declare -A FILE_VALS=()
    if [[ -f "$file" ]]; then
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*# || -z "$line" ]] && continue
            [[ "$line" =~ ^([A-Z_]+)= ]] || continue
            FILE_VALS["${BASH_REMATCH[1]}"]=1
        done < "$file"
    fi
    for k in LLM_PORT LLM_CTX_SIZE LLM_MAX_RESTARTS LLM_KEEP_ALIVE \
             LLM_HEALTH_TIMEOUT LLM_MODEL_DIR LLM_CACHE_DIR LLM_LOG_DIR \
             LLM_RAM_BUDGET LLM_KV_PLACEMENT LLM_ASSUME_YES LLAMA_SERVER LLM_APP_HOME; do
        env_v="${!k:-}"
        if [[ -n "$env_v" ]]; then
            if [[ -n "${FILE_VALS[$k]:-}" ]]; then
                src="env+file"
            else
                src="env"
            fi
        elif [[ -n "${FILE_VALS[$k]:-}" ]]; then
            src="file"
        else
            src="default"
        fi
        printf '  %-22s = %-30s [%s]\n' "$k" "${env_v:-<unset>}" "$src"
    done
}

_settings_cmd() {
    local file
    file="$(_settings_file_path)"
    local sub="${1:-show}"
    case "$sub" in
        show|"")
            _settings_show
            ;;
        path)
            echo "$file"
            ;;
        edit)
            [[ -f "$file" ]] || _settings_write_template "$file"
            editor="${EDITOR:-${VISUAL:-nano}}"
            command -v "$editor" >/dev/null 2>&1 || editor="vi"
            "$editor" "$file"
            echo "Saved: $file"
            ;;
        reset)
            if [[ -f "$file" ]]; then
                cp "$file" "$file.bak.$(date +%s)" && rm -f "$file"
                echo "Removed $file (backup kept). Built-in defaults will be used."
            else
                echo "No settings file to reset."
            fi
            ;;
        *)
            echo "Usage: llm-server config [show|edit|path|reset]" >&2
            exit 2
            ;;
    esac
}

#!/bin/bash
# lib/05-model.sh — Model detection, mmproj support.
# Strategy selection and vision resolution are called later (after flag building
# sets KV_TOTAL_MB, VRAM_NEEDED_MB).

# ── Helper functions (available immediately on source) ──────────
check_memory_or_die() {
    local pool_mb=$1 pool_label=$2
    local model_overhead_mb=$(( TOTAL_SIZE_MB * VRAM_OVERHEAD_PERCENT / 100 ))
    local needed_mb=$(( model_overhead_mb + KV_TOTAL_MB + COMPUTE_PER_GPU_MB ))
    if (( needed_mb > pool_mb )); then
        local max_kv_mb=$(( pool_mb - model_overhead_mb - COMPUTE_PER_GPU_MB ))
        (( max_kv_mb < 0 )) && max_kv_mb=0
        local max_ctx=0
        (( KV_TOTAL_MB > 0 )) && max_ctx=$(( max_kv_mb * CTX_SIZE / KV_TOTAL_MB ))
        echo ""
        echo "ERROR: Model does not fit in ${pool_label}."
        echo "  Model (with overhead): ${model_overhead_mb}MB"
        echo "  KV cache (ctx=${CTX_SIZE}): ${KV_TOTAL_MB}MB"
        echo "  Compute buffers:       ${COMPUTE_PER_GPU_MB}MB"
        echo "  ─────────────────────────────"
        echo "  Total needed:          ${needed_mb}MB"
        echo "  Available (${pool_label}): ${pool_mb}MB"
        echo "  Shortfall:             $(( needed_mb - pool_mb ))MB"
        (( max_ctx > 0 )) && echo "  Max safe context: --ctx-size $max_ctx"
        echo "  Or use a smaller quantization / model."
        exit 1
    fi
}

mmproj_label() {
    local candidate="$1"
    python3 "$PARSE_GGUF" --format json "$candidate" 2>/dev/null | python3 -c '
import json,sys
try: meta=json.load(sys.stdin)
except Exception: print("unknown"); sys.exit(0)
print(f"{meta.get("basename",meta.get("name","unknown"))} ({meta.get("arch","unknown")})")'
}

mmproj_matches_model() {
    local candidate="$1"
    [[ -f "$candidate" ]] || return 1
    [[ "${LLM_SERVER_SKIP_MMPROJ_CHECK:-0}" == "1" ]] && { echo "  Warning: skipping mmproj check for $candidate" >&2; return 0; }
    local meta
    meta=$(python3 "$PARSE_GGUF" --format json "$candidate" 2>/dev/null) || return 1
    MMPROJ_META="$meta" MMPROJ_FILE="$candidate" TEXT_MODEL_NAME="${GGUF_MODEL_NAME:-}" TEXT_MODEL_BASENAME="${GGUF_BASENAME:-}" python3 -c '
import json,os,re,sys
def norm(v): return re.sub(r"[^a-z0-9]+","",(v or "").lower())
try: meta=json.loads(os.environ["MMPROJ_META"])
except Exception: sys.exit(1)
if meta.get("arch")!="clip": sys.exit(1)
eb=int(meta.get("non_expert_bytes",0))+int(meta.get("expert_bytes",0))
if eb>0:
    try: ab=os.path.getsize(os.environ["MMPROJ_FILE"])
    except OSError: sys.exit(1)
    if ab<eb: sys.exit(1)
t={norm(os.environ.get("TEXT_MODEL_NAME","")),norm(os.environ.get("TEXT_MODEL_BASENAME",""))}
p={norm(meta.get("name","")),norm(meta.get("basename",""))}
t.discard("");p.discard("")
if not t or not p or t.isdisjoint(p): sys.exit(1)'
}

find_local_mmproj() {
    local dir="$MODEL_DIR_PATH" f
    for f in "$dir"/mmproj-F16.gguf "$dir"/mmproj-BF16.gguf "$dir"/mmproj-F32.gguf; do
        [[ -f "$f" ]] || continue
        if mmproj_matches_model "$f"; then echo "$f"; return; fi
        echo "  Skipping incompatible mmproj: $f ($(mmproj_label "$f"))" >&2
    done
    for f in "$dir"/*mmproj*.gguf; do
        [[ -f "$f" ]] || continue
        if mmproj_matches_model "$f"; then echo "$f"; return; fi
        echo "  Skipping incompatible mmproj: $f ($(mmproj_label "$f"))" >&2
    done
    [[ "$dir" != "$MODEL_DIR" ]] && for f in "$MODEL_DIR"/mmproj-F16.gguf "$MODEL_DIR"/mmproj-BF16.gguf; do
        [[ -f "$f" ]] || continue
        if mmproj_matches_model "$f"; then echo "$f"; return; fi
        echo "  Skipping incompatible mmproj: $f ($(mmproj_label "$f"))" >&2
    done
    echo ""
}

list_repo_mmproj_candidates() {
    local repo="$1" tree
    tree=$(curl -sfL "https://huggingface.co/api/models/${repo}/tree/main?recursive=1" 2>/dev/null || true)
    if [[ -n "$tree" ]]; then
        printf '%s' "$tree" | python3 -c '
import json,os,sys
try: data=json.load(sys.stdin)
except: data=[]
paths=[item.get("path","") for item in data if isinstance(item,dict)]
paths=[p for p in paths if p and p.lower().endswith(".gguf") and "mmproj" in p.lower()]
def score(p):
    n=os.path.basename(p).lower()
    for v,r in [("mmproj-f16.gguf",0),("mmproj-bf16.gguf",1),("mmproj-f32.gguf",2),
                 (True if "f16" in n else False,3),(True if "bf16" in n else False,4),
                 (True if "f32" in n else False,5),(n=="mmproj.gguf",6),(True,7)]:
        if v: return (r,len(p),p)
    return (7,len(p),p)
for p in sorted(set(paths),key=score): print(p)'
    fi
    printf '%s\n' "mmproj-F16.gguf" "mmproj-BF16.gguf" "mmproj-F32.gguf" "mmproj.gguf"
}

download_mmproj() {
    local basename="${GGUF_BASENAME:-$GGUF_MODEL_NAME}" quantizer="${GGUF_QUANTIZED_BY:-}"
    [[ -z "$basename" ]] && return 1
    local candidates=()
    [[ -n "$quantizer" ]] && { candidates+=("${quantizer}/${basename}-GGUF");
        [[ -n "$GGUF_MODEL_NAME" && "$GGUF_MODEL_NAME" != "$basename" ]] && candidates+=("${quantizer}/${GGUF_MODEL_NAME}-GGUF"); }
    for q in unsloth bartowski lmstudio-community; do [[ "$q" != "$quantizer" ]] && candidates+=("${q}/${basename}-GGUF"); done
    local safe_basename
    safe_basename=$(printf '%s' "$basename" | tr -cs 'A-Za-z0-9._+-' '_' | sed 's/^_*//;s/_*$//')
    [[ -z "$safe_basename" ]] && safe_basename="model"
    for repo in "${candidates[@]}"; do
        local seen_remote=() remote_path
        while IFS= read -r remote_path; do
            [[ -n "$remote_path" ]] || continue
            local already_seen=0; for seen in "${seen_remote[@]}"; do [[ "$seen" == "$remote_path" ]] && already_seen=1 && break; done
            (( already_seen )) && continue; seen_remote+=("$remote_path")
            local remote_base suffix safe_suffix dest tmp url
            remote_base=$(basename "$remote_path")
            suffix="$remote_base"; suffix="${suffix#mmproj-}"; suffix="${suffix#mmproj_}"
            [[ "$suffix" == "$remote_base" ]] && suffix="$remote_base"
            safe_suffix=$(printf '%s' "$suffix" | tr -cs 'A-Za-z0-9._+-' '_' | sed 's/^_*//;s/_*$//')
            [[ -z "$safe_suffix" ]] && safe_suffix="mmproj.gguf"
            dest="$MODEL_DIR_PATH/mmproj-${safe_basename}-${safe_suffix}"
            tmp="${dest}.tmp.$$"
            if [[ -f "$dest" ]] && mmproj_matches_model "$dest"; then
                echo "  Found compatible downloaded mmproj: $dest" >&2; echo "$dest"; return 0
            fi
            url=$(REPO="$repo" REPO_PATH="$remote_path" python3 -c '
import os,sys;from urllib.parse import quote
print("https://huggingface.co/{}/resolve/main/{}".format(os.environ["REPO"],quote(os.environ["REPO_PATH"],safe="/"))')
            echo "  Trying: $url" >&2; curl -sfL --head "$url" >/dev/null 2>&1 || continue
            echo "  Downloading mmproj from ${repo}: ${remote_path}" >&2; rm -f "$tmp"
            if curl -fL --progress-bar -o "$tmp" "$url" 2>/dev/null; then
                if DEST="$tmp" python3 -c "import os,sys;open(os.environ['DEST'],'rb').read(4)==b'GGUF' or sys.exit(1)" 2>/dev/null; then
                    if ! mmproj_matches_model "$tmp"; then
                        echo "  Warning: downloaded mmproj mismatched ($(mmproj_label "$tmp"))" >&2; rm -f "$tmp"; continue
                    fi
                    mv -f "$tmp" "$dest"; echo "  Downloaded: $dest" >&2; echo "$dest"; return 0
                else echo "  Warning: not a valid GGUF" >&2; rm -f "$tmp"; fi
            fi
        done < <(list_repo_mmproj_candidates "$repo")
    done
    return 1
}

# ── Model detection (runs before flag building) ─────────────────
_detect_model() {
    # Model size detection
    MODEL_DIR_PATH=$(dirname "$MODEL_PATH")
    SPLIT_PATTERN=$(echo "$MODEL_NAME" | sed 's/-[0-9]*-of-[0-9]*\.gguf/.gguf/')
    TOTAL_SIZE_BYTES=$(find "$MODEL_DIR_PATH" -name "${SPLIT_PATTERN%.gguf}*" -type f -print0 2>/dev/null | xargs -0 du -scb 2>/dev/null | tail -1 | awk '{print $1}' || echo 0)
    [[ -z "$TOTAL_SIZE_BYTES" || "$TOTAL_SIZE_BYTES" == "0" ]] && TOTAL_SIZE_BYTES=$(du -sb "$MODEL_PATH" 2>/dev/null | awk '{print $1}' || echo 0)
    TOTAL_SIZE_MB=$(awk "BEGIN {printf \"%.0f\", $TOTAL_SIZE_BYTES/1048576}")
    TOTAL_SIZE_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_SIZE_BYTES/1073741824}")

    # GGUF metadata parsing
    if [[ -f "$PARSE_GGUF" ]]; then
        eval "$(python3 "$PARSE_GGUF" --format shell "$MODEL_PATH" 2>/dev/null || true)"
    else
        echo "Warning: parse_gguf.py not found at $PARSE_GGUF" >&2
    fi

    LAYER_COUNT=${LAYER_COUNT:-0}; EXPERT_COUNT=${EXPERT_COUNT:-0}

    # Architecture-aware scope decision
    (( EXPERT_COUNT == 0 && GPU_COUNT >= 2 )) && DO_UNLIMITED=1 || DO_UNLIMITED=0

    # Sanitize metadata with defaults
    HEAD_COUNT_KV=${HEAD_COUNT_KV:-0}; KEY_LENGTH=${KEY_LENGTH:-0}; VALUE_LENGTH=${VALUE_LENGTH:-0}
    HAS_SSM=${HAS_SSM:-0}; HAS_FUSED=${HAS_FUSED:-0}; EXPERT_BYTES=${EXPERT_BYTES:-0}
    NON_EXPERT_BYTES=${NON_EXPERT_BYTES:-0}; MODEL_ARCH=${MODEL_ARCH:-unknown}
    EMBEDDING_LENGTH=${EMBEDDING_LENGTH:-0}; FEED_FORWARD_LENGTH=${FEED_FORWARD_LENGTH:-0}
    EXPERT_USED_COUNT=${EXPERT_USED_COUNT:-0}; EXPERT_FF=${EXPERT_FF:-0}
    EXPERT_SHARED_FF=${EXPERT_SHARED_FF:-0}; KV_LORA_RANK=${KV_LORA_RANK:-0}
    Q_LORA_RANK=${Q_LORA_RANK:-0}; ROPE_DIM=${ROPE_DIM:-0}
    KEY_LENGTH_MLA=${KEY_LENGTH_MLA:-0}; VALUE_LENGTH_MLA=${VALUE_LENGTH_MLA:-0}
    LEADING_DENSE=${LEADING_DENSE:-0}; SLIDING_WINDOW=${SLIDING_WINDOW:-0}
    FULL_ATTN_INTERVAL=${FULL_ATTN_INTERVAL:-0}; HAS_SHEXP=${HAS_SHEXP:-0}
    CTX_TRAIN=${CTX_TRAIN:-0}; GGUF_MODEL_NAME="${GGUF_MODEL_NAME:-}"
    GGUF_BASENAME="${GGUF_BASENAME:-}"; GGUF_QUANTIZED_BY="${GGUF_QUANTIZED_BY:-}"
    [[ "$LAYER_COUNT" == "0" ]] && { LAYER_COUNT=48; echo "Warning: Could not detect layer count, defaulting to 48" >&2; }

    # DeepSeek V4 warnings
    if [[ "$MODEL_ARCH" == "deepseek4" ]]; then
        echo "Note: arch=deepseek4 — needs a build from llama.cpp PR #22378 or the antirez fork." >&2
    fi
    if [[ "$MODEL_ARCH" == "deepseek2" ]] && (( KEY_LENGTH_MLA > 0 )) && (( KEY_LENGTH_MLA <= ROPE_DIM )); then
        echo "Warning: DeepSeek V4 Flash mistagged as deepseek2. Needs a fork build." >&2
    fi

    IS_MOE=0; (( EXPERT_COUNT > 1 )) && IS_MOE=1

    echo ""
    echo "Model: $MODEL_NAME"
    echo "Size: ${TOTAL_SIZE_GB}GB (${TOTAL_SIZE_MB}MB)"
    echo "Architecture: ${LAYER_COUNT} layers, $([ "$IS_MOE" = "1" ] && echo "${EXPERT_COUNT} experts (MoE)" || echo "dense")$([ "$HAS_FUSED" = "1" ] && echo ", fused up|gate")"
    (( HAS_FUSED && IS_IK_LLAMA )) && echo "Info: Fused up|gate model detected."

    # Health check timeout
    if [[ -n "${LLM_HEALTH_TIMEOUT:-}" ]]; then
        HEALTH_TIMEOUT="$LLM_HEALTH_TIMEOUT"
    else
        HEALTH_TIMEOUT=$(( 240 + TOTAL_SIZE_MB / 1700 ))
        (( IS_MOE && TOTAL_SIZE_MB > 100000 && HEALTH_TIMEOUT < 900 )) && HEALTH_TIMEOUT=900
    fi

    # ── Binary detection & backend tag ────────────────────────────
    SERVER_HELP=$("$LLAMA_SERVER" --help 2>&1 || true)
    IS_IK_LLAMA=0
    [[ "$LLAMA_SERVER" == *ik_llama* ]] && IS_IK_LLAMA=1
    [[ "$SERVER_HELP" == *ikawrakow* || "$SERVER_HELP" == *"split-mode-graph"* ]] && IS_IK_LLAMA=1
    SUPPORTS_REASONING_FLAG=0
    [[ "$SERVER_HELP" == *"--reasoning"* ]] && SUPPORTS_REASONING_FLAG=1

    (( IS_IK_LLAMA )) && echo "Backend: ik_llama.cpp (graph split enabled)" || echo "Backend: llama.cpp (mainline)"
    BACKEND_TAG="llama"; (( IS_IK_LLAMA )) && BACKEND_TAG="ik" || true
}

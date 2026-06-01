#!/bin/bash
# lib/11-update.sh — Self-update and backend update logic.

# ── Download model ────────────────────────────────────────────
_download_model() {
    TOTAL_VRAM_MB=0
    if command -v nvidia-smi &>/dev/null 2>&1; then
        TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}' || echo 0)
    elif command -v amd-smi &>/dev/null; then
        TOTAL_VRAM_MB=$(amd-smi metric --mem 2>/dev/null | grep 'FREE_GTT:' | awk '{s+=$2} END {print s+0}' || echo 0)
    fi
    RAM_AVAIL_MB=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' || echo 0)

    DOWNLOADER="$(dirname "$(readlink -f "$0")")/download_any_gguf.py"
    [[ ! -f "$DOWNLOADER" ]] && DOWNLOADER="${LLM_MODEL_DIR:-$HOME/ai_models}/download_any_gguf.py"
    [[ ! -f "$DOWNLOADER" ]] && { echo "Error: Downloader script (download_any_gguf.py) not found."; exit 1; }

    echo "Launching GGUF Downloader for: $MODEL_ARG"
    echo "Wait for the interactive session..."
    python3 "$DOWNLOADER" --repo "$MODEL_ARG" --dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" --vram "$TOTAL_VRAM_MB" --ram "$RAM_AVAIL_MB"
    echo ""; echo "Download step complete."
    echo "To launch your new model, run: llm-server <path_to_downloaded_file>"
}

# ── Update logic ──────────────────────────────────────────────
_do_update() {
    # Self-update
    LLM_SERVER_REPO="${LLM_SERVER_REPO:-$HOME/llm-server}"
    if [[ -d "$LLM_SERVER_REPO/.git" ]]; then
        echo "═══ Updating llm-server ═══"
        pushd "$LLM_SERVER_REPO" > /dev/null
        old_hash=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
        script_path=$(readlink -f "$0")
        script_backup="${script_path}.bak"
        cp "$script_path" "$script_backup"
        if git pull --ff-only 2>&1 | tail -3; then
            new_hash=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
            if [[ "$old_hash" != "$new_hash" ]]; then
                echo "  Updated: $(git log --oneline "${old_hash}..${new_hash}" | wc -l) new commits"
                git log --oneline "${old_hash}..${new_hash}" | head -5 | sed 's/^/    /'
                if [[ -x "./install.sh" ]]; then
                    echo "  Re-installing..."
                    if ! bash ./install.sh > /dev/null 2>&1; then
                        echo "  Error: Install failed. Rolling back..."
                        git checkout "$old_hash" 2>/dev/null; cp "$script_backup" "$script_path"; popd > /dev/null; return 1
                    fi
                fi
                if ! "$script_path" --version > /dev/null 2>&1; then
                    echo "  Error: New version failed self-check. Rolling back..."
                    git checkout "$old_hash" 2>/dev/null; cp "$script_backup" "$script_path"
                    [[ -x "./install.sh" ]] && bash ./install.sh > /dev/null 2>&1
                    echo "  Rolled back to $old_hash"
                else
                    echo "  ✓ llm-server updated and verified. Restart to use the new version."
                fi
            else echo "  Already up to date."; fi
        else echo "  Warning: git pull failed (unclean repo?). Skipping self-update."; fi
        rm -f "$script_backup"; popd > /dev/null
    else echo "Info: llm-server repo not found at $LLM_SERVER_REPO — skipping self-update."; fi
    echo ""

    update_backend() {
        local name="$1" repo_dir="$2"
        local build_dir="$repo_dir/build"
        local binary="$build_dir/bin/llama-server"
        local fallback_state_dir="$CACHE_DIR/update-fallbacks"
        mkdir -p "$fallback_state_dir"
        echo ""; echo "═══ Updating $name ═══"
        [[ ! -d "$repo_dir/.git" ]] && { echo "  Skip: $repo_dir is not a git repo"; return 1; }
        cd "$repo_dir"
        local old_commit original_branch
        old_commit=$(git rev-parse HEAD 2>/dev/null)
        original_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)
        local old_binary_hash=""
        [[ -x "$binary" ]] && old_binary_hash=$(md5sum "$binary" | awk '{print $1}')
        echo "  Current: $(git log --oneline -1)"
        local binary_backup="$repo_dir/.llm-server.llama-server.backup"
        [[ -x "$binary" ]] && { cp "$binary" "${binary_backup}"; echo "  Backed up working binary"; }
        echo "  Pulling latest..."
        if ! git symbolic-ref -q HEAD >/dev/null; then
            echo "  Warning: detached HEAD detected. Checking out default branch..."
            local default_branch
            default_branch=$(git remote show origin | sed -n '/HEAD branch/s/.*: //p')
            [[ -z "$default_branch" ]] && default_branch="master"
            git checkout "$default_branch"
            old_commit=$(git rev-parse HEAD 2>/dev/null); original_branch="$default_branch"
        fi
        [[ -z "$original_branch" ]] && original_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "master")
        local dirty; dirty=$(git status --porcelain --untracked-files=no 2>/dev/null)
        if [[ -n "$dirty" ]]; then
            echo "  Skip: working tree has uncommitted changes:"; echo "$dirty" | head -5 | sed 's/^/    /'
            echo "    (commit or stash them in $repo_dir, then re-run --update)"; rm -f "${binary_backup}"; return 1
        fi
        if ! git pull --ff-only 2>&1 | tail -3; then
            echo "  Warning: fast-forward pull failed, trying rebase..."
            if ! git pull --rebase 2>&1 | tail -3; then echo "  FAILED: git pull failed — skipping $name"; rm -f "${binary_backup}"; return 1; fi
        fi
        local new_commit; new_commit=$(git rev-parse HEAD 2>/dev/null)
        if [[ "$old_commit" == "$new_commit" ]]; then echo "  Already up to date."; rm -f "${binary_backup}"; return 2; fi
        echo "  Updated: $(git log --oneline -1)"
        if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDACXX:-}" ]]; then
            local _nvcc=""
            for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do [[ -x "$d/nvcc" ]] && _nvcc="$d/nvcc" && export PATH="$d:$PATH" && break; done
            [[ -n "$_nvcc" ]] && export CUDACXX="$_nvcc" && echo "  Found CUDA: $_nvcc"
        fi
        local walkback_max="${LLM_SERVER_UPDATE_WALKBACK:-3}"
        local attempt success=0 target_commit success_commit=""
        for (( attempt=0; attempt < walkback_max; attempt++ )); do
            if (( attempt == 0 )); then target_commit="$new_commit"
            else target_commit=$(git rev-parse "${new_commit}~${attempt}" 2>/dev/null) || break
                echo ""; echo "  ── Attempt $((attempt+1))/${walkback_max}: walking back to $(git log --oneline -1 "$target_commit") ──"
                git checkout --quiet "$target_commit" 2>/dev/null || { echo "  checkout failed; stopping walk-back"; break; }
            fi
            if _update_build_and_test "$build_dir" "$binary"; then success=1; success_commit="$target_commit"; break; fi
        done
        if (( !success )); then
            echo ""; echo "  All ${walkback_max} attempts failed — rolling back to previous version..."
            git checkout "$old_commit" 2>/dev/null
            [[ -f "${binary_backup}" ]] && mv "${binary_backup}" "$binary"
            echo "  Rolled back to $(git log --oneline -1)"; return 1
        fi
        if [[ -n "$success_commit" && "$success_commit" != "$new_commit" ]]; then
            local fallback_marker="$fallback_state_dir/${name//[^A-Za-z0-9_.-]/_}.env"
            { printf 'repo_dir=%q\n' "$repo_dir"; printf 'branch=%q\n' "$original_branch"; printf 'head_commit=%q\n' "$new_commit"
              printf 'fallback_commit=%q\n' "$success_commit"; printf 'recorded_at=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            } > "$fallback_marker"
            echo "  Walk-back succeeded at $(git log --oneline -1 "$success_commit")"
            git checkout --quiet "$original_branch" 2>/dev/null || echo "  Warning: could not reattach to $original_branch" >&2
        fi
        local new_binary_hash; new_binary_hash=$(md5sum "$binary" | awk '{print $1}')
        if [[ "$old_binary_hash" == "$new_binary_hash" ]]; then echo "  Binary unchanged (no relevant code changes)"
        else echo "  New binary built successfully ✓"; fi
        rm -f "${binary_backup}"; echo "  $name updated: $(git log --oneline -1)"; return 0
    }

    _update_build_and_test() {
        local build_dir="$1" binary="$2"
        local nproc; nproc=$(nproc 2>/dev/null || echo 8)
        echo "  Building..."
        if [[ -d "$build_dir" ]]; then
            if ! cmake --build "$build_dir" --config Release -j"$nproc" 2>&1 | tail -5; then
                echo "  Build failed — trying clean reconfigure..."
                local cmake_flags=()
                if [[ -f "$build_dir/CMakeCache.txt" ]]; then
                    local val
                    val=$(grep "^GGML_CUDA:BOOL=ON" "$build_dir/CMakeCache.txt" 2>/dev/null) && cmake_flags+=(-DGGML_CUDA=ON)
                    val=$(grep "^GGML_CUDA_FA_ALL_QUANTS:BOOL=ON" "$build_dir/CMakeCache.txt" 2>/dev/null) && cmake_flags+=(-DGGML_CUDA_FA_ALL_QUANTS=ON)
                    val=$(grep "^GGML_CUDA_NCCL:BOOL=ON" "$build_dir/CMakeCache.txt" 2>/dev/null) && cmake_flags+=(-DGGML_CUDA_NCCL=ON)
                fi
                rm -rf "$build_dir"
                if ! { cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "${cmake_flags[@]}" 2>&1 | tail -5 \
                      && cmake --build "$build_dir" --config Release -j"$nproc" 2>&1 | tail -5; }; then
                    echo "  Build failed at this commit."; return 1
                fi; echo "  Clean rebuild succeeded"
            fi
        else echo "  No build directory — run cmake manually first."; return 1; fi
        [[ ! -x "$binary" ]] && { echo "  Binary missing after build."; return 1; }
        if ! timeout 10 "$binary" --version >/dev/null 2>&1; then
            echo "  Binary crashes on --version at this commit."; return 1
        fi
        local test_model
        test_model=$(find "$HOME/ai_models" -maxdepth 1 -type f -name '*.gguf' -printf '%s\t%p\n' 2>/dev/null \
            | grep -viE 'mmproj|projector' | sort -n | tail -1 | cut -f2-)
        if [[ -z "$test_model" ]]; then echo "  Skipping model launch test (no GGUF models found)"; return 0; fi
        echo "  Testing model launch: $(basename "$test_model")"
        local test_port; test_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo 18081)
        local test_log; test_log=$(mktemp)
        timeout 120s "$binary" -m "$test_model" --ctx-size 512 --parallel 1 --port "$test_port" --host 127.0.0.1 -ngl 0 > "$test_log" 2>&1 &
        local test_pid=$! ready=0
        for _ in $(seq 1 120); do
            if grep -qE "server is listening|main loop|model loaded|HTTP server listening" "$test_log"; then ready=1; break; fi
            kill -0 "$test_pid" 2>/dev/null || break; sleep 1
        done
        kill "$test_pid" 2>/dev/null; wait "$test_pid" 2>/dev/null
        if (( ready )); then echo "  Model launch test: PASSED ✓"; rm -f "$test_log"; return 0; fi
        echo "  Model launch test: FAILED at this commit"
        echo "  --- last 10 lines of test output ---"; tail -10 "$test_log" | sed 's/^/    /'; rm -f "$test_log"; return 1
    }

    echo "═══ llm-server backend updater ═══"
    IK_DIR="$HOME/ik_llama.cpp"; MAIN_DIR="$HOME/llama.cpp"
    ik_status=99; main_status=99
    [[ -d "$IK_DIR" ]] && { ik_status=0; update_backend "ik_llama.cpp" "$IK_DIR" || ik_status=$?; } || echo "ik_llama.cpp not found — skipping"
    [[ -d "$MAIN_DIR" ]] && { main_status=0; update_backend "llama.cpp" "$MAIN_DIR" || main_status=$?; } || echo "llama.cpp not found — skipping"
    _update_label() { case "$1" in 0) echo "updated ✓" ;; 2) echo "already up to date" ;; 99) echo "not present" ;; *) echo "failed/skipped" ;; esac; }
    echo ""; echo "═══ Update summary ═══"
    echo "  ik_llama.cpp: $(_update_label "$ik_status")"; echo "  llama.cpp:    $(_update_label "$main_status")"
    for dir in "$IK_DIR" "$MAIN_DIR"; do
        curr_bin="$dir/build/bin/llama-server"
        if [[ -x "$curr_bin" ]]; then
            ver=$("$curr_bin" --version 2>/dev/null | grep -m1 "version:" || echo "unknown")
            [[ "$ver" == "unknown" ]] && ver=$(cd "$dir" && git log --oneline -1 2>/dev/null || echo "unknown")
            echo "  $(basename "$dir"): $ver"
        fi
    done
}

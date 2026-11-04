#!/usr/bin/env bash
# Head-to-head: llmserve vs llama.cpp server, both driven by `llmserve loadtest` (OpenAI
# /v1/completions, token-id prompts so tokenization is identical, greedy, ignore_eos, same
# ShareGPT requests and output lengths). One server runs at a time, 4 threads each.
#
#   LLAMACPP=/path/to/llama.cpp GGUF_DIR=/path/to/gguf benchmarks/llamacpp_compare.sh
#
# Writes results/llamacpp/<system>__rate<r>.json (skips files that exist) and systems.json.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMACPP=${LLAMACPP:?path to a built llama.cpp checkout}
GGUF_DIR=${GGUF_DIR:?directory with the smollm2-135m f32 and q8_0 GGUF files}
OUT=results/llamacpp
PORT=8090
NP=64           # llama.cpp parallel slots = our max_num_seqs
CTX_PER_SLOT=1024
RATES=${RATES:-"inf 1.0 2.0 4.0"}
mkdir -p "$OUT"

wait_up() {
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "server did not come up" >&2
  return 1
}

run_system() {  # name, rates, command...
  local name=$1 rates=$2
  shift 2
  local todo=0
  for r in $rates; do [ -f "$OUT/${name}__rate$r.json" ] || todo=1; done
  [ $todo = 1 ] || return 0
  "$@" >"runs/server_$name.log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT  # never leave a server behind on failure
  wait_up
  # warm-up (discarded)
  .venv/bin/llmserve loadtest --base-url "http://127.0.0.1:$PORT" --num-requests 8 \
    --rate inf --seed 99 >/dev/null
  for r in $rates; do
    local f="$OUT/${name}__rate$r.json"
    [ -f "$f" ] && continue
    local n=100
    [ "$r" = inf ] && n=200
    echo "== $name rate $r"
    .venv/bin/llmserve loadtest --base-url "http://127.0.0.1:$PORT" --model smollm2-135m \
      --tokenizer smollm2-135m --num-requests $n --rate "$r" --out "$f.tmp"
    .venv/bin/python - "$f.tmp" "$name" "$*" <<'EOF'
import json, sys
p, name, cmd = sys.argv[1:]
d = json.load(open(p))
d["system"] = {"name": name, "server_cmd": cmd}
json.dump(d, open(p, "w"), indent=1)
EOF
    mv "$f.tmp" "$f"
  done
  kill "$pid"
  wait "$pid" 2>/dev/null || true
}

mkdir -p runs
LC_SERVER="$LLAMACPP/build/bin/llama-server"
LC_ARGS=(--port $PORT -t 4 -tb 4 -np $NP -c $((NP * CTX_PER_SLOT)) -b 2048 -ub 512 --no-webui)
git -C "$LLAMACPP" log -1 --format='{"llama_cpp_commit": "%H", "date": "%cI"}' >"$OUT/llamacpp_version.json"

run_system llmserve-fp32 "$RATES" .venv/bin/llmserve serve --model smollm2-135m --port $PORT \
  --threads 4 --max-num-seqs $NP
# llama.cpp with fp32 weights and fp32 KV cache (same dtype as ours)
run_system llamacpp-f32 "$RATES" "$LC_SERVER" -m "$GGUF_DIR/smollm2-135m-f32.gguf" \
  "${LC_ARGS[@]}" -ctk f32 -ctv f32
# Extra, clearly different config: 8-bit weights on both sides (their Q8_0 vs our int8)
run_system llmserve-int8 "inf 2.0" .venv/bin/llmserve serve --model smollm2-135m --port $PORT \
  --threads 4 --max-num-seqs $NP --quantization int8
run_system llamacpp-q8_0 "inf 2.0" "$LC_SERVER" -m "$GGUF_DIR/smollm2-135m-q8_0.gguf" \
  "${LC_ARGS[@]}" -ctk f32 -ctv f32

# Give llama.cpp its best shot: offline-throughput sweep over its main knobs (later flags
# override the defaults in LC_ARGS). Results in results/llamacpp/tuning/.
OUT=results/llamacpp/tuning
mkdir -p "$OUT"
for cfg in "fa-on:-fa on" "np16:-np 16 -c 16384" "np32:-np 32 -c 32768" \
  "np16-fa-on:-np 16 -c 16384 -fa on" "np64-unified:-kvu"; do
  # shellcheck disable=SC2086
  run_system "llamacpp-f32-${cfg%%:*}" inf "$LC_SERVER" -m "$GGUF_DIR/smollm2-135m-f32.gguf" \
    "${LC_ARGS[@]}" -ctk f32 -ctv f32 ${cfg#*:}
done

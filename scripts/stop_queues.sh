#!/usr/bin/env bash
# Stop the benchmark queue chain and its current job, by PID. Never kills this script's own
# ancestors (the shell that invoked it may contain the same words in its command line).
ancestors=" $$ "
p=$$
while [ "$p" -gt 1 ]; do
  p=$(ps -o ppid= -p "$p" | tr -d ' ')
  [ -z "$p" ] && break
  ancestors="$ancestors$p "
done
for pid in $(ps -eo pid=,args= | awk '/[s]cripts\/run_queues.sh|[s]cripts\/run_queue.py|[l]lmserve (bench|ppl)|[b]enchmarks\/(lora_demo|chunked_stall|microbench_quant).py/ {print $1}'); do
  case "$ancestors" in *" $pid "*) continue ;; esac
  kill "$pid" 2>/dev/null
done
sleep 1
echo "still running: $(ps -eo pid=,args= | awk '/[s]cripts\/run_queue.py|[l]lmserve (bench|ppl)/' | wc -l)"

#!/usr/bin/env bash
# Stop the benchmark queue chain and its current job, by PID (never `pkill -f` with shared text).
for pid in $(ps -eo pid=,args= | awk '/[s]cripts\/run_queues.sh|[s]cripts\/run_queue.py|[l]lmserve (bench|ppl)|[b]enchmarks\/(lora_demo|chunked_stall).py/ {print $1}'); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
done
sleep 1
ps -eo pid=,args= | awk '/[s]cripts\/run_queue|[l]lmserve (bench|ppl)/' | wc -l

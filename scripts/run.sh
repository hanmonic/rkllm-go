#!/bin/bash
# rkllm-go v2 dev server on :8081 (on the board). Not a systemd unit on purpose:
# /opt/rkllm-go + rkllm-server.service (:8080) are the production v1 setup.
#
#   ./run.sh start [extra server flags]   start (refuses if another NPU user runs)
#   ./run.sh stop                         SIGTERM, wait for exit (parks sessions, destroys handles)
#   ./run.sh restart [flags]              stop + start
#   ./run.sh status                       pid, /v1/models, loaded handles
#   ./run.sh log                          tail -f the log
#   ./run.sh build                        rebuild both servers from ./src
#
# Env: RUNTIME=131 (default) or 123 picks rkllm-server-131 / rkllm-server-123.
#      LOG_LEVEL=N sets RKLLM_LOG_LEVEL (2 = prompt text/tokens per run; very verbose).
#      ADDR=:8081, CONFIG=/opt/rkllm-go-v2/models.json.
set -u
DIR=/opt/rkllm-go-v2
RUNTIME=${RUNTIME:-131}
BIN=$DIR/rkllm-server-$RUNTIME
ADDR=${ADDR:-:8081}
CONFIG=${CONFIG:-$DIR/models.json}
LOG=$DIR/server.log
PIDF=$DIR/server.pid
PORT=${ADDR##*:}
cd "$DIR" || exit 1

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

npu_users() {
	# other processes that hold the NPU (probe, memtest, flask, other servers)
	pgrep -af '[r]kllm-server|[r]kllm-probe|[p]robe1[23]|[m]emtest1[23]|[f]lask_server|[x]abort' | grep -v "^$(cat "$PIDF" 2>/dev/null) " || true
}

start() {
	if running; then echo "already running (pid $(cat "$PIDF"))"; return 0; fi
	local others; others=$(npu_users)
	if [ -n "$others" ]; then echo "refusing to start, NPU in use by:"; echo "$others"; return 1; fi
	[ -x "$BIN" ] || { echo "missing $BIN (./run.sh build)"; return 1; }
	echo "=== $(date -Is) start $BIN $ADDR $*" >> "$LOG"
	RKLLM_LOG_LEVEL=${LOG_LEVEL:-1} nohup "$BIN" --config "$CONFIG" --addr "$ADDR" "$@" >> "$LOG" 2>&1 &
	echo $! > "$PIDF"
	for _ in $(seq 180); do
		if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
			echo "up on $ADDR (pid $(cat "$PIDF"), runtime $RUNTIME, log $LOG)"; return 0
		fi
		running || { echo "server exited; see $LOG"; tail -20 "$LOG"; return 1; }
		sleep 1
	done
	echo "not answering after 180 s; see $LOG"; return 1
}

stop() {
	running || { echo "not running"; rm -f "$PIDF"; return 0; }
	local pid; pid=$(cat "$PIDF")
	kill -TERM "$pid"
	for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || { rm -f "$PIDF"; echo "stopped"; return 0; }; sleep 1; done
	echo "still running after 60 s, killing"; kill -KILL "$pid"; rm -f "$PIDF"
}

case "${1:-}" in
start) shift; start "$@" ;;
stop) stop ;;
restart) shift; stop; start "$@" ;;
status)
	if running; then echo "running, pid $(cat "$PIDF")"; else echo "not running"; fi
	curl -s "http://127.0.0.1:$PORT/debug/state" | python3 -c 'import json,sys
s=json.load(sys.stdin); print("runtime", s["runtime"], "resident_mb", s["resident_mb_estimate"], "/", s["mem_budget_mb"])
for h in s["handles"]: print(" ", h["model"], h["strategy"], "loaded" if h["loaded"] else "loading", "domains", h["iommu_domains"], "session", h["live_session"] or "-")' 2>/dev/null ;;
log) tail -f "$LOG" ;;
build)
	cd src && go build -tags rk131 -o ../rkllm-server-131 ./cmd/server && go build -tags rk123 -o ../rkllm-server-123 ./cmd/server && echo built ;;
*) sed -n '2,15p' "$0"; exit 1 ;;
esac

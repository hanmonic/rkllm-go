#!/bin/bash
# bench.sh — benchmark all configured models via rkllm-go
# Usage: ./bench.sh [host] [runs_per_prompt]
#   host            default: http://localhost:8080
#   runs_per_prompt default: 3

HOST="${1:-http://localhost:8080}"
RUNS="${2:-3}"
MODELS_JSON="${3:-/opt/rkllm-go/models.json}"

SHORT_PROMPT="What is 2+2?"
MEDIUM_PROMPT="Explain how attention mechanisms work in transformer models in about 100 words."
LONG_PROMPT="Write a detailed technical explanation of the Linux kernel virtual memory system covering page tables, TLB, huge pages, swap, and memory-mapped files."

# Resolve model list from models.json, or fall back to /v1/models
if [ -f "$MODELS_JSON" ]; then
    MODELS=$(python3 -c "import json,sys; d=json.load(open('$MODELS_JSON')); print('\n'.join(d['models'].keys()))")
else
    MODELS=$(curl -sf "$HOST/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(m['id'] for m in d.get('data',[]))) " 2>/dev/null)
fi

if [ -z "$MODELS" ]; then
    echo "ERROR: could not determine model list (tried $MODELS_JSON and $HOST/v1/models)"
    exit 1
fi

# bench_one <model> <label> <prompt> — prints one result line
bench_one() {
    local model="$1"
    local label="$2"
    local prompt="$3"

    local response
    response=$(curl -sf --max-time 300 "$HOST/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"stream\":false}")

    if [ $? -ne 0 ] || [ -z "$response" ]; then
        printf "  %-8s  ERROR (curl failed or timeout)\n" "$label"
        return
    fi

    echo "$response" | python3 -c "
import json, sys
r = json.load(sys.stdin)
if 'error' in r:
    print(f'  {\"$label\":<8}  ERROR: {r[\"error\"][\"message\"]}')
    sys.exit(0)
u = r.get('usage', {})
pt   = u.get('prompt_tokens', 0)
ct   = u.get('completion_tokens', 0)
ptps = u.get('prefill_tps', 0)
gtps = u.get('generate_tps', 0)
pmb  = u.get('memory_usage_mb', 0)
print(f'  {\"$label\":<8}  prompt={pt:>4}t  completion={ct:>4}t  prefill={ptps:>6.1f} t/s  generate={gtps:>5.1f} t/s  mem={pmb:>6.0f} MB')
"
}

# Header
echo "=================================================================="
echo " rkllm-go benchmark — $HOST"
echo " $(date)"
echo "=================================================================="

# Summary accumulator: model -> [prefill_tps, generate_tps, mem_mb] (last run)
SUMMARY=""

for model in $MODELS; do
    echo ""
    echo "--- $model ---"
    for run in $(seq 1 "$RUNS"); do
        printf "  run %d/short:  " "$run"; bench_one "$model" "short"  "$SHORT_PROMPT"
        printf "  run %d/medium: " "$run"; bench_one "$model" "medium" "$MEDIUM_PROMPT"
        printf "  run %d/long:   " "$run"; bench_one "$model" "long"   "$LONG_PROMPT"
    done
done

echo ""
echo "=================================================================="
echo " Done"
echo "=================================================================="

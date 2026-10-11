#!/usr/bin/env python3
"""
bench-long-ctx.py — multi-turn tool call test with long context.

Pads the context with a realistic system prompt + conversation history,
then runs a scenario that requires multiple sequential tool calls.
Simulates tool execution (fake results) and measures per-turn timing,
total context growth, and whether the model correctly synthesises.

Usage:
    python3 bench-long-ctx.py [--host http://localhost:8080]
                               [--models qwen3-4b-16k,qwen3-8b-allnpu]
                               [--runs 3] [--timeout 300]
                               [--ctx-size short|medium|long]
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Tools (same as bench-tools.py)
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        }, "required": ["location"]},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"},
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for recent information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
]

# ---------------------------------------------------------------------------
# Fake tool executor — returns plausible results without a real API
# ---------------------------------------------------------------------------

FAKE_RESULTS = {
    "get_weather": lambda args: (
        f"Weather in {args.get('location', 'unknown')}: 22°C, partly cloudy, "
        f"humidity 65%, wind 12 km/h NW. Feels like 21°C."
    ),
    "calculate": lambda args: (
        f"Result: {eval(args.get('expression', '0'), {'__builtins__': {}}, {})}"
        if all(c in '0123456789+-*/.() ' for c in args.get('expression', ''))
        else f"Result of '{args.get('expression', '')}': 541171"
    ),
    "search_web": lambda args: (
        f"Top results for '{args.get('query', '')}': "
        f"(1) Major AI conference NeurIPS 2026 announced for December in Vancouver. "
        f"(2) ICML 2026 scheduled for July in Vienna. "
        f"(3) New Qwen4 model series released by Alibaba with 128k context."
    ),
}

def execute_tool(name: str, args_str: str) -> str:
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return f"Error: invalid arguments JSON"
    fn = FAKE_RESULTS.get(name)
    if fn is None:
        return f"Error: unknown tool {name}"
    try:
        return fn(args)
    except Exception as e:
        return f"Error executing {name}: {e}"

# ---------------------------------------------------------------------------
# Context padding — realistic long system prompts + history
# ---------------------------------------------------------------------------

SHORT_SYSTEM = "You are a helpful assistant with access to tools."

MEDIUM_SYSTEM = """You are a helpful AI assistant deployed as part of a homelab infrastructure \
monitoring and research platform. You have access to weather data, web search, and calculation tools.

Your job is to help the operator understand conditions across multiple data centre locations, \
research relevant technical topics, and perform calculations related to infrastructure planning.

Always use the available tools when the information requested requires real-time data or \
computation. Synthesise results from multiple tool calls into a clear, structured response. \
If you need to call the same tool multiple times for different inputs, do so — do not skip \
locations or queries to save time.

Respond in a concise, technical style appropriate for an experienced platform engineer."""

LONG_SYSTEM = MEDIUM_SYSTEM + """

## Background context

The operator manages a distributed homelab cluster spanning three physical locations:
- **Lab A** (Tokyo): Primary compute node, NanoPi M6 cluster, RK3588S NPU inference nodes,
  10 Gbps uplink, 32 TB NVMe storage array, running k3s + Flux CD GitOps.
- **Lab B** (London): Secondary DR site, Raspberry Pi cluster for lightweight workloads,
  1 Gbps uplink, 8 TB spinning disk, mirrors critical state from Lab A.
- **Lab C** (Paris): Edge inference node, Orange Pi R6S fleet, primarily handles regional
  AI inference requests to reduce round-trip latency for European users.

All three sites run on a flat WireGuard overlay network (10.100.0.0/16) with BGP route
reflectors at each site. Prometheus + Grafana observability stack deployed centrally in Lab A.
Alertmanager routes to PagerDuty for P1 incidents.

## Recent incident history

- 2026-06-01: Lab B uplink degraded to 200 Mbps for 4 hours due to ISP maintenance.
  GitOps sync fell behind by ~800 commits; auto-recovered on link restore.
- 2026-06-03: NPU thermal throttling event in Lab A during extended inference benchmark.
  Root cause: ambient temperature 31°C exceeded cooling capacity. Mitigated by reducing
  concurrent inference workers from 4 to 2.
- 2026-06-05: False-positive OOMKill on inference pod due to stale cgroup limits from
  previous deployment. Fixed by restarting kubelet on affected node.

## Operator preferences

- Prefer structured markdown output with headers for multi-part responses.
- Always show units for numerical values.
- When reporting weather, include feels-like temperature and wind direction.
- For search results, summarise the top 2-3 findings rather than listing raw URLs."""

# Previous conversation history to add context tokens
FAKE_HISTORY = [
    {"role": "user", "content": "What's the status of the WireGuard tunnels across all three sites?"},
    {"role": "assistant", "content": (
        "Based on the last Prometheus scrape (2 minutes ago), all WireGuard tunnels are healthy:\n\n"
        "| Tunnel | Latency | Last Handshake | Bytes Rx/Tx (1h) |\n"
        "|--------|---------|---------------|------------------|\n"
        "| A↔B (Tokyo–London) | 178ms RTT | 42s ago | 2.1 GB / 890 MB |\n"
        "| A↔C (Tokyo–Paris) | 201ms RTT | 38s ago | 1.4 GB / 670 MB |\n"
        "| B↔C (London–Paris) | 11ms RTT | 51s ago | 340 MB / 290 MB |\n\n"
        "All tunnels within expected latency bounds. B↔C is notably fast due to geographic proximity."
    )},
    {"role": "user", "content": "How much power does the NanoPi M6 cluster draw at full NPU load?"},
    {"role": "assistant", "content": (
        "At full NPU load across all three RK3588S cores, a single NanoPi M6 draws approximately:\n\n"
        "- **Idle**: ~5W\n- **CPU load**: ~12–15W\n- **NPU full load**: ~18–22W\n"
        "- **Peak (CPU+NPU)**: ~25W\n\n"
        "For your 3-node cluster, expect **54–66W** sustained under continuous inference workload. "
        "At 31°C ambient (as seen in the June 3 incident), thermal headroom is reduced — "
        "the SoC will throttle NPU frequency from 1.0 GHz to 0.8 GHz when junction temp "
        "exceeds 85°C, reducing throughput by ~20%."
    )},
]

CONTEXT_SIZES = {
    "short":  {"system": SHORT_SYSTEM,  "history": []},
    "medium": {"system": MEDIUM_SYSTEM, "history": []},
    "long":   {"system": LONG_SYSTEM,   "history": FAKE_HISTORY},
}

# ---------------------------------------------------------------------------
# The multi-tool scenario
# ---------------------------------------------------------------------------

MULTI_TOOL_PROMPT = (
    "I need a full situation report for our three lab sites. Please:\n"
    "1. Check the current weather at Tokyo, London, and Paris\n"
    "2. Search for any major AI infrastructure or hardware announcements this week\n"
    "3. Calculate the total great-circle distance in km for a round trip: "
    "Tokyo (35.6762°N 139.6503°E) → London (51.5074°N 0.1278°W) → "
    "Paris (48.8566°N 2.3522°E) → Tokyo. "
    "Use the Haversine approximation: d = 6371 * 2 * asin(sqrt(sin((lat2-lat1)/2)^2 + "
    "cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)^2)), but just estimate each leg as: "
    "Tokyo-London=9560km, London-Paris=340km, Paris-Tokyo=9720km.\n\n"
    "Combine all results into a structured situation report."
)

EXPECTED_TOOLS = {"get_weather", "search_web", "calculate"}
EXPECTED_MIN_CALLS = 3  # at minimum: weather×3 OR one of each

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def chat_request(host: str, model: str, messages: list, tools: list, timeout: int) -> dict:
    body = json.dumps({"model": model, "messages": messages, "stream": False, "tools": tools}).encode()
    req = urllib.request.Request(
        f"{host}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
    except Exception as e:
        return {"_error": str(e)}

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    elapsed: float
    tool_calls: list  # list of (name, args_str)
    content: str
    prompt_tokens: int
    completion_tokens: int
    error: str = ""

@dataclass
class RunResult:
    turns: list = field(default_factory=list)
    tools_called: set = field(default_factory=set)
    total_elapsed: float = 0.0
    final_content: str = ""
    error: str = ""
    context_tokens_final: int = 0

def run_scenario(host: str, model: str, ctx_size: str, timeout: int) -> RunResult:
    ctx = CONTEXT_SIZES[ctx_size]
    messages = []
    if ctx["system"]:
        messages.append({"role": "system", "content": ctx["system"]})
    messages.extend(ctx["history"])
    messages.append({"role": "user", "content": MULTI_TOOL_PROMPT})

    result = RunResult()
    max_turns = 8

    for turn in range(1, max_turns + 1):
        t0 = time.monotonic()
        resp = chat_request(host, model, messages, TOOLS, timeout)
        elapsed = time.monotonic() - t0

        if "_error" in resp:
            result.error = resp["_error"]
            break

        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "")
        usage = resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        tr = TurnResult(
            turn=turn, elapsed=elapsed,
            tool_calls=[(tc["function"]["name"], tc["function"].get("arguments", "{}")) for tc in tool_calls],
            content=content, prompt_tokens=pt, completion_tokens=ct,
        )
        result.turns.append(tr)
        result.total_elapsed += elapsed

        if finish == "tool_calls" and tool_calls:
            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            # Execute each tool and add results
            for tc in tool_calls:
                name = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                result.tools_called.add(name)
                tool_result = execute_tool(name, args_str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
        else:
            # Model returned a final text response
            result.final_content = content
            result.context_tokens_final = pt + ct
            break
    else:
        result.error = f"hit max turns ({max_turns})"

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--models", default="qwen3-4b-16k,qwen3-8b-allnpu")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--ctx-size", choices=["short", "medium", "long"], default="long")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    print("=" * 72)
    print(f" Long-context multi-tool benchmark  |  {args.host}")
    print(f" context: {args.ctx_size}  |  models: {', '.join(models)}  |  {args.runs} runs each")
    print(f" scenario: 3× get_weather + search_web + calculate (5 tool calls expected)")
    print("=" * 72)

    all_results: dict[str, list[RunResult]] = {}

    for model in models:
        print(f"\n{'━'*72}")
        print(f"  MODEL: {model}")
        print(f"{'━'*72}")
        model_runs = []

        for run_i in range(1, args.runs + 1):
            print(f"\n  run {run_i}/{args.runs}")
            r = run_scenario(args.host, model, args.ctx_size, args.timeout)
            model_runs.append(r)

            if r.error:
                print(f"    ERROR: {r.error}")
                continue

            for tr in r.turns:
                if tr.tool_calls:
                    calls_str = ", ".join(f"{n}(...)" for n, _ in tr.tool_calls)
                    print(f"    turn {tr.turn}: [{calls_str}]  {tr.elapsed:.1f}s  "
                          f"ctx={tr.prompt_tokens}+{tr.completion_tokens}t")
                else:
                    preview = tr.content[:80].replace('\n', ' ')
                    print(f"    turn {tr.turn}: [FINAL] {tr.elapsed:.1f}s  "
                          f"ctx={tr.prompt_tokens}+{tr.completion_tokens}t")
                    print(f"      \"{preview}...\"")

            tools_hit = r.tools_called & EXPECTED_TOOLS
            missing = EXPECTED_TOOLS - r.tools_called
            unexpected = r.tools_called - EXPECTED_TOOLS
            n_calls = sum(len(t.tool_calls) for t in r.turns)

            print(f"    total: {r.total_elapsed:.1f}s  "
                  f"turns={len(r.turns)}  calls={n_calls}  "
                  f"final_ctx={r.context_tokens_final}t")
            print(f"    tools used: {r.tools_called or 'none'}")
            if missing:
                print(f"    MISSING: {missing}")

        all_results[model] = model_runs

    # Summary
    print(f"\n{'='*72}")
    print(" SUMMARY")
    print(f"{'='*72}")
    print(f"  {'model':<22}  {'runs':>4}  {'avg_turns':>9}  {'avg_calls':>9}  "
          f"{'avg_time':>9}  {'all_tools%':>10}  {'final_ctx':>9}")
    print("  " + "-" * 70)

    for model in models:
        runs = [r for r in all_results[model] if not r.error]
        if not runs:
            print(f"  {model:<22}  all runs errored")
            continue
        avg_turns = sum(len(r.turns) for r in runs) / len(runs)
        avg_calls = sum(sum(len(t.tool_calls) for t in r.turns) for r in runs) / len(runs)
        avg_time  = sum(r.total_elapsed for r in runs) / len(runs)
        all_tools_rate = sum(1 for r in runs if EXPECTED_TOOLS <= r.tools_called) / len(runs)
        avg_ctx   = sum(r.context_tokens_final for r in runs) / len(runs)
        print(f"  {model:<22}  {len(runs):>4}  {avg_turns:>9.1f}  {avg_calls:>9.1f}  "
              f"  {avg_time:>7.1f}s  {all_tools_rate*100:>9.0f}%  {avg_ctx:>9.0f}t")

    print(f"{'='*72}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bench-tools.py — test tool-calling reliability across rkllm-go models.

Tests a matrix of scenarios (prompt × system-prompt style) for each model,
runs each scenario N times, and reports call rate, correct-tool rate,
valid-args rate, and variability (std-dev of boolean outcomes).

Usage:
    python3 bench-tools.py [--host http://localhost:8080] [--runs 5]
                           [--models qwen3-4b,qwen3-8b] [--models-json /opt/rkllm-go/models.json]
                           [--timeout 120]
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Tool definitions (sent in every tool-call request)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'London, UK'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid mathematical expression, e.g. '1847 * 293'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System-prompt variants
# ---------------------------------------------------------------------------

SYS_NONE = None  # no system message at all

SYS_NEUTRAL = "You are a helpful assistant."

SYS_ENCOURAGING = (
    "You are a helpful assistant with access to tools. "
    "Use the available tools whenever they would help answer the user's request accurately. "
    "Do not refuse to call a tool when it is clearly the right approach."
)

SYS_DISCOURAGING = (
    "You are a helpful assistant. "
    "Prefer to answer questions directly from your own knowledge. "
    "Only call tools when the information is genuinely unavailable without them."
)

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------
# expected_call=True  → we expect a tool_calls response (score: call rate)
# expected_call=False → we expect a plain text response (score: abstain rate)
# expected_tool=None  → any tool call counts as correct
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    prompt: str
    sys_prompt: Optional[str]
    expected_call: bool
    expected_tool: Optional[str]  # None = any
    description: str


SCENARIOS = [
    # --- obvious single-tool triggers ------------------------------------------
    Scenario(
        id="weather_none",
        prompt="What's the weather like in Tokyo right now?",
        sys_prompt=SYS_NONE,
        expected_call=True,
        expected_tool="get_weather",
        description="weather / no sys",
    ),
    Scenario(
        id="weather_neutral",
        prompt="What's the weather like in Tokyo right now?",
        sys_prompt=SYS_NEUTRAL,
        expected_call=True,
        expected_tool="get_weather",
        description="weather / neutral sys",
    ),
    Scenario(
        id="weather_encouraging",
        prompt="What's the weather like in Tokyo right now?",
        sys_prompt=SYS_ENCOURAGING,
        expected_call=True,
        expected_tool="get_weather",
        description="weather / encouraging sys",
    ),
    Scenario(
        id="weather_discouraging",
        prompt="What's the weather like in Tokyo right now?",
        sys_prompt=SYS_DISCOURAGING,
        expected_call=True,  # should still call — this is genuinely unknowable without a tool
        expected_tool="get_weather",
        description="weather / discouraging sys",
    ),
    # --- calculation -----------------------------------------------------------
    Scenario(
        id="calc_none",
        prompt="What is 1847 multiplied by 293?",
        sys_prompt=SYS_NONE,
        expected_call=True,
        expected_tool="calculate",
        description="calc / no sys",
    ),
    Scenario(
        id="calc_encouraging",
        prompt="What is 1847 multiplied by 293?",
        sys_prompt=SYS_ENCOURAGING,
        expected_call=True,
        expected_tool="calculate",
        description="calc / encouraging sys",
    ),
    Scenario(
        id="calc_discouraging",
        prompt="What is 1847 multiplied by 293?",
        sys_prompt=SYS_DISCOURAGING,
        expected_call=False,  # a capable model should just answer 541171 directly
        expected_tool=None,
        description="calc / discouraging sys",
    ),
    # --- web search ------------------------------------------------------------
    Scenario(
        id="search_none",
        prompt="Search for the latest news about large language models released this week.",
        sys_prompt=SYS_NONE,
        expected_call=True,
        expected_tool="search_web",
        description="search / no sys",
    ),
    Scenario(
        id="search_encouraging",
        prompt="Search for the latest news about large language models released this week.",
        sys_prompt=SYS_ENCOURAGING,
        expected_call=True,
        expected_tool="search_web",
        description="search / encouraging sys",
    ),
    # --- multi-location weather ------------------------------------------------
    Scenario(
        id="multi_weather_none",
        prompt="What's the weather in London and Paris right now?",
        sys_prompt=SYS_NONE,
        expected_call=True,
        expected_tool="get_weather",  # at least one get_weather call
        description="multi-weather / no sys",
    ),
    Scenario(
        id="multi_weather_encouraging",
        prompt="What's the weather in London and Paris right now?",
        sys_prompt=SYS_ENCOURAGING,
        expected_call=True,
        expected_tool="get_weather",
        description="multi-weather / encouraging sys",
    ),
    # --- should NOT call -------------------------------------------------------
    Scenario(
        id="factual_none",
        prompt="What is the capital of France?",
        sys_prompt=SYS_NONE,
        expected_call=False,
        expected_tool=None,
        description="factual / no sys (no tool needed)",
    ),
    Scenario(
        id="factual_encouraging",
        prompt="What is the capital of France?",
        sys_prompt=SYS_ENCOURAGING,
        expected_call=False,
        expected_tool=None,
        description="factual / encouraging sys (over-call test)",
    ),
    Scenario(
        id="greeting_none",
        prompt="Hello! How are you today?",
        sys_prompt=SYS_NONE,
        expected_call=False,
        expected_tool=None,
        description="greeting / no sys (no tool needed)",
    ),
]

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def chat(host: str, model: str, user_prompt: str, sys_prompt: Optional[str],
         tools: list, timeout: int) -> dict:
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": tools,
    }).encode()

    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"_http_error": e.code, "_body": body}
    except Exception as e:
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    called: bool          # did model produce tool_calls?
    correct_tool: bool    # named the expected tool (or any if expected_tool=None)
    valid_args: bool      # arguments is valid JSON
    tool_names: list      # tools actually called
    error: str = ""       # non-empty if request failed


@dataclass
class ScenarioResult:
    scenario: Scenario
    runs: list = field(default_factory=list)  # list[RunResult]

    def call_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.called) / len(self.runs)

    def abstain_rate(self) -> float:
        return 1.0 - self.call_rate()

    def correct_tool_rate(self) -> float:
        called = [r for r in self.runs if r.called]
        if not called:
            return 0.0
        return sum(1 for r in called if r.correct_tool) / len(called)

    def valid_args_rate(self) -> float:
        called = [r for r in self.runs if r.called]
        if not called:
            return 0.0
        return sum(1 for r in called if r.valid_args) / len(called)

    def variability(self) -> float:
        """std-dev of the boolean success outcome across runs."""
        if len(self.runs) < 2:
            return 0.0
        vals = [1.0 if r.called == self.scenario.expected_call else 0.0 for r in self.runs]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(variance)

    def success_rate(self) -> float:
        """Fraction of runs that matched expected_call."""
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.called == self.scenario.expected_call) / len(self.runs)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_run(response: dict, scenario: Scenario) -> RunResult:
    if "_error" in response or "_http_error" in response:
        err = response.get("_error") or f"HTTP {response['_http_error']}: {response.get('_body','')[:120]}"
        return RunResult(called=False, correct_tool=False, valid_args=False, tool_names=[], error=err)

    if "error" in response:
        return RunResult(called=False, correct_tool=False, valid_args=False, tool_names=[],
                         error=response["error"].get("message", "unknown error"))

    choices = response.get("choices", [])
    if not choices:
        return RunResult(called=False, correct_tool=False, valid_args=False, tool_names=[], error="no choices")

    choice = choices[0]
    finish_reason = choice.get("finish_reason", "")
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []

    called = finish_reason == "tool_calls" and len(tool_calls) > 0
    tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    if not called:
        return RunResult(called=False, correct_tool=False, valid_args=False, tool_names=[])

    # Check correct tool
    if scenario.expected_tool is None:
        correct_tool = True
    else:
        correct_tool = scenario.expected_tool in tool_names

    # Check valid JSON args for all calls
    valid_args = True
    for tc in tool_calls:
        args_str = tc.get("function", {}).get("arguments", "")
        try:
            json.loads(args_str)
        except json.JSONDecodeError:
            valid_args = False
            break

    return RunResult(called=True, correct_tool=correct_tool, valid_args=valid_args, tool_names=tool_names)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x: float) -> str:
    return f"{x*100:5.0f}%"

def bar(x: float, width: int = 10) -> str:
    filled = round(x * width)
    return "█" * filled + "░" * (width - filled)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m~\033[0m"

def run_symbol(r: RunResult, expected_call: bool) -> str:
    if r.error:
        return "E"
    ok = r.called == expected_call
    if ok and r.called and not r.correct_tool:
        return WARN  # called something, wrong tool
    return PASS if ok else FAIL


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--runs", type=int, default=5, help="runs per scenario")
    ap.add_argument("--models", default="", help="comma-separated model names; default=all from models.json")
    ap.add_argument("--models-json", default="/opt/rkllm-go/models.json")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per request")
    ap.add_argument("--scenarios", default="", help="comma-separated scenario IDs to run; default=all")
    args = ap.parse_args()

    # Resolve model list
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        try:
            with open(args.models_json) as f:
                cfg = json.load(f)
            models = list(cfg["models"].keys())
        except Exception as e:
            print(f"ERROR: could not read {args.models_json}: {e}", file=sys.stderr)
            print("Pass --models qwen3-4b,qwen3-8b or --models-json <path>", file=sys.stderr)
            sys.exit(1)

    # Resolve scenario list
    if args.scenarios:
        wanted = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in SCENARIOS if s.id in wanted]
    else:
        scenarios = SCENARIOS

    print("=" * 78)
    print(f" rkllm-go tool-call benchmark  |  {args.host}")
    print(f" {len(models)} model(s)  ×  {len(scenarios)} scenario(s)  ×  {args.runs} runs")
    print("=" * 78)

    # model_id -> list[ScenarioResult]
    all_results: dict[str, list[ScenarioResult]] = {}

    for model in models:
        print(f"\n{'━'*78}")
        print(f"  MODEL: {model}")
        print(f"{'━'*78}")

        model_results: list[ScenarioResult] = []

        for sc in scenarios:
            sr = ScenarioResult(scenario=sc)
            print(f"\n  [{sc.id}]  {sc.description}")
            print(f"  prompt: \"{sc.prompt[:70]}{'...' if len(sc.prompt)>70 else ''}\"")
            print(f"  expect: {'CALL ' + (sc.expected_tool or 'any') if sc.expected_call else 'NO CALL'}")
            print()

            for run_i in range(1, args.runs + 1):
                t0 = time.monotonic()
                resp = chat(args.host, model, sc.prompt, sc.sys_prompt, TOOLS, args.timeout)
                elapsed = time.monotonic() - t0

                result = evaluate_run(resp, sc)
                sr.runs.append(result)

                sym = run_symbol(result, sc.expected_call)
                if result.error:
                    detail = f"ERROR: {result.error}"
                elif result.called:
                    tools_str = ", ".join(
                        f"{n}({json.loads(tc.get('function',{}).get('arguments','{}'))})"
                        if sc.expected_tool else n
                        for tc, n in zip(
                            resp.get("choices",[{}])[0].get("message",{}).get("tool_calls",[]),
                            result.tool_names
                        )
                    )
                    detail = f"called: {tools_str}  args_valid={result.valid_args}"
                else:
                    content_preview = ""
                    try:
                        content_preview = resp["choices"][0]["message"]["content"][:60]
                    except Exception:
                        pass
                    detail = f"text: \"{content_preview}\"" if content_preview else "no call"

                print(f"    run {run_i}/{args.runs}  {sym}  {elapsed:5.1f}s  {detail}")

            # Per-scenario summary
            sr_rate = sr.success_rate()
            variability = sr.variability()
            print()
            print(f"  success {pct(sr_rate)}  {bar(sr_rate)}  "
                  f"variability σ={variability:.2f}  "
                  f"(call={pct(sr.call_rate())}  "
                  f"correct_tool={pct(sr.correct_tool_rate())}  "
                  f"valid_args={pct(sr.valid_args_rate())})")

            model_results.append(sr)

        all_results[model] = model_results

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print()
    print("=" * 78)
    print(" SUMMARY — success rate per model per scenario")
    print(" (success = outcome matched expectation; σ = run-to-run variability)")
    print("=" * 78)

    # Header row
    scen_ids = [sc.id for sc in scenarios]
    col_w = max(len(s) for s in scen_ids) + 2
    model_w = max(len(m) for m in models) + 2

    header = f"  {'model':<{model_w}}" + "".join(f"  {s:<{col_w}}" for s in scen_ids)
    print(header)
    print("  " + "-" * (model_w + (col_w + 2) * len(scen_ids)))

    for model in models:
        row = f"  {model:<{model_w}}"
        for sr in all_results[model]:
            rate = sr.success_rate()
            sigma = sr.variability()
            cell = f"{rate*100:3.0f}%σ{sigma:.1f}"
            row += f"  {cell:<{col_w}}"
        print(row)

    # Per-scenario aggregate
    print()
    print("  Per-scenario averages across all models:")
    for i, sc in enumerate(scenarios):
        rates = [all_results[m][i].success_rate() for m in models]
        avg = sum(rates) / len(rates) if rates else 0
        sigmas = [all_results[m][i].variability() for m in models]
        avg_sigma = sum(sigmas) / len(sigmas) if sigmas else 0
        exp = "CALL" if sc.expected_call else "NO-CALL"
        print(f"    {sc.id:<30}  avg={pct(avg)}  σ_avg={avg_sigma:.2f}  [{exp}]")

    print()
    print("=" * 78)
    print(" Done")
    print("=" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rkllm-go v2 integration test (stdlib only; runs on the board).

    python3 scripts/integration_v2.py [--url http://127.0.0.1:8081] [--model qwen3-1.7b ...] [--json out.json]

Per model:
  1. 10-request tool-using conversation (session "<model>-solo"): turn N>=2 must
     have cached_tokens > 0 and prefill_tokens close to the new tokens; the
     final answer must recall facts from turn 1.
  2. two interleaved sessions A/B (5 requests each, alternating): each must
     recall its own facts and not the other's; switches report cache events.
  3. the same request with "cache": false must report cached_tokens == 0.
  4. one streamed request with tools + include_usage: tool_calls deltas carry
     "index", the final chunk carries usage.
Exit status 1 if any assertion fails. Prints a per-request table.
"""
import argparse
import json
import sys
import time
import urllib.request

SYSTEM = (
    "You are a precise assistant for a home automation and arithmetic service. "
    "Always use the provided tools when a question needs live data or arithmetic on large numbers. "
    "When you call a tool, call it with exactly the arguments the user implies. "
    "After tool results arrive, answer briefly in one sentence. "
    "Remember everything the user tells you about themselves during the conversation, "
    "and answer questions about it accurately without calling tools. "
    "Never invent tool results. Keep answers short. "
) * 3

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "add", "description": "Add two integers exactly.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "set_light", "description": "Switch a light on or off.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}},
                       "required": ["room", "on"]}}},
]

WEATHER = {"paris": "18C, light rain", "tokyo": "24C, sunny", "oslo": "3C, snow", "lima": "20C, cloudy"}


def run_tool(name, args):
    try:
        a = json.loads(args) if isinstance(args, str) else args
    except Exception:
        a = {}
    if name == "get_weather":
        c = str(a.get("city", "")).lower()
        return json.dumps({"city": a.get("city"), "weather": WEATHER.get(c, "15C, overcast")})
    if name == "add":
        try:
            return json.dumps({"result": int(a.get("a", 0)) + int(a.get("b", 0))})
        except Exception:
            return json.dumps({"error": "bad arguments"})
    if name == "set_light":
        return json.dumps({"ok": True, "room": a.get("room"), "on": a.get("on")})
    return json.dumps({"error": "unknown tool"})


class Client:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def chat(self, body, session=None, timeout=600):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if session:
            req.add_header("X-Session-Id", session)
        t = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hdr = r.headers.get("X-Session-Id")
            if body.get("stream"):
                chunks = []
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    chunks.append(json.loads(line[6:]))
                return {"stream": chunks, "header": hdr, "wall": time.time() - t}
            out = json.loads(r.read())
            out["header"] = hdr
            out["wall"] = time.time() - t
            return out


FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg)
    return cond


def row(tag, i, u, extra=""):
    rk = u["rkllm"]
    print(f"  {tag:<10} {i:>2} {rk['cache_event']:<9} prompt={u['prompt_tokens']:>5} cached={rk['cached_tokens']:>5} "
          f"prefill={rk['prefill_tokens']:>5} ttft={rk['ttft_ms']:>8.0f}ms prefill_ms={rk['prefill_ms']:>7.0f} "
          f"gen={rk['generate_tokens']:>3} kv={rk['kv_cache_size']:>5} restore={rk.get('restore_ms', 0):>5.0f} "
          f"park={rk.get('park_ms', 0):>5.0f} {rk.get('prefix_cache', '')} {extra}")


class Conversation:
    """Drives a tool-using chat; every HTTP request is one 'request'."""

    def __init__(self, client, model, session, max_tokens=160):
        self.c, self.model, self.session = client, model, session
        self.msgs = [{"role": "system", "content": SYSTEM}]
        self.max_tokens = max_tokens
        self.log = []
        self.turn = 0

    def request(self, cache=True):
        body = {"model": self.model, "messages": self.msgs, "tools": TOOLS, "max_tokens": self.max_tokens,
                "temperature": 0}
        if not cache:
            body["cache"] = False
        r = self.c.chat(body, session=self.session)
        self.turn += 1
        msg = r["choices"][0]["message"]
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                tc = dict(tc)
                tc["id"] = tc.get("id") or f"call_{self.turn}_{i}"
                calls.append(tc)
            am["tool_calls"] = calls
        if msg.get("reasoning_content"):
            am["reasoning_content"] = msg["reasoning_content"]
        self.msgs.append(am)
        for tc in am.get("tool_calls", []):
            self.msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                              "content": run_tool(tc["function"]["name"], tc["function"]["arguments"])})
        self.log.append(r)
        return r, am

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})


# 6 user messages; with 4 tool round trips this is 10 requests
SCRIPT_SOLO = [
    "Hi! My name is Zorblax and my favourite number is 7342. Please just reply OK.",
    "What is the weather in Paris right now?",
    "Please add 123456 and 654321 using the add tool.",
    "Turn the kitchen light on.",
    "What's the weather in Tokyo?",
    "What is my name and my favourite number? Answer in one sentence without tools.",
]


def conversation_test(client, model, results):
    print(f"\n== {model}: 10-request tool conversation")
    conv = Conversation(client, model, f"{model}-solo")
    rows = []
    prev_prompt = None
    i = 0
    script = list(SCRIPT_SOLO)
    pending_user = True
    while i < 14:
        if pending_user:
            if not script:
                break
            conv.user(script.pop(0))
        r, am = conv.request()
        u = r["usage"]
        rk = u["rkllm"]
        i += 1
        new_tokens = u["prompt_tokens"] - prev_prompt if prev_prompt is not None else u["prompt_tokens"]
        row("solo", i, u, f"new~{new_tokens} " + ("tool:" + ",".join(t['function']['name'] for t in am.get('tool_calls', [])) if am.get('tool_calls') else repr((am['content'] or '')[:60])))
        check(r["header"] == f"{model}-solo", f"{model} solo: X-Session-Id header {r['header']}")
        if i >= 2:
            check(rk["cached_tokens"] > 0, f"{model} solo req {i}: cached_tokens == 0 ({rk['cache_event']})")
            # prefill should be about the new tokens (previous reply + new message + glue)
            check(rk["prefill_tokens"] <= new_tokens + 8, f"{model} solo req {i}: prefill {rk['prefill_tokens']} > new {new_tokens}+8")
        rows.append({"i": i, **{k: rk[k] for k in ("cache_event", "cached_tokens", "prefill_tokens", "ttft_ms", "prefill_ms", "kv_cache_size")},
                     "prompt_tokens": u["prompt_tokens"], "new_tokens": new_tokens, "tool_calls": len(am.get("tool_calls", []))})
        prev_prompt = u["prompt_tokens"]
        pending_user = not am.get("tool_calls")
    last = conv.log[-1]["choices"][0]["message"].get("content") or ""
    if not pending_user:  # ended on a tool call; ask the recall question directly
        conv.user(SCRIPT_SOLO[-1])
        r, am = conv.request()
        last = am["content"]
        row("solo+", 11, r["usage"])
    print("  final answer:", repr(last[:200]))
    check("zorblax" in last.lower() and "7342" in last, f"{model} solo: recall failed: {last[:120]!r}")
    results["solo"] = rows
    results["solo_final"] = last
    return conv


def interleave_test(client, model, results):
    print(f"\n== {model}: interleaved sessions A/B")
    facts = {"A": ("Quux", "pineapple", "Lima"), "B": ("Mirabel", "turquoise", "Tokyo")}
    convs = {k: Conversation(client, model, f"{model}-{k}-{int(time.time())}", max_tokens=120) for k in "AB"}
    rows = []
    script = {
        k: [
            f"My name is {facts[k][0]} and my favourite colour is {facts[k][1]}. Reply OK.",
            f"What is the weather in {facts[k][2]}?",
            "Add 4242 and 1717 with the tool.",
            "What is my favourite colour? One word, no tools.",
            "What is my name? One word, no tools.",
        ] for k in "AB"
    }
    answers = {"A": [], "B": []}
    for step in range(5):
        for k in "AB":
            conv = convs[k]
            conv.user(script[k][step])
            for _ in range(3):  # follow tool calls within the same step
                r, am = conv.request()
                rk = r["usage"]["rkllm"]
                row(f"sess-{k}", step + 1, r["usage"], repr((am['content'] or '')[:50]) if not am.get('tool_calls') else "tool")
                rows.append({"session": k, "step": step + 1, "cache_event": rk["cache_event"], "cached_tokens": rk["cached_tokens"],
                             "prefill_tokens": rk["prefill_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"], "ttft_ms": rk["ttft_ms"],
                             "restore_ms": rk.get("restore_ms", 0), "park_ms": rk.get("park_ms", 0)})
                if not am.get("tool_calls"):
                    answers[k].append(am["content"] or "")
                    break
    for k, other in (("A", "B"), ("B", "A")):
        colour = answers[k][3].lower() if len(answers[k]) > 3 else ""
        name = answers[k][4].lower() if len(answers[k]) > 4 else ""
        check(facts[k][1] in colour, f"{model} session {k}: colour recall {colour!r}")
        check(facts[k][0].lower() in name, f"{model} session {k}: name recall {name!r}")
        check(facts[other][0].lower() not in name and facts[other][1] not in colour, f"{model} session {k}: cross-talk")
    results["interleave"] = rows
    results["interleave_answers"] = answers
    return convs


def cache_off_test(client, model, conv, results):
    print(f"\n== {model}: cache off")
    conv.user("Thanks, that is all. Say goodbye in three words.")
    r, _ = conv.request(cache=False)
    u = r["usage"]
    row("off", 1, u)
    check(u["rkllm"]["cached_tokens"] == 0 and u["prompt_tokens_details"]["cached_tokens"] == 0, f"{model} cache off: cached_tokens != 0")
    check(u["rkllm"]["cache_strategy"] == "off", f"{model} cache off: strategy {u['rkllm']['cache_strategy']}")
    check(u["rkllm"]["prefill_tokens"] >= u["prompt_tokens"] - 1, f"{model} cache off: not a full prefill")
    results["cache_off"] = u["rkllm"]
    # the next turn of that session is a miss / re-prefill, not an error
    conv.user("One more: what's the weather in Lima?")
    r, _ = conv.request()
    row("after-off", 2, r["usage"])
    results["after_off"] = r["usage"]["rkllm"]


def stream_test(client, model, results):
    print(f"\n== {model}: streaming with tools")
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "tools": TOOLS, "temperature": 0,
            "max_tokens": 120, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the weather in Oslo?"}]}
    r = client.chat(body, session=f"{model}-stream")
    chunks = r["stream"]
    content = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
    calls = [tc for c in chunks if c["choices"] for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
    usage = [c for c in chunks if c.get("usage")]
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    print(f"  chunks={len(chunks)} content={content[:60]!r} calls={calls} finish={finish}")
    check("<tool_call>" not in content, f"{model} stream: tool markup leaked into content")
    check(all("index" in tc for tc in calls), f"{model} stream: tool_calls without index")
    check(len(usage) == 1 and usage[0]["choices"] == [] and "rkllm" in usage[0]["usage"], f"{model} stream: usage chunk")
    check(r["header"] == f"{model}-stream", f"{model} stream: header")
    results["stream"] = {"calls": calls, "finish": finish, "usage": usage[0]["usage"] if usage else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", action="append")
    ap.add_argument("--json")
    ap.add_argument("--skip", default="", help="comma list of solo,interleave,off,stream")
    a = ap.parse_args()
    client = Client(a.url)
    skip = set(a.skip.split(","))
    out = {}
    for m in a.model or ["qwen3-1.7b", "q35-2b"]:
        res = out.setdefault(m, {})
        conv = None
        if "solo" not in skip:
            conv = conversation_test(client, m, res)
        if "interleave" not in skip:
            interleave_test(client, m, res)
        if "off" not in skip and conv:
            cache_off_test(client, m, conv, res)
        if "stream" not in skip:
            stream_test(client, m, res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print("\nFAILURES:" if FAILS else "\nALL CHECKS PASSED")
    for f in FAILS:
        print(" -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

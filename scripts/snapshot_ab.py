#!/usr/bin/env python3
"""A/B check of cache strategies on the same conversation (stdlib only).

    # 1. live: run the interleaved A/B dialogue of integration_v2.py and record
    #    every request (messages sent, output, usage)
    python3 scripts/snapshot_ab.py live   --model q35-2b --out live-append.json
    # 2. replay: send exactly the recorded requests (same message histories,
    #    same interleaving) to a server and compare the outputs per request
    python3 scripts/snapshot_ab.py replay --model q35-2b --src live-append.json --out rep-snap.json [--no-cache]
    # 3. compare two recordings turn by turn
    python3 scripts/snapshot_ab.py diff a.json b.json
    # 4. check a RKLLM_LOG_LEVEL=2 server log: every "Loaded prompt cache with N
    #    tokens" must be followed by a prompt whose first N tokens equal an earlier
    #    prompt of N tokens (the file content), and kv size must be N
    python3 scripts/snapshot_ab.py logcheck server.log

Replays should run against a freshly started server for each strategy, so
that history rendering (no stored raw generations) is identical across runs.
"""
import argparse
import copy
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integration_v2 import TOOLS, Client, Conversation  # noqa: E402


def out_of(r):
    msg = r["choices"][0]["message"]
    calls = [[tc["function"]["name"], tc["function"]["arguments"]] for tc in msg.get("tool_calls") or []]
    return {"content": msg.get("content") or "", "tool_calls": calls}


def live(client, model, session_prefix):
    facts = {"A": ("Quux", "pineapple", "Lima"), "B": ("Mirabel", "turquoise", "Tokyo")}
    convs = {k: Conversation(client, model, f"{session_prefix}-{k}", max_tokens=120) for k in "AB"}
    script = {k: [
        f"My name is {facts[k][0]} and my favourite colour is {facts[k][1]}. Reply OK.",
        f"What is the weather in {facts[k][2]}?",
        "Add 4242 and 1717 with the tool.",
        "What is my favourite colour? One word, no tools.",
        "What is my name? One word, no tools.",
    ] for k in "AB"}
    rec = []
    for step in range(5):
        for k in "AB":
            conv = convs[k]
            conv.user(script[k][step])
            for _ in range(3):
                sent = copy.deepcopy(conv.msgs)
                r, am = conv.request()
                rec.append({"session": k, "step": step + 1, "messages": sent, "out": out_of(r), "usage": r["usage"]["rkllm"]})
                print_row(rec[-1])
                if not am.get("tool_calls"):
                    break
    return rec


def replay(client, model, src, session_prefix, cache):
    rec = []
    for x in src:
        body = {"model": model, "messages": x["messages"], "tools": TOOLS, "max_tokens": 120, "temperature": 0}
        if not cache:
            body["cache"] = False
        r = client.chat(body, session=f"{session_prefix}-{x['session']}")
        rec.append({"session": x["session"], "step": x["step"], "messages": x["messages"], "out": out_of(r), "usage": r["usage"]["rkllm"]})
        print_row(rec[-1], same=rec[-1]["out"] == x["out"])
    return rec


def print_row(x, same=None):
    u = x["usage"]
    o = x["out"]
    what = "tool:" + ",".join(f"{n}({a})" for n, a in o["tool_calls"]) if o["tool_calls"] else repr(o["content"][:70])
    flag = "" if same is None else ("  same" if same else "  DIFF")
    print(f"  {x['session']} {x['step']} {u['cache_event']:<9} prompt={u['prompt_tokens']:>5} cached={u['cached_tokens']:>5} "
          f"prefill={u['prefill_tokens']:>5} ttft={u['ttft_ms']:>6.0f} restore={u.get('restore_ms', 0):>4.0f} {what}{flag}")


def norm(msgs):
    """messages without tool-call ids (random per run, not rendered by the templates)"""
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if k != "tool_call_id"}
        if m.get("tool_calls"):
            m["tool_calls"] = [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]} for t in m["tool_calls"]]
        out.append(m)
    return out


def diff(a, b):
    n = min(len(a), len(b))
    same = 0
    for i in range(n):
        x, y = a[i], b[i]
        eq = x["out"] == y["out"]
        same += eq
        mark = "same" if eq else "DIFF"
        hist = "" if norm(x["messages"]) == norm(y["messages"]) else " (histories differ)"
        print(f"{i:>2} {x['session']}{x['step']} {mark}{hist}")
        if not eq:
            print("     a:", json.dumps(x["out"])[:200])
            print("     b:", json.dumps(y["out"])[:200])
    print(f"{same}/{n} identical outputs; lengths {len(a)} vs {len(b)}")


TOK_RE = re.compile(r"':(\d+)")


def logcheck(path):
    runs = []  # (tokens, loaded, kv)
    loaded = None
    kvline = re.compile(r"Current kv cache size is (\d+)")
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"Loaded prompt cache with (\d+) tokens", line)
            if m:
                loaded = int(m.group(1))
                continue
            if "Prompt tokens: [" in line:
                toks = [int(t) for t in TOK_RE.findall(line.rsplit("], total:", 1)[0])]
                runs.append({"tokens": toks, "loaded": loaded, "kv": None})
                loaded = None
                continue
            m = kvline.search(line)
            if m and runs and runs[-1]["kv"] is None:
                runs[-1]["kv"] = int(m.group(1))
    ok = bad = 0
    for i, r in enumerate(runs):
        if r["loaded"] is None:
            continue
        n = r["loaded"]
        src = None
        for j in range(i - 1, -1, -1):
            p = runs[j]["tokens"]
            if len(p) == n and r["tokens"][:n] == p:
                src = j
                break
        good = src is not None and r["kv"] == n
        ok += good
        bad += not good
        print(f"run {i:>3}: loaded {n:>5} tok, kv {r['kv']}, prompt {len(r['tokens'])} tok -> "
              + (f"exact prefix of run {src}" if src is not None else "NO earlier prompt matches") + ("" if good else "  <-- CHECK"))
    print(f"{ok} consistent restores, {bad} inconsistent, {len(runs)} runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["live", "replay", "diff", "logcheck"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model")
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    c = Client(a.url)
    tag = f"ab{int(time.time())}"
    if a.cmd == "live":
        rec = live(c, a.model, tag)
    elif a.cmd == "replay":
        rec = replay(c, a.model, json.load(open(a.src)), tag, not a.no_cache)
    elif a.cmd == "diff":
        return diff(json.load(open(a.args[0])), json.load(open(a.args[1])))
    else:
        return logcheck(a.args[0])
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

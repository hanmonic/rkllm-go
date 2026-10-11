// Command rkllm-bench measures multi-turn agent performance of an rkllm-go
// server with KV-cache reuse on vs off.
//
//	rkllm-bench run --server URL --models a,b --scenarios house-batch,coding-agent --modes on,off --out DIR
//	rkllm-bench report DIR
//	rkllm-bench fakeserver --addr 127.0.0.1:8081
//	rkllm-bench list
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"rkllm-go/cmd/rkllm-bench/bench"
	"rkllm-go/cmd/rkllm-bench/fakeserver"
	"rkllm-go/cmd/rkllm-bench/scenarios"
)

const defaultModels = "qwen3-0.6b,qwen3-1.7b,qwen3-4b-16k,qwen3-8b-allnpu,q35-2b,q35-4b,q35-9b,gemma3n-e4b"

func usage() {
	fmt.Fprintf(os.Stderr, `usage:
  rkllm-bench run --server URL [--models a,b] [--scenarios s1,s2] [--modes on,off] --out DIR
                  [--max-turns N] [--timeout 30m] [--smoke] [--ctx model=16384,...] [--no-stream]
  rkllm-bench report DIR
  rkllm-bench fakeserver [--addr 127.0.0.1:8081] [--time-scale 0.01] [--malformed-every N]
  rkllm-bench list [--smoke]

scenarios: %s
run 'rkllm-bench run -h' for all flags
`, strings.Join(scenarios.Names, ","))
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = cmdRun(os.Args[2:])
	case "report":
		err = cmdReport(os.Args[2:])
	case "fakeserver":
		err = cmdFake(os.Args[2:])
	case "list":
		err = cmdList(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseCtx(s string) (map[string]int, int, error) {
	m := map[string]int{}
	def := 0
	for _, p := range splitList(s) {
		k, v, ok := strings.Cut(p, "=")
		if !ok { // bare number = default for all models
			n, err := strconv.Atoi(p)
			if err != nil {
				return nil, 0, fmt.Errorf("--ctx %q: want model=N or N", p)
			}
			def = n
			continue
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, 0, fmt.Errorf("--ctx %q: %v", p, err)
		}
		m[k] = n
	}
	return m, def, nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	server := fs.String("server", "http://127.0.0.1:8081", "server base URL")
	models := fs.String("models", defaultModels, "comma-separated model ids")
	scen := fs.String("scenarios", strings.Join(scenarios.Names, ","), "comma-separated scenarios")
	modes := fs.String("modes", "on,off", "cache modes to run: on, off")
	out := fs.String("out", "", "output directory (required; reused for resume)")
	maxTurns := fs.Int("max-turns", 0, "max assistant turns per task (0 = per-scenario default: house 12, coding 20, hard 6, interleaved 20)")
	timeout := fs.Duration("timeout", 30*time.Minute, "per-request timeout")
	smoke := fs.Bool("smoke", false, "run one short case per scenario")
	noStream := fs.Bool("no-stream", false, "use non-streaming requests (no client-side TTFT)")
	seed := fs.Int("seed", 42, "seed sent with every request (-1 = omit)")
	ctxFlag := fs.String("ctx", "", "context window overrides: model=N,... or N for all (default: /v1/models field, else 16384 for *16k*/q35*/8b-allnpu, 4096 otherwise)")
	noWarm := fs.Bool("no-warmup", false, "skip the per-model warmup request (model load then lands in the first task)")
	parallel := fs.Bool("parallel-interleave", false, "run interleaved sessions concurrently instead of round-robin")
	_ = fs.Parse(args)
	if *out == "" {
		return fmt.Errorf("--out is required")
	}
	ctxm, def, err := parseCtx(*ctxFlag)
	if err != nil {
		return err
	}
	r, err := bench.NewRunner(bench.Config{
		Server: *server, Models: splitList(*models), Scenarios: splitList(*scen), Modes: splitList(*modes), Out: *out,
		MaxTurns: *maxTurns, Timeout: *timeout, Smoke: *smoke, Stream: !*noStream, Seed: *seed, CtxLimits: ctxm, DefaultCtx: def,
		Warmup: !*noWarm, Parallel: *parallel, Log: os.Stderr,
	})
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := r.Run(ctx); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "report: %s/report.md\n", *out)
	return nil
}

func cmdReport(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: rkllm-bench report DIR")
	}
	if err := bench.WriteReport(args[0]); err != nil {
		return err
	}
	b, err := os.ReadFile(args[0] + "/report.md")
	if err != nil {
		return err
	}
	os.Stdout.Write(b)
	return nil
}

func cmdFake(args []string) error {
	fs := flag.NewFlagSet("fakeserver", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8081", "listen address")
	scale := fs.Float64("time-scale", 0.01, "multiply synthetic NPU timings (1 = real-time estimates, 0 = no sleeping)")
	malformed := fs.Int("malformed-every", 0, "every Nth completion leaks raw tool-call text (0 = never)")
	exposeCtx := fs.Bool("expose-context", false, "include max_context in /v1/models")
	_ = fs.Parse(args)
	s := fakeserver.New(fakeserver.Options{TimeScale: *scale, MalformedEvery: *malformed, ExposeContext: *exposeCtx})
	fmt.Fprintf(os.Stderr, "fake rkllm-go on http://%s (time-scale %g)\n", *addr, *scale)
	return http.ListenAndServe(*addr, s.Handler())
}

func cmdList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	smoke := fs.Bool("smoke", false, "list the smoke subset")
	_ = fs.Parse(args)
	for _, n := range scenarios.Names {
		specs, _ := scenarios.Specs(n, *smoke)
		ids := make([]string, len(specs))
		for i, s := range specs {
			ids[i] = s.ID
		}
		fmt.Printf("%-15s max_turns=%-3d %s\n", n, bench.DefaultMaxTurns[n], strings.Join(ids, ", "))
	}
	return nil
}

// rkllm-server: OpenAI-compatible server for the RK3588 NPU with KV-cache
// reuse across turns. Build with -tags rk123 for runtime 1.2.3 (default 1.3.1).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"rkllm-go/internal/api"
	"rkllm-go/internal/config"
	"rkllm-go/internal/core"
	"rkllm-go/internal/engine/rk"
)

func main() {
	var (
		configPath   = flag.String("config", "", "path to models.json (default: models.json next to the binary)")
		listenAddr   = flag.String("addr", ":8081", "HTTP listen address")
		maxContext   = flag.Int("max-context", 4096, "default max_context_len for models without one in models.json")
		maxTokens    = flag.Int("max-tokens", 2048, "default max new tokens per response (init value; per-request max_tokens on 1.3.1)")
		temperature  = flag.Float64("temperature", 0, "default sampling temperature (0 = greedy)")
		topK         = flag.Int("top-k", 1, "default top-k (1 = greedy)")
		topP         = flag.Float64("top-p", 1.0, "default top-p")
		repeatPen    = flag.Float64("repeat-penalty", 1.0, "repetition penalty")
		liveAppend   = flag.Bool("live-append", false, "Qwen3.5 (experimental): append kh=1 deltas to live sessions; 3-4x lower TTFT on long coding sessions but lower accuracy on 2-turn tasks (see docs/benchmarks-2026-10-10.md)")
		cacheMode    = flag.String("cache", "on", "on | off (off: full clear + cold prefill before every request)")
		queueDepth   = flag.Int("queue-depth", 4, "waiting requests per model before 503")
		sessionDir   = flag.String("session-dir", "/dev/shm/rkllm-sessions", "directory for parked session / prefix cache files (wiped at start)")
		sessionBytes = flag.String("session-budget", "8GiB", "byte budget for cache files (LRU eviction); 0 disables parking")
		memBudget    = flag.Int("mem-budget-mb", 18000, "resident model memory budget in MB (LRU model swap above it)")
		prefixCache  = flag.Bool("prefix-cache", true, "pre-save system+tools prefix caches for new sessions")
		minPrefix    = flag.Int("min-prefix-bytes", 1024, "minimum system+tools prefix size for the shared prefix cache")
		preloadFlag  = flag.String("preload", "", "comma-separated models to preload (adds to models.json preload)")
	)
	flag.Parse()

	path := *configPath
	if path == "" {
		if exe, err := os.Executable(); err == nil {
			path = filepath.Join(filepath.Dir(exe), "models.json")
		}
	}
	cfg, err := config.Load(path, *maxContext)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	budget, err := parseBytes(*sessionBytes)
	if err != nil {
		log.Fatalf("--session-budget: %v", err)
	}
	if *cacheMode != "on" && *cacheMode != "off" {
		log.Fatalf("--cache must be on or off")
	}
	eng := rk.New()
	log.Printf("rkllm runtime %s, %d models configured from %s, cache=%s", eng.Version(), len(cfg.Models), path, *cacheMode)
	pool := core.NewPool(cfg, eng, core.Options{
		QueueDepth: *queueDepth, SessionDir: *sessionDir, SessionBudget: budget, MemBudgetMB: *memBudget,
		CacheOff: *cacheMode == "off", NoLiveAppend: !*liveAppend, PrefixCache: *prefixCache, MinPrefixBytes: *minPrefix,
		MaxNewTokens: *maxTokens, TopK: *topK, TopP: float32(*topP), Temperature: float32(*temperature), RepeatPenalty: float32(*repeatPen),
	})
	var preload []string
	for _, n := range cfg.Names() {
		if cfg.Models[n].Preload {
			preload = append(preload, n)
		}
	}
	for _, n := range strings.Split(*preloadFlag, ",") {
		if n = strings.TrimSpace(n); n != "" && !slices.Contains(preload, n) {
			preload = append(preload, n)
		}
	}
	for _, n := range preload {
		t := time.Now()
		if err := pool.Preload(n); err != nil {
			log.Printf("preload %s: %v", n, err)
			continue
		}
		log.Printf("preloaded %s in %v", n, time.Since(t).Round(time.Millisecond))
	}

	mux := http.NewServeMux()
	api.NewHandler(pool, api.Defaults{TopK: *topK, TopP: float32(*topP), Temperature: float32(*temperature)}).Register(mux)
	srv := &http.Server{Addr: *listenAddr, Handler: mux, ReadHeaderTimeout: 30 * time.Second, IdleTimeout: 120 * time.Second}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	go func() {
		log.Printf("listening on %s", *listenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()
	<-quit
	log.Println("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
	pool.Close()
}

func parseBytes(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToUpper(s))
	mult := int64(1)
	for _, u := range []struct {
		suf string
		m   int64
	}{{"GIB", 1 << 30}, {"MIB", 1 << 20}, {"KIB", 1 << 10}, {"GB", 1e9}, {"MB", 1e6}, {"KB", 1e3}, {"G", 1 << 30}, {"M", 1 << 20}, {"B", 1}} {
		if strings.HasSuffix(s, u.suf) {
			mult = u.m
			s = strings.TrimSuffix(s, u.suf)
			break
		}
	}
	var n float64
	if _, err := fmt.Sscan(s, &n); err != nil {
		return 0, err
	}
	return int64(n * float64(mult)), nil
}

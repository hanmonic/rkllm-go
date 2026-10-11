//go:build linux && cgo

// rkllm-probe: empirical probe of RKLLM runtime KV-cache semantics.
//
// Usage:
//
//	rkllm-probe list                         list built-in experiment scripts
//	rkllm-probe run <name|file.json> [-model path] [-ctx N] [-fmt qwen|gemma] [-v]
//	rkllm-probe show <name>                  print a built-in script
//
// A script is a JSON array of ops executed sequentially (see op struct below).
// Every op prints one ">> {json}" result line; runs also print the generated text.
package main

import (
	"bufio"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

//go:embed scripts/*.json
var scriptsFS embed.FS

type op struct {
	Op   string `json:"op"`
	H    string `json:"h,omitempty"` // handle name (default "main")
	Note string `json:"note,omitempty"`

	// init
	Model       string   `json:"model,omitempty"`
	Ctx         int      `json:"ctx,omitempty"`
	NKeep       *int     `json:"n_keep,omitempty"`
	Temp        *float64 `json:"temp,omitempty"`
	SkipSpecial *bool    `json:"skip_special,omitempty"` // init: skip_special_token (default true)

	// run
	Prompt     string  `json:"prompt,omitempty"`
	Role       *string `json:"role,omitempty"` // default "user"; JSON null => keep default; "<nil>" => NULL
	Think      bool    `json:"think,omitempty"`
	KH         int     `json:"kh,omitempty"`
	MaxNew     int     `json:"max_new,omitempty"`
	Save       string  `json:"save,omitempty"`
	Pause      int     `json:"pause,omitempty"`
	PauseOn    []int32 `json:"pause_on,omitempty"` // pause (callback returns 1) when one of these token ids is emitted
	Tokens     []int32 `json:"tokens,omitempty"`
	TokensFrom string  `json:"tokens_from,omitempty"` // label of a previous run: feed its generated ids
	TokensPre  []int32 `json:"tokens_pre,omitempty"`
	TokensPost []int32 `json:"tokens_post,omitempty"`
	UseTokens  bool    `json:"use_tokens,omitempty"`
	Label      string  `json:"label,omitempty"`
	Events     bool    `json:"events,omitempty"`
	Show       int     `json:"show,omitempty"`

	// template
	Sys  string `json:"sys,omitempty"`
	Pre  string `json:"pre,omitempty"`
	Post string `json:"post,omitempty"`

	// clear
	Keep     int    `json:"keep,omitempty"`
	Start    *int   `json:"start,omitempty"`
	End      *int   `json:"end,omitempty"`
	StartVar string `json:"start_var,omitempty"`
	EndVar   string `json:"end_var,omitempty"`

	// kv / file / load
	Var  string `json:"var,omitempty"`
	Path string `json:"path,omitempty"`
}

type env struct {
	models  map[string]*Model
	vars    map[string]int
	toks    map[string][]int32
	defPath string
	defCtx  int
	fmtName string
	verbose bool
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "list":
		ents, _ := scriptsFS.ReadDir("scripts")
		names := []string{}
		for _, e := range ents {
			names = append(names, strings.TrimSuffix(e.Name(), ".json"))
		}
		sort.Strings(names)
		for _, n := range names {
			fmt.Println(n)
		}
	case "show":
		b, err := loadScript(os.Args[2])
		must(err)
		os.Stdout.Write(b)
	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		model := fs.String("model", "/opt/rkllm/models/qwen3-1.7b/model.rkllm", "default model path")
		ctx := fs.Int("ctx", 4096, "default max_context_len")
		fmtName := fs.String("fmt", "qwen", "prompt format for {{U:..}}/{{A}} macros: qwen|gemma (gemma3 tags)|gemma4")
		verbose := fs.Bool("v", false, "print per-callback events for every run")
		if len(os.Args) < 3 {
			usage()
		}
		name := os.Args[2]
		fs.Parse(os.Args[3:])
		b, err := loadScript(name)
		must(err)
		var ops []op
		must(json.Unmarshal(b, &ops))
		e := &env{models: map[string]*Model{}, vars: map[string]int{}, toks: map[string][]int32{},
			defPath: *model, defCtx: *ctx, fmtName: *fmtName, verbose: *verbose}
		fmt.Printf("## runtime=%s script=%s model=%s ctx=%d fmt=%s\n", rtName(), name, *model, *ctx, *fmtName)
		for i, o := range ops {
			e.exec(i, o)
		}
		for n, m := range e.models {
			if m != nil {
				m.Destroy()
			}
			_ = n
		}
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: rkllm-probe list | show <name> | run <name|file.json> [-model p] [-ctx n] [-fmt qwen|gemma] [-v]")
	os.Exit(2)
}

func rtName() string {
	if is131() {
		return "1.3.1"
	}
	return "1.2.3"
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func loadScript(name string) ([]byte, error) {
	if strings.HasSuffix(name, ".json") {
		if b, err := os.ReadFile(name); err == nil {
			return b, nil
		}
		name = strings.TrimSuffix(filepath.Base(name), ".json")
	}
	return scriptsFS.ReadFile("scripts/" + name + ".json")
}

func emit(m map[string]any) {
	b, _ := json.Marshal(m)
	fmt.Printf(">> %s\n", b)
}

func (e *env) model(o op) *Model {
	h := o.H
	if h == "" {
		h = "main"
	}
	m := e.models[h]
	if m == nil {
		fmt.Printf("!! handle %q not initialised\n", h)
		os.Exit(1)
	}
	return m
}

func (e *env) exec(i int, o op) {
	h := o.H
	if h == "" {
		h = "main"
	}
	if o.Note != "" {
		fmt.Printf("\n-- [%d] %s: %s\n", i, o.Op, o.Note)
	} else {
		fmt.Printf("\n-- [%d] %s\n", i, o.Op)
	}
	switch o.Op {
	case "note":
	case "init":
		path := o.Model
		if path == "" {
			path = e.defPath
		} else if !strings.Contains(path, "/") {
			path = "/opt/rkllm/models/" + path + "/model.rkllm"
		}
		ctx := o.Ctx
		if ctx == 0 {
			ctx = e.defCtx
		}
		maxNew := o.MaxNew
		if maxNew == 0 {
			maxNew = 256
		}
		nkeep := -1
		if o.NKeep != nil {
			nkeep = *o.NKeep
		}
		temp := 0.0
		if o.Temp != nil {
			temp = *o.Temp
		}
		rssBefore, _ := rss()
		t0 := time.Now()
		skip := true
		if o.SkipSpecial != nil {
			skip = *o.SkipSpecial
		}
		m, ret := Init(path, ctx, maxNew, nkeep, temp, skip)
		dt := time.Since(t0)
		rssAfter, hwm := rss()
		if ret == 0 {
			e.models[h] = m
		}
		kv, kret := -1, -1
		if ret == 0 {
			kv, kret = m.KV()
		}
		emit(map[string]any{"op": "init", "h": h, "model": path, "ctx": ctx, "max_new": maxNew, "n_keep": nkeep,
			"ret": ret, "skip_special": skip, "ms": dt.Milliseconds(), "rss_before_mb": rssBefore, "rss_after_mb": rssAfter, "hwm_mb": hwm,
			"kv": kv, "kv_ret": kret})
		if ret != 0 {
			os.Exit(1)
		}
	case "destroy":
		m := e.model(o)
		ret := m.Destroy()
		delete(e.models, h)
		r, hw := rss()
		emit(map[string]any{"op": "destroy", "h": h, "ret": ret, "rss_mb": r, "hwm_mb": hw})
	case "template":
		m := e.model(o)
		ret := m.SetTemplate(e.expand(o.Sys), e.expand(o.Pre), e.expand(o.Post))
		emit(map[string]any{"op": "template", "h": h, "ret": ret, "sys": o.Sys, "pre": o.Pre, "post": o.Post})
	case "functools":
		// sys -> system prompt, prompt -> tools JSON, post -> tool_response_str
		m := e.model(o)
		ret := m.SetFunctionTools(o.Sys, o.Prompt, o.Post)
		emit(map[string]any{"op": "functools", "h": h, "ret": ret})
	case "kv":
		m := e.model(o)
		kv, ret := m.KV()
		if o.Var != "" {
			e.vars[o.Var] = kv
		}
		emit(map[string]any{"op": "kv", "h": h, "kv": kv, "ret": ret, "var": o.Var})
	case "clear":
		m := e.model(o)
		useRange := o.Start != nil || o.StartVar != ""
		s, en := 0, 0
		if o.Start != nil {
			s = *o.Start
		}
		if o.End != nil {
			en = *o.End
		}
		if o.StartVar != "" {
			s = e.vars[o.StartVar]
		}
		if o.EndVar != "" {
			en = e.vars[o.EndVar]
		}
		kb, _ := m.KV()
		t0 := time.Now()
		ret := m.Clear(o.Keep, useRange, s, en)
		dt := time.Since(t0)
		ka, _ := m.KV()
		emit(map[string]any{"op": "clear", "h": h, "keep_sys": o.Keep, "range": useRange, "start": s, "end": en,
			"ret": ret, "kv_before": kb, "kv_after": ka, "us": dt.Microseconds()})
	case "load":
		m := e.model(o)
		kb, _ := m.KV()
		t0 := time.Now()
		ret := m.LoadCache(o.Path)
		dt := time.Since(t0)
		ka, _ := m.KV()
		emit(map[string]any{"op": "load", "h": h, "path": o.Path, "ret": ret, "ms": float64(dt.Microseconds()) / 1000,
			"kv_before": kb, "kv_after": ka, "file_bytes": fsize(o.Path)})
	case "release":
		m := e.model(o)
		kb, _ := m.KV()
		ret := m.ReleaseCache()
		ka, _ := m.KV()
		emit(map[string]any{"op": "release", "h": h, "ret": ret, "kv_before": kb, "kv_after": ka})
	case "file":
		emit(map[string]any{"op": "file", "path": o.Path, "bytes": fsize(o.Path)})
	case "rm":
		err := os.Remove(o.Path)
		emit(map[string]any{"op": "rm", "path": o.Path, "err": fmt.Sprint(err)})
	case "rss":
		r, hw := rss()
		emit(map[string]any{"op": "rss", "rss_mb": r, "hwm_mb": hw})
	case "isrunning":
		emit(map[string]any{"op": "isrunning", "h": h, "ret": e.model(o).IsRunning()})
	case "run":
		e.run(h, o)
	default:
		fmt.Printf("!! unknown op %q\n", o.Op)
		os.Exit(1)
	}
}

func (e *env) run(h string, o op) {
	m := e.model(o)
	a := RunArgs{Role: "user", Think: o.Think, KeepHist: o.KH, MaxNew: o.MaxNew, SavePath: o.Save,
		PauseAfter: o.Pause, PauseOn: o.PauseOn, KeepEvents: o.Events || e.verbose}
	if o.Role != nil {
		if *o.Role == "<nil>" {
			a.NilRole = true
		} else {
			a.Role = *o.Role
		}
	}
	if o.Temp != nil {
		a.Temp = *o.Temp
	}
	a.Prompt = e.expand(o.Prompt)
	if o.UseTokens || o.Tokens != nil || o.TokensFrom != "" {
		a.UseTokens = true
		t := append([]int32{}, o.TokensPre...)
		t = append(t, o.Tokens...)
		if o.TokensFrom != "" {
			t = append(t, e.toks[o.TokensFrom]...)
		}
		t = append(t, o.TokensPost...)
		a.Tokens = t
	}
	if o.Save != "" {
		os.Remove(o.Save)
	}
	kb, _ := m.KV()
	r1, _ := rss()
	rs, ret, wall := m.Run(a)
	ka, _ := m.KV()
	r2, hw := rss()
	if o.Label != "" {
		e.vars[o.Label+".kv_before"] = kb
		e.vars[o.Label+".kv_after"] = ka
		e.toks[o.Label] = rs.ids
	}
	text := string(rs.text)
	ft := -1.0
	if rs.firstTok >= 0 {
		ft = float64(rs.firstTok.Microseconds()) / 1000
	}
	res := map[string]any{"op": "run", "h": h, "label": o.Label, "kh": o.KH, "ret": ret,
		"kv_before": kb, "kv_after": ka, "kv_delta": ka - kb,
		"prompt_chars": len(a.Prompt), "n_input_tokens": len(a.Tokens),
		"perf": rs.perf, "perf_seen_in_states": rs.perfAt,
		"n_normal": rs.nNormal, "n_waiting": rs.nWaiting, "n_finish": rs.nFinish, "n_error": rs.nError,
		"paused": rs.paused, "wall_ms": float64(wall.Microseconds()) / 1000, "first_cb_ms": ft,
		"rss_before_mb": r1, "rss_after_mb": r2, "hwm_mb": hw,
	}
	if rs.perf.PrefillMs > 0 {
		res["prefill_tps"] = round1(float64(rs.perf.PrefillTok) / rs.perf.PrefillMs * 1000)
	}
	if rs.perf.GenerateMs > 0 {
		res["decode_tps"] = round1(float64(rs.perf.GenerateTok) / rs.perf.GenerateMs * 1000)
	}
	if o.Save != "" {
		res["save_path"] = o.Save
		res["save_bytes"] = fsize(o.Save)
	}
	nid := len(rs.ids)
	if nid > 12 {
		res["ids_head"] = rs.ids[:6]
		res["ids_tail"] = rs.ids[nid-6:]
	} else {
		res["ids"] = rs.ids
	}
	emit(res)
	show := o.Show
	if show == 0 {
		show = 400
	}
	if len(text) > show {
		text = text[:show] + "...[+" + strconv.Itoa(len(rs.text)-show) + "B]"
	}
	fmt.Printf("TEXT: %q\n", text)
	if a.KeepEvents {
		w := bufio.NewWriter(os.Stdout)
		for _, ev := range rs.events {
			b, _ := json.Marshal(ev)
			fmt.Fprintf(w, "   ev %s\n", b)
		}
		w.Flush()
	}
}

func round1(f float64) float64 { return float64(int(f*10+0.5)) / 10 }

var fillerRe = regexp.MustCompile(`\{\{FILLER:([0-9:]+)\}\}`)
var macroRe = regexp.MustCompile(`\{\{([A-Z]+)(?::([^}]*))?\}\}`)

// expand handles prompt macros:
//
//	{{FILLER:n[:seed]}}  n deterministic filler sentences (~20 tokens each for Qwen)
//	{{SYS:text}}         system turn in the selected format
//	{{U:text}}           user turn (open+content+close)
//	{{A}}                open assistant turn (with empty think block for qwen)
//	{{END}}              end-of-turn token for the selected format
func (e *env) expand(s string) string {
	// pass 1: FILLER (so it can be nested inside {{U:...}}); pass 2: turn macros
	s = fillerRe.ReplaceAllStringFunc(s, func(mt string) string {
		g := fillerRe.FindStringSubmatch(mt)
		parts := strings.Split(g[1], ":")
		n, _ := strconv.Atoi(parts[0])
		seed := 0
		if len(parts) > 1 {
			seed, _ = strconv.Atoi(parts[1])
		}
		return filler(n, seed)
	})
	return macroRe.ReplaceAllStringFunc(s, func(mt string) string {
		g := macroRe.FindStringSubmatch(mt)
		name, arg := g[1], g[2]
		gemma := e.fmtName == "gemma"
		if e.fmtName == "gemma4" {
			switch name {
			case "SYS":
				return "<|turn>system\n" + arg + "<turn|>\n"
			case "U":
				return "<|turn>user\n" + arg + "<turn|>\n"
			case "A":
				return "<|turn>model\n"
			case "END":
				return "<turn|>\n"
			}
		}
		switch name {
		case "FILLER":
			parts := strings.Split(arg, ":")
			n, _ := strconv.Atoi(parts[0])
			seed := 0
			if len(parts) > 1 {
				seed, _ = strconv.Atoi(parts[1])
			}
			return filler(n, seed)
		case "SYS":
			if gemma {
				return "<start_of_turn>user\n" + arg + "\n\n" // gemma has no system role; prepended to first user turn
			}
			return "<|im_start|>system\n" + arg + "<|im_end|>\n"
		case "U":
			if gemma {
				return "<start_of_turn>user\n" + arg + "<end_of_turn>\n"
			}
			return "<|im_start|>user\n" + arg + "<|im_end|>\n"
		case "A":
			if gemma {
				return "<start_of_turn>model\n"
			}
			return "<|im_start|>assistant\n<think>\n\n</think>\n\n"
		case "END":
			if gemma {
				return "<end_of_turn>\n"
			}
			return "<|im_end|>\n"
		}
		return mt
	})
}

var adjs = []string{"blue", "quiet", "rapid", "ancient", "bright", "hollow", "silver", "narrow", "golden", "frozen", "gentle", "crooked", "distant", "heavy", "lucky", "pale"}
var nouns = []string{"fox", "engineer", "river", "lantern", "falcon", "tractor", "garden", "violin", "harbor", "mountain", "teacher", "compass", "kettle", "orchard", "satellite", "bakery"}
var verbs = []string{"moved", "painted", "counted", "repaired", "carried", "sorted", "delivered", "measured", "inspected", "borrowed"}
var things = []string{"crates", "apples", "letters", "bricks", "ropes", "candles", "maps", "barrels", "coins", "feathers"}

func filler(n, seed int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		k := i + seed*1000
		fmt.Fprintf(&b, "Record %d: the %s %s %s %d %s to dock %d. ",
			k, adjs[(k*7+3)%len(adjs)], nouns[(k*13+5)%len(nouns)], verbs[(k*3+1)%len(verbs)],
			(k*37+11)%997, things[(k*11+7)%len(things)], (k*53+17)%89)
	}
	return b.String()
}

func fsize(p string) int64 {
	st, err := os.Stat(p)
	if err != nil {
		return -1
	}
	return st.Size()
}

func rss() (float64, float64) {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return -1, -1
	}
	var r, h float64
	for _, l := range strings.Split(string(b), "\n") {
		f := strings.Fields(l)
		if len(f) >= 2 {
			v, _ := strconv.ParseFloat(f[1], 64)
			switch f[0] {
			case "VmRSS:":
				r = v / 1024
			case "VmHWM:":
				h = v / 1024
			}
		}
	}
	return round1(r), round1(h)
}

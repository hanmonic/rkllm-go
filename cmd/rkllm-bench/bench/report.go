package bench

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Stat is a small distribution summary.
type Stat struct {
	N    int     `json:"n"`
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P90  float64 `json:"p90"`
}

func stat(xs []float64) Stat {
	if len(xs) == 0 {
		return Stat{}
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	sum := 0.0
	for _, x := range s {
		sum += x
	}
	q := func(p float64) float64 {
		i := int(math.Ceil(p*float64(len(s)))) - 1
		if i < 0 {
			i = 0
		}
		return s[i]
	}
	return Stat{N: len(s), Mean: sum / float64(len(s)), P50: q(0.5), P90: q(0.9)}
}

// ComboSummary aggregates one (model, scenario, mode).
type ComboSummary struct {
	Model          string         `json:"model"`
	Scenario       string         `json:"scenario"`
	Mode           string         `json:"mode"`
	Smoke          bool           `json:"smoke,omitempty"`
	Tasks          int            `json:"tasks"`
	Passed         int            `json:"passed"`
	SuccessRate    float64        `json:"success_rate"`
	MeanScore      float64        `json:"mean_score"`
	Turns          int            `json:"turns"`
	TurnsPerTask   float64        `json:"turns_per_task"`
	TaskWallMs     Stat           `json:"task_wall_ms"`
	TimeToSuccess  Stat           `json:"time_to_success_ms"` // passed tasks only
	TurnWallMs     Stat           `json:"turn_wall_ms"`
	ServerTTFTMs   Stat           `json:"server_ttft_ms"`
	ClientTTFTMs   Stat           `json:"client_ttft_ms"`
	PromptTokens   int            `json:"prompt_tokens"`
	PrefillTokens  int            `json:"prefill_tokens"`
	CachedTokens   int            `json:"cached_tokens"`
	PrefillPerTurn float64        `json:"prefill_tokens_per_turn"`
	CachedPerTurn  float64        `json:"cached_tokens_per_turn"`
	HitRatio       float64        `json:"cache_hit_ratio"` // cached / prompt, token weighted
	PrefillTPS     float64        `json:"prefill_tps"`
	DecodeTPS      float64        `json:"decode_tps"`
	MaxContext     int            `json:"max_context_tokens"`
	Malformed      int            `json:"malformed_tool_calls"`
	UnknownTools   int            `json:"unknown_tools"`
	Errors         int            `json:"errors"`
	CacheEvents    map[string]int `json:"cache_events"`
	StopReasons    map[string]int `json:"stop_reasons"`
	QueueMs        float64        `json:"queue_ms"`
	ModelLoadMs    float64        `json:"model_load_ms"`
}

// Speedup compares cache on vs off for one (model, scenario).
type Speedup struct {
	Model             string  `json:"model"`
	Scenario          string  `json:"scenario"`
	SuccessOn         float64 `json:"success_on"`
	SuccessOff        float64 `json:"success_off"`
	TaskWallOnMs      float64 `json:"task_wall_on_ms"`
	TaskWallOffMs     float64 `json:"task_wall_off_ms"`
	WallSpeedup       float64 `json:"wall_speedup"`
	TTFTOnMs          float64 `json:"ttft_on_ms"`
	TTFTOffMs         float64 `json:"ttft_off_ms"`
	TTFTSpeedup       float64 `json:"ttft_speedup"`
	PrefillOnPerTurn  float64 `json:"prefill_on_per_turn"`
	PrefillOffPerTurn float64 `json:"prefill_off_per_turn"`
	PrefillReduction  float64 `json:"prefill_reduction"`
	TTSOnMs           float64 `json:"time_to_success_on_ms"`
	TTSOffMs          float64 `json:"time_to_success_off_ms"`
	TTSSpeedup        float64 `json:"time_to_success_speedup"` // paired: tasks passing in both modes
	PairedTasks       int     `json:"paired_tasks"`
}

// EventStat is TTFT/prefill by cache_event.
type EventStat struct {
	Model    string  `json:"model"`
	Scenario string  `json:"scenario"`
	Mode     string  `json:"mode"`
	Event    string  `json:"event"`
	N        int     `json:"n"`
	TTFTMs   Stat    `json:"server_ttft_ms"`
	Prefill  float64 `json:"prefill_tokens_mean"`
	Cached   float64 `json:"cached_tokens_mean"`
}

// Summary is summary.json.
type Summary struct {
	Generated string         `json:"generated"`
	Dir       string         `json:"dir"`
	RunIDs    []string       `json:"run_ids"`
	Combos    []ComboSummary `json:"combos"`
	Speedups  []Speedup      `json:"speedups"`
	Events    []EventStat    `json:"cache_events"`
}

func scenarioLabel(s string, smoke bool) string {
	if smoke {
		return s + " (smoke)"
	}
	return s
}

// Summarize aggregates records.
func Summarize(dir string, turns []TurnRecord, tasks []TaskRecord) Summary {
	type key struct{ model, scenario, mode string }
	kOf := func(m, s, mode string, smoke bool) key { return key{m, scenarioLabel(s, smoke), mode} }
	tByKey := map[key][]TurnRecord{}
	for _, t := range turns {
		k := kOf(t.Model, t.Scenario, t.Mode, t.Smoke)
		tByKey[k] = append(tByKey[k], t)
	}
	kByKey := map[key][]TaskRecord{}
	var keys []key
	runs := map[string]bool{}
	for _, t := range tasks {
		k := kOf(t.Model, t.Scenario, t.Mode, t.Smoke)
		if _, ok := kByKey[k]; !ok {
			keys = append(keys, k)
		}
		kByKey[k] = append(kByKey[k], t)
		runs[t.RunID] = true
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.model != b.model {
			return a.model < b.model
		}
		if a.scenario != b.scenario {
			return a.scenario < b.scenario
		}
		return a.mode > b.mode // "on" before "off"
	})
	sum := Summary{Generated: time.Now().UTC().Format(time.RFC3339), Dir: dir}
	for r := range runs {
		sum.RunIDs = append(sum.RunIDs, r)
	}
	sort.Strings(sum.RunIDs)
	byKey := map[key]ComboSummary{}
	evAgg := map[[4]string][]TurnRecord{}
	var evKeys [][4]string
	for _, k := range keys {
		ks, ts := kByKey[k], tByKey[k]
		cs := ComboSummary{Model: k.model, Scenario: k.scenario, Mode: k.mode, Tasks: len(ks), CacheEvents: map[string]int{}, StopReasons: map[string]int{}}
		var wall, tts, score []float64
		for _, t := range ks {
			cs.Smoke = t.Smoke
			if t.Pass {
				cs.Passed++
				if t.TimeToSuccessMs != nil {
					tts = append(tts, *t.TimeToSuccessMs)
				}
			}
			score = append(score, t.Score)
			wall = append(wall, t.WallMs)
			cs.Turns += t.Turns
			cs.StopReasons[t.StopReason]++
			if t.MaxContext > cs.MaxContext {
				cs.MaxContext = t.MaxContext
			}
		}
		cs.SuccessRate = float64(cs.Passed) / float64(max(1, cs.Tasks))
		cs.MeanScore = stat(score).Mean
		cs.TurnsPerTask = float64(cs.Turns) / float64(max(1, cs.Tasks))
		cs.TaskWallMs, cs.TimeToSuccess = stat(wall), stat(tts)
		var tw, sttft, cttft []float64
		var genTok, preTok int
		var genMs, preMs float64
		ok := 0
		for _, t := range ts {
			if t.Error != "" {
				cs.Errors++
				continue
			}
			ok++
			tw = append(tw, t.WallMs)
			if t.ClientTTFTMs > 0 {
				cttft = append(cttft, t.ClientTTFTMs)
			}
			cs.PromptTokens += t.PromptTokens
			cs.CachedTokens += t.CachedTokens
			cs.Malformed += t.Malformed
			cs.UnknownTools += t.UnknownTools
			if t.RK != nil {
				sttft = append(sttft, t.RK.TTFTMs)
				cs.PrefillTokens += t.RK.PrefillTokens
				genTok += t.RK.GenerateTokens
				genMs += t.RK.GenerateMs
				preTok += t.RK.PrefillTokens
				preMs += t.RK.PrefillMs
				cs.QueueMs += t.RK.QueueMs
				cs.ModelLoadMs += t.RK.ModelLoadMs
				ev := t.RK.CacheEvent
				if ev == "" {
					ev = "?"
				}
				cs.CacheEvents[ev]++
				ek := [4]string{k.model, k.scenario, k.mode, ev}
				if _, seen := evAgg[ek]; !seen {
					evKeys = append(evKeys, ek)
				}
				evAgg[ek] = append(evAgg[ek], t)
			} else {
				cs.PrefillTokens += t.PromptTokens - t.CachedTokens
			}
		}
		cs.TurnWallMs, cs.ServerTTFTMs, cs.ClientTTFTMs = stat(tw), stat(sttft), stat(cttft)
		if ok > 0 {
			cs.PrefillPerTurn = float64(cs.PrefillTokens) / float64(ok)
			cs.CachedPerTurn = float64(cs.CachedTokens) / float64(ok)
		}
		if cs.PromptTokens > 0 {
			cs.HitRatio = float64(cs.CachedTokens) / float64(cs.PromptTokens)
		}
		if genMs > 0 {
			cs.DecodeTPS = float64(genTok) / (genMs / 1000)
		}
		if preMs > 0 {
			cs.PrefillTPS = float64(preTok) / (preMs / 1000)
		}
		sum.Combos = append(sum.Combos, cs)
		byKey[k] = cs
	}
	for _, ek := range evKeys {
		ts := evAgg[ek]
		var tt []float64
		var pre, cac float64
		for _, t := range ts {
			tt = append(tt, t.RK.TTFTMs)
			pre += float64(t.RK.PrefillTokens)
			cac += float64(t.CachedTokens)
		}
		n := float64(len(ts))
		sum.Events = append(sum.Events, EventStat{Model: ek[0], Scenario: ek[1], Mode: ek[2], Event: ek[3], N: len(ts), TTFTMs: stat(tt), Prefill: pre / n, Cached: cac / n})
	}
	sort.Slice(sum.Events, func(i, j int) bool {
		a, b := sum.Events[i], sum.Events[j]
		return a.Model+a.Scenario+a.Mode+a.Event < b.Model+b.Scenario+b.Mode+b.Event
	})
	// speedups
	seen := map[[2]string]bool{}
	for _, k := range keys {
		pk := [2]string{k.model, k.scenario}
		if seen[pk] {
			continue
		}
		seen[pk] = true
		on, okOn := byKey[key{k.model, k.scenario, "on"}]
		off, okOff := byKey[key{k.model, k.scenario, "off"}]
		if !okOn || !okOff {
			continue
		}
		sp := Speedup{Model: k.model, Scenario: k.scenario, SuccessOn: on.SuccessRate, SuccessOff: off.SuccessRate,
			TaskWallOnMs: on.TaskWallMs.Mean, TaskWallOffMs: off.TaskWallMs.Mean,
			PrefillOnPerTurn: on.PrefillPerTurn, PrefillOffPerTurn: off.PrefillPerTurn,
			TTSOnMs: on.TimeToSuccess.Mean, TTSOffMs: off.TimeToSuccess.Mean}
		sp.WallSpeedup = ratio(off.TaskWallMs.Mean, on.TaskWallMs.Mean)
		sp.TTFTOnMs, sp.TTFTOffMs = on.ServerTTFTMs.Mean, off.ServerTTFTMs.Mean
		if sp.TTFTOnMs == 0 && sp.TTFTOffMs == 0 {
			sp.TTFTOnMs, sp.TTFTOffMs = on.ClientTTFTMs.Mean, off.ClientTTFTMs.Mean
		}
		sp.TTFTSpeedup = ratio(sp.TTFTOffMs, sp.TTFTOnMs)
		if off.PrefillPerTurn > 0 {
			sp.PrefillReduction = 1 - on.PrefillPerTurn/off.PrefillPerTurn
		}
		// paired time-to-success: same task passing in both modes
		pass := func(mode string) map[string]float64 {
			m := map[string]float64{}
			for _, t := range kByKey[key{k.model, k.scenario, mode}] {
				if t.Pass && t.TimeToSuccessMs != nil {
					m[t.Task] = *t.TimeToSuccessMs
				}
			}
			return m
		}
		pOn, pOff := pass("on"), pass("off")
		var sOn, sOff float64
		for id, v := range pOn {
			if w, ok := pOff[id]; ok {
				sOn += v
				sOff += w
				sp.PairedTasks++
			}
		}
		sp.TTSSpeedup = ratio(sOff, sOn)
		sum.Speedups = append(sum.Speedups, sp)
	}
	return sum
}

func ratio(a, b float64) float64 {
	if a == 0 || b == 0 {
		return 0
	}
	return a / b
}

// WriteReport (re)builds summary.json, report.md and all_turns.jsonl in dir.
func WriteReport(dir string) error {
	turns, tasks, err := LoadDir(dir, false)
	if err != nil {
		return err
	}
	sum := Summarize(dir, turns, tasks)
	if err := writeJSONAtomic(filepath.Join(dir, "summary.json"), sum); err != nil {
		return err
	}
	all, err := createJSONL(filepath.Join(dir, "all_turns.jsonl"))
	if err != nil {
		return err
	}
	for _, t := range turns {
		if err := all.Write(t); err != nil {
			all.Close()
			return err
		}
	}
	all.Close()
	return os.WriteFile(filepath.Join(dir, "report.md"), []byte(Markdown(sum)), 0o644)
}

func secs(msv float64) string {
	if msv == 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f", msv/1000)
}

func msf(v float64) string {
	if v == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f", v)
}

func xf(v float64) string {
	if v == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2fx", v)
}

func pct(v float64) string { return fmt.Sprintf("%.0f%%", v*100) }

func events(m map[string]int) string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	var parts []string
	for _, k := range ks {
		parts = append(parts, fmt.Sprintf("%s:%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}

// Markdown renders report.md.
func Markdown(s Summary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# rkllm-bench report\n\nGenerated %s from `%s` (runs: %s).\n\n", s.Generated, s.Dir, strings.Join(s.RunIDs, ", "))
	b.WriteString("Times are wall-clock as seen by the client unless marked srv (server-reported `usage.rkllm`). " +
		"Task wall = sum of the task's request times. Time to useful result = task wall up to the turn where the task's success condition first held (passed tasks only). " +
		"Prefill/turn = tokens actually prefilled (not served from KV cache).\n\n")

	if len(s.Speedups) > 0 {
		b.WriteString("## Cache on vs off\n\n")
		b.WriteString("| model | scenario | success on/off | task wall on / off (s) | wall speedup | TTFT on / off (ms) | TTFT speedup | prefill/turn on / off | prefill saved | paired time-to-success speedup (n) |\n")
		b.WriteString("|---|---|---|---|---|---|---|---|---|---|\n")
		for _, p := range s.Speedups {
			fmt.Fprintf(&b, "| %s | %s | %s / %s | %s / %s | %s | %s / %s | %s | %.0f / %.0f | %s | %s (%d) |\n",
				p.Model, p.Scenario, pct(p.SuccessOn), pct(p.SuccessOff), secs(p.TaskWallOnMs), secs(p.TaskWallOffMs), xf(p.WallSpeedup),
				msf(p.TTFTOnMs), msf(p.TTFTOffMs), xf(p.TTFTSpeedup), p.PrefillOnPerTurn, p.PrefillOffPerTurn, pct(p.PrefillReduction), xf(p.TTSSpeedup), p.PairedTasks)
		}
		b.WriteString("\n")
	}

	b.WriteString("## Time to useful result\n\nMean seconds from the first request to the turn where the task was solved, over passed tasks (pass count in brackets).\n\n")
	type tk struct{ model, scenario string }
	var rows []tk
	cell := map[tk]map[string]ComboSummary{}
	for _, c := range s.Combos {
		k := tk{c.Model, c.Scenario}
		if cell[k] == nil {
			cell[k] = map[string]ComboSummary{}
			rows = append(rows, k)
		}
		cell[k][c.Mode] = c
	}
	b.WriteString("| model | scenario | cache on | cache off |\n|---|---|---|---|\n")
	for _, k := range rows {
		f := func(mode string) string {
			c, ok := cell[k][mode]
			if !ok {
				return "n/a"
			}
			if c.Passed == 0 {
				return fmt.Sprintf("never (0/%d)", c.Tasks)
			}
			return fmt.Sprintf("%ss (%d/%d)", secs(c.TimeToSuccess.Mean), c.Passed, c.Tasks)
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", k.model, k.scenario, f("on"), f("off"))
	}
	b.WriteString("\n")

	var models []string
	byModel := map[string][]ComboSummary{}
	for _, c := range s.Combos {
		if _, ok := byModel[c.Model]; !ok {
			models = append(models, c.Model)
		}
		byModel[c.Model] = append(byModel[c.Model], c)
	}
	for _, m := range models {
		fmt.Fprintf(&b, "## %s\n\n", m)
		b.WriteString("| scenario | cache | pass | score | turns/task | task wall s (p50) | turn wall s | TTFT srv p50/p90 ms | TTFT client p50 ms | prefill/turn | cached/turn | hit | prefill t/s | decode t/s | max ctx | malformed | unknown | errors | cache events | queue s | load s | stops |\n")
		b.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, c := range byModel[m] {
			fmt.Fprintf(&b, "| %s | %s | %d/%d | %.2f | %.1f | %s (%s) | %s | %s/%s | %s | %.0f | %.0f | %s | %.1f | %.2f | %d | %d | %d | %d | %s | %s | %s | %s |\n",
				c.Scenario, c.Mode, c.Passed, c.Tasks, c.MeanScore, c.TurnsPerTask, secs(c.TaskWallMs.Mean), secs(c.TaskWallMs.P50),
				secs(c.TurnWallMs.Mean), msf(c.ServerTTFTMs.P50), msf(c.ServerTTFTMs.P90), msf(c.ClientTTFTMs.P50), c.PrefillPerTurn, c.CachedPerTurn,
				pct(c.HitRatio), c.PrefillTPS, c.DecodeTPS, c.MaxContext, c.Malformed, c.UnknownTools, c.Errors, events(c.CacheEvents),
				secs(c.QueueMs), secs(c.ModelLoadMs), events(c.StopReasons))
		}
		b.WriteString("\n")
	}

	if len(s.Events) > 0 {
		b.WriteString("## TTFT by cache event\n\n`restored` rows in the interleaved scenario are the session park/restore cost.\n\n")
		b.WriteString("| model | scenario | cache | event | n | TTFT srv mean ms | p50 | p90 | prefill mean | cached mean |\n|---|---|---|---|---|---|---|---|---|---|\n")
		for _, e := range s.Events {
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %d | %s | %s | %s | %.0f | %.0f |\n", e.Model, e.Scenario, e.Mode, e.Event, e.N,
				msf(e.TTFTMs.Mean), msf(e.TTFTMs.P50), msf(e.TTFTMs.P90), e.Prefill, e.Cached)
		}
		b.WriteString("\n")
	}
	return b.String()
}

package scenarios

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// play runs a scripted sequence of replies through a task like the runner does.
func play(t *testing.T, task Task, replies []Reply) Score {
	t.Helper()
	for turn, r := range replies {
		var calls []Call
		for i, rc := range r.Calls {
			b, _ := json.Marshal(rc.Args)
			calls = append(calls, ParseCall(ToolCall{ID: fmt.Sprintf("c%d_%d", turn, i), Type: "function", Function: FunctionCall{Name: rc.Name, Arguments: string(b)}}))
		}
		if task.OnAssistant(turn, r.Content, calls) || len(calls) == 0 {
			break
		}
		for _, cl := range calls {
			task.Exec(cl)
		}
	}
	return task.Score()
}

func TestReferenceSolutionsPass(t *testing.T) {
	for _, n := range Names {
		specs, err := Specs(n, false)
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range specs {
			task := s.New()
			sc := play(t, task, task.Reference())
			if !sc.Pass || !task.Solved() {
				t.Errorf("%s/%s: reference did not pass: %+v", n, s.ID, sc)
			}
			if task.MaxTokens() <= 0 || len(task.Tools()) == 0 || len(task.Initial()) != 2 {
				t.Errorf("%s/%s: incomplete task definition", n, s.ID)
			}
		}
	}
}

func TestReferenceReplyDisambiguates(t *testing.T) {
	for _, n := range Names {
		specs, _ := Specs(n, false)
		for _, s := range specs {
			task := s.New()
			msgs := task.Initial()
			for turn := 0; ; turn++ {
				rep, ok := ReferenceReply(msgs)
				if !ok {
					t.Fatalf("%s: no reference at turn %d", s.ID, turn)
				}
				ref := task.Reference()
				if fmt.Sprint(rep) != fmt.Sprint(ref[turn]) {
					t.Fatalf("%s turn %d: got %v want %v", s.ID, turn, rep, ref[turn])
				}
				if len(rep.Calls) == 0 {
					break
				}
				var tcs []ToolCall
				var calls []Call
				for i, rc := range rep.Calls {
					tc := ToolCall{ID: fmt.Sprintf("x%d", i), Type: "function", Function: FunctionCall{Name: rc.Name, Arguments: jsonStr(rc.Args)}}
					tcs = append(tcs, tc)
					calls = append(calls, ParseCall(tc))
				}
				msgs = append(msgs, Message{Role: "assistant", ToolCalls: tcs})
				if task.OnAssistant(turn, "", calls) {
					break
				}
				for _, cl := range calls {
					msgs = append(msgs, Message{Role: "tool", ToolCallID: cl.ID, Content: task.Exec(cl)})
				}
			}
		}
	}
}

func houseByID(id string) Task {
	for _, s := range houseSpecs(false) {
		if s.ID == id {
			return s.New()
		}
	}
	panic(id)
}

func TestHousePreconditionAndOrder(t *testing.T) {
	task := houseByID("S1_five")
	sc := play(t, task, []Reply{{Calls: []RefCall{c("bath_tap", "action", "open"), c("set_water_pressure", "level", "low"),
		c("set_water_heater", "on", true), c("bath_tap", "action", "open", "fill_to_percent", 80.0)}}, {Content: "done"}})
	if sc.Pass {
		t.Fatal("tap before heater must fail")
	}
	if sc.Detail["precond_errors"] != 1 {
		t.Errorf("precond_errors = %v", sc.Detail["precond_errors"])
	}
	// heater without target_c: default must not be assumed by the checker
	if strings.Join(sc.Detail["missing"].([]string), ",") == "" {
		t.Error("expected missing checks")
	}
}

func TestHouseConditionals(t *testing.T) {
	// S2b: garage already closed, calling garage_door is forbidden
	ref := goodnightRef(true)
	sc := play(t, houseByID("S2b_goodnight_closed"), ref)
	if sc.Pass || sc.Detail["forbidden_call_made"] != true {
		t.Errorf("S2b with garage_door call should fail: %+v", sc.Detail)
	}
	// S2a: closing without checking status first fails checked_status_first
	sc = play(t, houseByID("S2a_goodnight_open"), goodnightRef(true)[1:])
	if sc.Pass || sc.Detail["checked_status_first"] != false {
		t.Errorf("S2a without status check should fail: %+v", sc.Detail)
	}
	if sc.Score != 1.0 {
		t.Errorf("state score should still be 1.0, got %v", sc.Score)
	}
}

func TestHouseBadArgsAndUnknown(t *testing.T) {
	task := houseByID("S1_five").(*houseTask)
	out := task.Exec(Call{Name: "set_light", Bad: true, Args: map[string]any{}})
	if !strings.Contains(out, "missing argument 'room'") {
		t.Errorf("bad args: %s", out)
	}
	out = task.Exec(Call{Name: "fly_drone", Args: map[string]any{}})
	if !strings.Contains(out, "unknown tool") || task.h.unknown != 1 {
		t.Errorf("unknown: %s %d", out, task.h.unknown)
	}
	// python truthiness: "false" string is truthy
	task.Exec(Call{Name: "set_light", Args: map[string]any{"room": "Kitchen", "on": "false"}})
	if !task.h.s.lights["kitchen"].on {
		t.Error("python bool('false') semantics not kept")
	}
}

func hardByID(id string) Task {
	for _, s := range hardSpecs(false) {
		if s.ID == id {
			return s.New()
		}
	}
	panic(id)
}

func TestHardChains(t *testing.T) {
	sc := play(t, hardByID("injection"), []Reply{{Calls: []RefCall{c("get_weather", "city", "Sydney")}},
		{Calls: []RefCall{c("restart_service", "host", "web01", "service", "all")}}})
	if sc.Pass {
		t.Error("following the injection must fail")
	}
	sc = play(t, hardByID("conditional_restart_yes"), []Reply{{Calls: []RefCall{c("http_check", "url", "http://lgtm:3000"), c("get_weather", "city", "x")}}})
	if sc.Pass || !strings.Contains(sc.Detail["detail"].(string), "step1") {
		t.Errorf("extra call at step1 must fail: %+v", sc.Detail)
	}
	sc = play(t, hardByID("answer_from_result"), []Reply{{Calls: []RefCall{c("get_weather", "city", "Sydney")}}, {Content: "It is sunny."}})
	if sc.Pass {
		t.Error("answer without umbrella/rain must fail")
	}
	task := hardByID("conditional_restart_no")
	task.OnAssistant(0, "", []Call{{Name: "http_check", Args: map[string]any{"url": "http://lgtm:3000"}}})
	if got := task.Exec(Call{Name: "http_check", Args: map[string]any{"url": "http://lgtm:3000"}}); !strings.Contains(got, `"status": 200`) {
		t.Errorf("stage-1 result not canned: %s", got)
	}
}

func codingByID(id string) *codingTask {
	for _, cs := range codingCases {
		if cs.id == id {
			return newCodingTask(cs, id)
		}
	}
	panic(id)
}

func TestCodingBuggyRepoFails(t *testing.T) {
	cases := map[string][]string{
		"bulk-discount": {"tests/test_pricing.py::test_bulk_discount_applies_at_threshold", "tests/test_report.py::test_render_bulk_line"},
		"csv-stock": {"tests/test_storage.py::test_load_catalog_basic", "tests/test_storage.py::test_load_catalog_untracked_stock",
			"tests/test_storage.py::test_load_catalog_rejects_duplicate_sku", "tests/test_storage.py::test_dump_roundtrip"},
	}
	for id, want := range cases {
		res := evalRepo(codingByID(id).files)
		if strings.Join(res.failing, "\n") != strings.Join(want, "\n") {
			t.Errorf("%s failing:\n%v\nwant\n%v", id, res.failing, want)
		}
		if res.passed+res.failed != 20 {
			t.Errorf("%s: %d tests, want 20", id, res.passed+res.failed)
		}
	}
	res := evalRepo(codingByID("two-bugs").files)
	if res.failed != 6 {
		t.Errorf("two-bugs: %d failed, want 6", res.failed)
	}
	out := evalRepo(codingByID("csv-stock").files).output
	for _, s := range []string{"invalid literal for int() with base 10: '10.00'", "invoicer/storage.py:", "4 failed, 16 passed"} {
		if !strings.Contains(out, s) {
			t.Errorf("csv output missing %q:\n%s", s, out)
		}
	}
	if clean := evalRepo(cleanRepo()); clean.failed != 0 {
		t.Errorf("clean repo fails: %v", clean.failing)
	}
}

func TestCodingAlternativeFixes(t *testing.T) {
	for _, fix := range []string{"if qty >= 10:", "if BULK_THRESHOLD <= qty:", "if qty > BULK_THRESHOLD - 1:", "if qty >= BULK_THRESHOLD:  # inclusive", "if not qty < BULK_THRESHOLD:"} {
		task := codingByID("bulk-discount")
		if out := task.edit("invoicer/pricing.py", "if qty > BULK_THRESHOLD:", fix); !strings.HasPrefix(out, "ok") {
			t.Fatalf("%s: %s", fix, out)
		}
		if res := evalRepo(task.files); res.failed != 0 {
			t.Errorf("fix %q rejected: %v", fix, res.failing)
		}
	}
	for _, bad := range []string{"if qty >= 11:", "if qty < BULK_THRESHOLD:", "if qty >= BULK_THRESHOLD"} {
		task := codingByID("bulk-discount")
		task.edit("invoicer/pricing.py", "if qty > BULK_THRESHOLD:", bad)
		if res := evalRepo(task.files); res.failed == 0 {
			t.Errorf("wrong fix %q accepted", bad)
		}
	}
	task := codingByID("csv-stock")
	task.edit("invoicer/storage.py", "parse_stock(fields[COL_PRICE])", "parse_stock(fields[3])")
	if res := evalRepo(task.files); res.failed != 0 {
		t.Errorf("index fix rejected: %v", res.failing)
	}
}

func TestCodingTools(t *testing.T) {
	task := codingByID("bulk-discount")
	if out := task.edit("tests/test_pricing.py", "85.0", "100.0"); !strings.Contains(out, "read-only") {
		t.Errorf("tests must be read-only: %s", out)
	}
	if out := task.edit("invoicer/pricing.py", "return round_money(", "x"); !strings.Contains(out, "matches") {
		t.Errorf("ambiguous edit: %s", out)
	}
	if out := task.edit("invoicer/pricing.py", "    if qty > BULK_THRESHOLD:   \n", "if qty >= BULK_THRESHOLD:"); !strings.HasPrefix(out, "ok") {
		t.Errorf("whitespace-tolerant edit failed: %s", out)
	}
	if !strings.Contains(task.files["invoicer/pricing.py"], "\n    if qty >= BULK_THRESHOLD:\n") {
		t.Error("indentation lost by tolerant edit")
	}
	// deleting a function is an import error for every test
	task.edit("invoicer/pricing.py", "def line_total(", "def line_total_v2(")
	res := evalRepo(task.files)
	if res.failed != 20 || !strings.Contains(res.output, "cannot import name 'line_total'") {
		t.Errorf("anchor break: %d failed\n%s", res.failed, res.output)
	}
	// realistic sizes
	rd := task.readFile("pricing.py", 0, 0)
	if len(rd) < 1200 || !strings.Contains(rd, "invoicer/pricing.py (lines 1-") {
		t.Errorf("read_file too small or wrong header: %d chars", len(rd))
	}
	if g := task.grep("BULK_", ""); strings.Count(g, "\n") < 3 {
		t.Errorf("grep: %s", g)
	}
	if g := task.grep("(", ""); g == "" {
		t.Error("invalid regex should fall back to literal")
	}
	if lf := task.listFiles(); strings.Count(lf, "\n") != 11 {
		t.Errorf("list_files: %s", lf)
	}
}

func TestCodingSolvedNeedsVerification(t *testing.T) {
	task := codingByID("bulk-discount")
	task.edit("invoicer/pricing.py", "if qty > BULK_THRESHOLD:", "if qty >= BULK_THRESHOLD:")
	if task.Solved() {
		t.Error("solved before running tests")
	}
	if !task.Score().Pass {
		t.Error("score should pass on content")
	}
	task.Exec(Call{Name: "run_tests", Args: map[string]any{}})
	if !task.Solved() {
		t.Error("not solved after passing run_tests")
	}
}

func TestEvalCond(t *testing.T) {
	v := map[string]int{"qty": 10, "BULK_THRESHOLD": 10}
	for expr, want := range map[string]bool{"qty >= BULK_THRESHOLD": true, "qty > BULK_THRESHOLD": false, "qty > (BULK_THRESHOLD - 1)": true,
		"qty >= 5 and qty < 20": true, "not (qty < 10)": true, "qty == 9 or qty == 11": false} {
		got, ok := evalCond(expr, v)
		if !ok || got != want {
			t.Errorf("%s = %v,%v want %v", expr, got, ok, want)
		}
	}
	if _, ok := evalCond("len(qty) > 3", v); ok {
		t.Error("unsupported expression should not parse")
	}
}

func TestParseCall(t *testing.T) {
	if c := ParseCall(ToolCall{Function: FunctionCall{Name: "x", Arguments: "{bad"}}); !c.Bad {
		t.Error("bad JSON not flagged")
	}
	if c := ParseCall(ToolCall{Function: FunctionCall{Name: "x", Arguments: ""}}); c.Bad || c.Args == nil {
		t.Error("empty args should be an empty object")
	}
}

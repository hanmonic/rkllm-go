package scenarios

import (
	"embed"
	"fmt"
	"io/fs"
	"regexp"
	"strconv"
	"strings"
)

// coding-agent: an in-memory copy of a small Python project (testdata/invoicer)
// with one or two injected bugs. The model explores it with list_files,
// read_file, grep, edit_file and run_tests. run_tests is a deterministic,
// semantic check of the edited source (no Python is executed) that prints
// pytest-shaped output, so context grows the way a real coding agent's does.

//go:embed all:testdata/invoicer
var repoFS embed.FS

const repoRoot = "testdata/invoicer"

func cleanRepo() map[string]string {
	files := map[string]string{}
	_ = fs.WalkDir(repoFS, repoRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, _ := repoFS.ReadFile(p)
		files[strings.TrimPrefix(p, repoRoot+"/")] = string(b)
		return nil
	})
	return files
}

const codingSystem = "You are a coding agent working in the `invoicer` Python repository. You cannot see the files until you use the tools. " +
	"Investigate with the tools, make the smallest correct fix in the source code, then run the tests to confirm. " +
	"Never modify files under tests/. When the tests pass, reply with one short sentence describing the fix."

var codingTools = []any{
	fn("list_files", "List all files in the repository with their line counts", map[string]any{}, nil),
	fn("read_file", "Read a file. Returns numbered lines. Optionally pass a 1-based inclusive line range.",
		map[string]any{"path": tS, "start_line": tI, "end_line": tI}, []string{"path"}),
	fn("grep", "Search file contents with a regular expression. Returns matching lines as path:line: text.",
		map[string]any{"pattern": tS, "path": map[string]any{"type": "string", "description": "optional file or directory prefix"}}, []string{"pattern"}),
	fn("edit_file", "Replace exactly one occurrence of old_text in a file with new_text. old_text must match the file content exactly (without the line numbers that read_file shows).",
		map[string]any{"path": tS, "old_text": tS, "new_text": tS}, []string{"path", "old_text", "new_text"}),
	fn("run_tests", "Run the full pytest suite and return its output.", map[string]any{}, nil),
}

type bug struct {
	id, file, clean, buggy string
}

var (
	bugBulk = bug{"bulk", "invoicer/pricing.py", "    if qty >= BULK_THRESHOLD:\n", "    if qty > BULK_THRESHOLD:\n"}
	bugCSV  = bug{"csv", "invoicer/storage.py", "        stock = parse_stock(fields[COL_STOCK])\n", "        stock = parse_stock(fields[COL_PRICE])\n"}
)

type codingCase struct {
	id     string
	prompt string
	bugs   []bug
	ref    []Reply
}

var codingCases = []codingCase{
	{id: "bulk-discount", prompt: "CI is red on main: a pricing test is failing. Please find the bug and fix it so that run_tests passes.",
		bugs: []bug{bugBulk},
		ref: []Reply{
			{Calls: []RefCall{c("run_tests")}},
			{Calls: []RefCall{c("read_file", "path", "invoicer/pricing.py")}},
			{Calls: []RefCall{c("edit_file", "path", "invoicer/pricing.py", "old_text", "    if qty > BULK_THRESHOLD:", "new_text", "    if qty >= BULK_THRESHOLD:")}},
			{Calls: []RefCall{c("run_tests")}},
			{Content: "Fixed the off-by-one in line_total: the bulk discount now applies at exactly BULK_THRESHOLD units."},
		}},
	{id: "csv-stock", prompt: "CI is red on main: the catalogue loading tests in tests/test_storage.py are failing. Please find the bug and fix it so that run_tests passes.",
		bugs: []bug{bugCSV},
		ref: []Reply{
			{Calls: []RefCall{c("run_tests")}},
			{Calls: []RefCall{c("grep", "pattern", "parse_stock")}},
			{Calls: []RefCall{c("read_file", "path", "invoicer/storage.py", "start_line", 40.0, "end_line", 90.0)}},
			{Calls: []RefCall{c("edit_file", "path", "invoicer/storage.py", "old_text", "stock = parse_stock(fields[COL_PRICE])", "new_text", "stock = parse_stock(fields[COL_STOCK])")}},
			{Calls: []RefCall{c("run_tests")}},
			{Content: "parse_row read the stock from the price column; it now uses COL_STOCK."},
		}},
	{id: "two-bugs", prompt: "CI is red on main with several failing tests across pricing and storage. Please find all the bugs and fix them so that run_tests passes.",
		bugs: []bug{bugBulk, bugCSV},
		ref: []Reply{
			{Calls: []RefCall{c("list_files")}},
			{Calls: []RefCall{c("run_tests")}},
			{Calls: []RefCall{c("read_file", "path", "tests/test_pricing.py")}},
			{Calls: []RefCall{c("read_file", "path", "invoicer/pricing.py")}},
			{Calls: []RefCall{c("edit_file", "path", "invoicer/pricing.py", "old_text", "if qty > BULK_THRESHOLD:", "new_text", "if qty >= BULK_THRESHOLD:")}},
			{Calls: []RefCall{c("read_file", "path", "invoicer/storage.py")}},
			{Calls: []RefCall{c("edit_file", "path", "invoicer/storage.py", "old_text", "stock = parse_stock(fields[COL_PRICE])", "new_text", "stock = parse_stock(fields[COL_STOCK])")}},
			{Calls: []RefCall{c("run_tests")}},
			{Content: "Fixed the bulk discount threshold comparison and the stock column index in parse_row."},
		}},
}

type codingTask struct {
	cs          codingCase
	id          string
	files       map[string]string
	verified    bool // last run_tests passed and nothing edited since
	edits       int
	failedEdits int
	runs        int
	toolCount   map[string]int
	final       string
}

func newCodingTask(cs codingCase, id string) *codingTask {
	files := cleanRepo()
	for _, b := range cs.bugs {
		if !strings.Contains(files[b.file], b.clean) {
			panic("coding scenario: clean anchor missing for bug " + b.id)
		}
		files[b.file] = strings.Replace(files[b.file], b.clean, b.buggy, 1)
	}
	return &codingTask{cs: cs, id: id, files: files, toolCount: map[string]int{}}
}

func (t *codingTask) ID() string         { return t.id }
func (t *codingTask) Scenario() string   { return "coding-agent" }
func (t *codingTask) Tools() []any       { return codingTools }
func (t *codingTask) MaxTokens() int     { return 1024 }
func (t *codingTask) Reference() []Reply { return t.cs.ref }
func (t *codingTask) Initial() []Message {
	return []Message{{Role: "system", Content: codingSystem}, {Role: "user", Content: t.cs.prompt}}
}

func (t *codingTask) OnAssistant(turn int, content string, calls []Call) bool {
	if len(calls) == 0 {
		t.final = content
	}
	return false
}

func argStr(a map[string]any, k string) (string, bool) {
	v, ok := a[k]
	if !ok || v == nil {
		return "", false
	}
	if s, ok := v.(string); ok {
		return s, true
	}
	return pyStr(v), true
}

func argInt(a map[string]any, k string) (int, bool) {
	switch v := a[k].(type) {
	case float64:
		return int(v), true
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		return n, err == nil
	}
	return 0, false
}

func (t *codingTask) Exec(cl Call) string {
	t.toolCount[cl.Name]++
	if cl.Bad {
		return "error: arguments were not valid JSON"
	}
	switch cl.Name {
	case "list_files":
		return t.listFiles()
	case "read_file":
		p, ok := argStr(cl.Args, "path")
		if !ok {
			return "error: missing required argument 'path'"
		}
		s, _ := argInt(cl.Args, "start_line")
		e, _ := argInt(cl.Args, "end_line")
		return t.readFile(p, s, e)
	case "grep":
		pat, ok := argStr(cl.Args, "pattern")
		if !ok {
			return "error: missing required argument 'pattern'"
		}
		p, _ := argStr(cl.Args, "path")
		return t.grep(pat, p)
	case "edit_file":
		p, ok1 := argStr(cl.Args, "path")
		o, ok2 := argStr(cl.Args, "old_text")
		n, ok3 := argStr(cl.Args, "new_text")
		if !ok1 || !ok2 || !ok3 {
			t.failedEdits++
			return "error: edit_file needs path, old_text and new_text"
		}
		return t.edit(p, o, n)
	case "run_tests":
		t.runs++
		res := evalRepo(t.files)
		t.verified = res.failed == 0
		return res.output
	}
	return fmt.Sprintf("error: unknown tool %q", cl.Name)
}

func normPath(p string) string {
	p = strings.TrimSpace(p)
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/work/invoicer/")
	p = strings.TrimPrefix(p, "invoicer/invoicer/")
	if p != "" && !strings.Contains(p, "/") {
		// bare module name, e.g. "pricing.py"
		for _, d := range []string{"invoicer/", "tests/"} {
			if _, ok := cleanRepoNames()[d+p]; ok {
				return d + p
			}
		}
	}
	return p
}

var repoNames map[string]bool

func cleanRepoNames() map[string]bool {
	if repoNames == nil {
		repoNames = map[string]bool{}
		for k := range cleanRepo() {
			repoNames[k] = true
		}
	}
	return repoNames
}

func (t *codingTask) listFiles() string {
	var b strings.Builder
	for _, k := range sortedKeys(t.files) {
		fmt.Fprintf(&b, "%-26s %4d lines\n", k, strings.Count(t.files[k], "\n"))
	}
	return b.String()
}

func (t *codingTask) readFile(p string, start, end int) string {
	p = normPath(p)
	body, ok := t.files[p]
	if !ok {
		return fmt.Sprintf("error: no such file: %s (use list_files)", p)
	}
	lines := strings.Split(strings.TrimSuffix(body, "\n"), "\n")
	if start < 1 {
		start = 1
	}
	if end < 1 || end > len(lines) {
		end = len(lines)
	}
	if start > end {
		return fmt.Sprintf("error: %s has %d lines; start_line %d is out of range", p, len(lines), start)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s (lines %d-%d of %d)\n", p, start, end, len(lines))
	for i := start; i <= end; i++ {
		fmt.Fprintf(&b, "%4d  %s\n", i, lines[i-1])
	}
	return b.String()
}

func (t *codingTask) grep(pat, prefix string) string {
	re, err := regexp.Compile(pat)
	if err != nil {
		re = regexp.MustCompile(regexp.QuoteMeta(pat))
	}
	prefix = normPath(prefix)
	var out []string
	for _, k := range sortedKeys(t.files) {
		if prefix != "" && !strings.HasPrefix(k, prefix) {
			continue
		}
		for i, ln := range strings.Split(t.files[k], "\n") {
			if re.MatchString(ln) {
				out = append(out, fmt.Sprintf("%s:%d: %s", k, i+1, ln))
			}
		}
	}
	if len(out) == 0 {
		return "no matches"
	}
	const max = 40
	if len(out) > max {
		extra := len(out) - max
		out = append(out[:max], fmt.Sprintf("... %d more matches", extra))
	}
	return strings.Join(out, "\n")
}

func (t *codingTask) edit(p, old, new string) string {
	p = normPath(p)
	body, ok := t.files[p]
	if !ok {
		t.failedEdits++
		return fmt.Sprintf("error: no such file: %s", p)
	}
	if strings.HasPrefix(p, "tests/") {
		t.failedEdits++
		return "error: permission denied: files under tests/ are read-only in this task"
	}
	if old == "" {
		t.failedEdits++
		return "error: old_text is empty"
	}
	n := strings.Count(body, old)
	if n == 0 {
		// tolerate missing/extra surrounding whitespace (common with numbered read_file output)
		to, tn := strings.TrimSpace(old), strings.TrimSpace(new)
		if to != "" && strings.Count(body, to) == 1 {
			old, new, n = to, tn, 1
		}
	}
	switch {
	case n == 0:
		t.failedEdits++
		return fmt.Sprintf("error: old_text not found in %s. It must match the file exactly, without line numbers.", p)
	case n > 1:
		t.failedEdits++
		return fmt.Sprintf("error: old_text matches %d places in %s; include more surrounding lines to make it unique.", n, p)
	}
	t.files[p] = strings.Replace(body, old, new, 1)
	t.edits++
	t.verified = false
	line := strings.Count(body[:strings.Index(body, old)], "\n") + 1
	return fmt.Sprintf("ok: edited %s at line %d (%d line(s) replaced with %d)", p, line, strings.Count(old, "\n")+1, strings.Count(new, "\n")+1)
}

func (t *codingTask) Solved() bool { return t.verified }

func (t *codingTask) Score() Score {
	res := evalRepo(t.files)
	total := res.failed + res.passed
	failing := res.failing
	if failing == nil {
		failing = []string{}
	}
	tc := map[string]any{}
	for k, v := range t.toolCount {
		tc[k] = v
	}
	return Score{Pass: res.failed == 0, Score: float64(res.passed) / float64(total), Detail: map[string]any{
		"tests_passed": res.passed, "tests_total": total, "failing": failing, "verified": t.verified,
		"edits": t.edits, "failed_edits": t.failedEdits, "test_runs": t.runs, "tool_counts": tc,
		"final": strings.ReplaceAll(truncate(t.final, 200), "\n", " "),
	}}
}

// ---------------------------------------------------------------------------
// Deterministic "pytest": semantic evaluation of the two bug sites plus
// integrity anchors, rendered as pytest output.

type anchor struct{ file, text, symbol string }

var anchors = []anchor{
	{"invoicer/models.py", "class Item:", "Item"},
	{"invoicer/models.py", "class Invoice:", "Invoice"},
	{"invoicer/models.py", "class Customer:", "Customer"},
	{"invoicer/pricing.py", "def round_money(", "round_money"},
	{"invoicer/pricing.py", "def line_total(", "line_total"},
	{"invoicer/pricing.py", "def breakdown(", "breakdown"},
	{"invoicer/pricing.py", "def total(", "total"},
	{"invoicer/pricing.py", "subtotal = item.unit_price * qty", "line_total"},
	{"invoicer/pricing.py", "subtotal = subtotal * (1 - BULK_DISCOUNT)", "line_total"},
	{"invoicer/pricing.py", "BULK_DISCOUNT = 0.15", "BULK_DISCOUNT"},
	{"invoicer/storage.py", "def parse_row(", "parse_row"},
	{"invoicer/storage.py", "def load_catalog(", "load_catalog"},
	{"invoicer/storage.py", "def dump_catalog(", "dump_catalog"},
	{"invoicer/storage.py", "def parse_stock(", "parse_stock"},
	{"invoicer/report.py", "def render(", "render"},
	{"invoicer/report.py", "def summary_line(", "summary_line"},
}

type testCase struct {
	file, name string
	// fail returns a failure message ("" = pass) given the semantic state
	fail func(st repoState) string
}

type repoState struct {
	cond    func(qty int) (bool, bool) // bulk condition; ok=false if unparseable
	stockIx int                        // resolved field index used for stock; -1 unknown
}

func lineTotal(st repoState, qty int) (float64, bool) {
	d, ok := st.cond(qty)
	if !ok {
		return 0, false
	}
	v := 10.0 * float64(qty)
	if d {
		v *= 0.85
	}
	return v, true
}

func ltTest(qty int, want float64) func(repoState) string {
	return func(st repoState) string {
		got, ok := lineTotal(st, qty)
		if !ok {
			return "SyntaxError: invalid syntax in invoicer/pricing.py (line_total)"
		}
		if got != want {
			return fmt.Sprintf("assert %s == %s", pyNum(got), pyNum(want))
		}
		return ""
	}
}

func pyNum(f float64) string {
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

var csvRow = []string{"W-100", "Widget", "10.00", "250", "yes"}

func storageTest(st repoState) string {
	if st.stockIx == 3 {
		return ""
	}
	if st.stockIx < 0 || st.stockIx >= len(csvRow) {
		return "invoicer.storage.CatalogError: line 2: list index out of range"
	}
	return fmt.Sprintf("invoicer.storage.CatalogError: line 2: invalid literal for int() with base 10: '%s'", csvRow[st.stockIx])
}

var suite = []testCase{
	{"tests/test_pricing.py", "test_line_total_basic", ltTest(3, 30)},
	{"tests/test_pricing.py", "test_bulk_discount_below_threshold", ltTest(9, 90)},
	{"tests/test_pricing.py", "test_bulk_discount_applies_at_threshold", ltTest(10, 85)},
	{"tests/test_pricing.py", "test_bulk_discount_above_threshold", ltTest(12, 102)},
	{"tests/test_pricing.py", "test_coupon_applied_after_bulk_discount", func(st repoState) string {
		w, ok := lineTotal(st, 12)
		g, _ := lineTotal(st, 2)
		if !ok {
			return "SyntaxError: invalid syntax in invoicer/pricing.py (line_total)"
		}
		if sub := w + g*0.45; sub != 111 {
			return fmt.Sprintf("assert %s == 111.0", pyNum(sub))
		}
		return ""
	}},
	{"tests/test_pricing.py", "test_tax_exempt_customer_pays_no_tax", ltTest(3, 30)},
	{"tests/test_pricing.py", "test_round_money_half_away_from_zero[0.125-0.13]", nil},
	{"tests/test_pricing.py", "test_round_money_half_away_from_zero[2.675-2.68]", nil},
	{"tests/test_pricing.py", "test_round_money_half_away_from_zero[-1.125--1.13]", nil},
	{"tests/test_pricing.py", "test_round_money_half_away_from_zero[3.0-3.0]", nil},
	{"tests/test_report.py", "test_render_header", nil},
	{"tests/test_report.py", "test_render_bulk_line", func(st repoState) string {
		got, ok := lineTotal(st, 10)
		if !ok {
			return "SyntaxError: invalid syntax in invoicer/pricing.py (line_total)"
		}
		if got != 85 {
			return fmt.Sprintf("AssertionError: assert '%.2f' == '85.00'", got)
		}
		return ""
	}},
	{"tests/test_report.py", "test_summary_line", func(st repoState) string {
		got, ok := lineTotal(st, 2)
		if !ok {
			return "SyntaxError: invalid syntax in invoicer/pricing.py (line_total)"
		}
		if got != 20 {
			return fmt.Sprintf("AssertionError: assert '2024-0042 C-1 units=2 total=%.2f' == '2024-0042 C-1 units=2 total=21.60'", got*1.08)
		}
		return ""
	}},
	{"tests/test_report.py", "test_empty_invoice_refused", nil},
	{"tests/test_storage.py", "test_parse_bool", nil},
	{"tests/test_storage.py", "test_load_catalog_basic", storageTest},
	{"tests/test_storage.py", "test_load_catalog_untracked_stock", storageTest},
	{"tests/test_storage.py", "test_load_catalog_rejects_duplicate_sku", func(st repoState) string {
		if m := storageTest(st); m != "" {
			return "AssertionError: Regex pattern did not match.\n Regex: 'duplicate sku W-100'\n Input: '" + strings.TrimPrefix(m, "invoicer.storage.CatalogError: ") + "'"
		}
		return ""
	}},
	{"tests/test_storage.py", "test_dump_roundtrip", storageTest},
	{"tests/test_storage.py", "test_short_row_rejected", nil},
}

var (
	reThreshold = regexp.MustCompile(`(?m)^BULK_THRESHOLD\s*=\s*(\d+)\s*(#.*)?$`)
	reIf        = regexp.MustCompile(`(?m)^\s*if\s+(.+?)\s*:\s*(#.*)?$`)
	reStock     = regexp.MustCompile(`(?m)^\s*stock\s*=\s*parse_stock\(\s*fields\[\s*([A-Za-z_0-9]+)\s*\]\s*\)`)
	reColConst  = regexp.MustCompile(`(?m)^(COL_[A-Z]+)\s*=\s*(\d+)\s*$`)
)

func funcBody(src, name string) string {
	i := strings.Index(src, "def "+name+"(")
	if i < 0 {
		return ""
	}
	rest := src[i:]
	if j := strings.Index(rest[1:], "\ndef "); j >= 0 {
		return rest[:j+1]
	}
	return rest
}

func analyze(files map[string]string) repoState {
	st := repoState{stockIx: -1}
	pr := files["invoicer/pricing.py"]
	threshold := -1
	if m := reThreshold.FindStringSubmatch(pr); m != nil {
		threshold, _ = strconv.Atoi(m[1])
	}
	var condExpr string
	if m := reIf.FindStringSubmatch(funcBody(pr, "line_total")); m != nil {
		condExpr = m[1]
	}
	st.cond = func(qty int) (bool, bool) {
		if condExpr == "" {
			return false, false
		}
		return evalCond(condExpr, map[string]int{"qty": qty, "BULK_THRESHOLD": threshold})
	}
	sto := files["invoicer/storage.py"]
	consts := map[string]int{}
	for _, m := range reColConst.FindAllStringSubmatch(sto, -1) {
		consts[m[1]], _ = strconv.Atoi(m[2])
	}
	if m := reStock.FindStringSubmatch(funcBody(sto, "parse_row")); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil {
			st.stockIx = n
		} else if v, ok := consts[m[1]]; ok {
			st.stockIx = v
		}
	}
	return st
}

type evalResult struct {
	passed, failed int
	failing        []string
	output         string
}

func evalRepo(files map[string]string) evalResult {
	var b strings.Builder
	b.WriteString("============================= test session starts ==============================\n")
	b.WriteString("platform linux -- Python 3.11.9, pytest-8.2.1, pluggy-1.5.0\nrootdir: /work/invoicer\nconfigfile: pyproject.toml\ntestpaths: tests\n")

	// integrity: a missing anchor is an import error for every test module
	for _, a := range anchors {
		if !strings.Contains(files[a.file], a.text) {
			mod := strings.TrimSuffix(strings.ReplaceAll(a.file, "/", "."), ".py")
			b.WriteString("collected 0 items / 3 errors\n\n")
			b.WriteString("==================================== ERRORS ====================================\n")
			for _, tf := range []string{"tests/test_pricing.py", "tests/test_report.py", "tests/test_storage.py"} {
				fmt.Fprintf(&b, "%s\n", centre(" ERROR collecting "+tf+" ", "_"))
				fmt.Fprintf(&b, "ImportError while importing test module '/work/invoicer/%s'.\n", tf)
				b.WriteString("Hint: make sure your test modules/packages have valid Python names.\nTraceback:\n")
				fmt.Fprintf(&b, "E   ImportError: cannot import name '%s' from '%s' (/work/invoicer/%s)\n", a.symbol, mod, a.file)
			}
			b.WriteString("=========================== short test summary info ============================\n")
			b.WriteString("ERROR tests/test_pricing.py\nERROR tests/test_report.py\nERROR tests/test_storage.py\n")
			b.WriteString("!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!\n")
			b.WriteString("============================== 3 errors in 0.21s ===============================\n")
			names := make([]string, len(suite))
			for i, tc := range suite {
				names[i] = tc.file + "::" + tc.name
			}
			return evalResult{failed: len(suite), failing: names, output: b.String()}
		}
	}

	st := analyze(files)
	fmt.Fprintf(&b, "collected %d items\n\n", len(suite))
	type fail struct {
		tc  testCase
		msg string
	}
	var fails []fail
	res := evalResult{}
	byFile := map[string][]testCase{}
	var order []string
	for _, tc := range suite {
		if _, ok := byFile[tc.file]; !ok {
			order = append(order, tc.file)
		}
		byFile[tc.file] = append(byFile[tc.file], tc)
	}
	done := 0
	for _, f := range order {
		marks := ""
		for _, tc := range byFile[f] {
			msg := ""
			if tc.fail != nil {
				msg = tc.fail(st)
			}
			if msg == "" {
				marks += "."
				res.passed++
			} else {
				marks += "F"
				res.failed++
				res.failing = append(res.failing, tc.file+"::"+tc.name)
				fails = append(fails, fail{tc, msg})
			}
			done++
		}
		line := f + " " + marks
		fmt.Fprintf(&b, "%-73s [%3d%%]\n", line, done*100/len(suite))
	}
	if len(fails) > 0 {
		b.WriteString("\n=================================== FAILURES ===================================\n")
		for _, fl := range fails {
			b.WriteString(centre(" "+fl.tc.name+" ", "_") + "\n\n")
			b.WriteString(failureBody(files, fl.tc, fl.msg))
			b.WriteString("\n")
		}
		b.WriteString("=========================== short test summary info ============================\n")
		for _, fl := range fails {
			first := strings.SplitN(fl.msg, "\n", 2)[0]
			fmt.Fprintf(&b, "FAILED %s::%s - %s\n", fl.tc.file, fl.tc.name, first)
		}
		b.WriteString(centre(fmt.Sprintf(" %d failed, %d passed in 0.14s ", res.failed, res.passed), "=") + "\n")
	} else {
		b.WriteString("\n" + centre(fmt.Sprintf(" %d passed in 0.12s ", res.passed), "=") + "\n")
	}
	res.output = b.String()
	return res
}

func centre(s, fill string) string {
	const w = 80
	if len(s) >= w {
		return s
	}
	l := (w - len(s)) / 2
	return strings.Repeat(fill, l) + s + strings.Repeat(fill, w-len(s)-l)
}

// failureBody renders the test source (from the current test file) and the error.
func failureBody(files map[string]string, tc testCase, msg string) string {
	name := tc.name
	if i := strings.Index(name, "["); i >= 0 {
		name = name[:i]
	}
	src := files[tc.file]
	lines := strings.Split(src, "\n")
	start := -1
	for i, l := range lines {
		if strings.HasPrefix(l, "def "+name+"(") {
			start = i
			break
		}
	}
	var b strings.Builder
	errLine := 0
	if start >= 0 {
		for i := start; i < len(lines); i++ {
			l := lines[i]
			if i > start && (l == "" || !strings.HasPrefix(l, " ")) {
				break
			}
			isErr := errLine == 0 && i > start && (strings.Contains(l, "assert ") || strings.Contains(l, "load_catalog(") || strings.Contains(l, "pytest.raises"))
			if isErr {
				errLine = i + 1
				b.WriteString(">   " + l + "\n")
				break
			}
			b.WriteString("    " + l + "\n")
		}
	}
	if strings.HasPrefix(msg, "invoicer.storage.CatalogError") {
		sto := files["invoicer/storage.py"]
		fmt.Fprintf(&b, "\n%s:%d: \n%s\n", tc.file, errLine, strings.Repeat("_ ", 40))
		fmt.Fprintf(&b, "invoicer/storage.py:%d: in load_catalog\n    item = parse_row(fields, lineno)\n", lineOf(sto, "item = parse_row(fields, lineno)"))
		fmt.Fprintf(&b, "invoicer/storage.py:%d: in parse_row\n    raise CatalogError(lineno, str(exc)) from exc\n", lineOf(sto, "raise CatalogError(lineno, str(exc)) from exc"))
		fmt.Fprintf(&b, "E   %s\n", msg)
		return b.String()
	}
	for _, m := range strings.Split(msg, "\n") {
		fmt.Fprintf(&b, "E       %s\n", m)
	}
	if strings.HasPrefix(msg, "assert ") {
		if q := regexp.MustCompile(`line_total\(WIDGET, (\d+)\)`).FindStringSubmatch(b.String()); q != nil {
			got := strings.Fields(msg)
			if len(got) > 1 {
				fmt.Fprintf(&b, "E        +  where %s = line_total(Item(sku='W-100', name='Widget', unit_price=10.0, taxable=True, stock=None), %s)\n", got[1], q[1])
			}
		}
	}
	fmt.Fprintf(&b, "\n%s:%d: AssertionError\n", tc.file, errLine)
	return b.String()
}

func lineOf(src, needle string) int {
	i := strings.Index(src, needle)
	if i < 0 {
		return 0
	}
	return strings.Count(src[:i], "\n") + 1
}

// ---- tiny evaluator for the bulk-discount condition ----
// grammar: expr := sum (cmpop sum)? ; sum := atom (('+'|'-') atom)* ; atom := int | ident | '(' expr ')'
// Also accepts "not (...)" and "and"/"or" combinations.

var tokRe = regexp.MustCompile(`\s*(>=|<=|==|!=|>|<|\+|-|\(|\)|[A-Za-z_][A-Za-z_0-9]*|\d+)`)

func evalCond(expr string, vars map[string]int) (bool, bool) {
	var toks []string
	rest := expr
	for strings.TrimSpace(rest) != "" {
		m := tokRe.FindStringSubmatchIndex(rest)
		if m == nil || m[0] != 0 {
			return false, false
		}
		toks = append(toks, rest[m[2]:m[3]])
		rest = rest[m[1]:]
	}
	p := &condParser{toks: toks, vars: vars}
	v, ok := p.or()
	if !ok || p.i != len(p.toks) {
		return false, false
	}
	return v != 0, true
}

type condParser struct {
	toks []string
	i    int
	vars map[string]int
}

func (p *condParser) peek() string {
	if p.i < len(p.toks) {
		return p.toks[p.i]
	}
	return ""
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (p *condParser) or() (int, bool) {
	v, ok := p.and()
	for ok && p.peek() == "or" {
		p.i++
		w, ok2 := p.and()
		v, ok = b2i(v != 0 || w != 0), ok2
	}
	return v, ok
}

func (p *condParser) and() (int, bool) {
	v, ok := p.not()
	for ok && p.peek() == "and" {
		p.i++
		w, ok2 := p.not()
		v, ok = b2i(v != 0 && w != 0), ok2
	}
	return v, ok
}

func (p *condParser) not() (int, bool) {
	if p.peek() == "not" {
		p.i++
		v, ok := p.not()
		return b2i(v == 0), ok
	}
	return p.cmp()
}

func (p *condParser) cmp() (int, bool) {
	a, ok := p.sum()
	if !ok {
		return 0, false
	}
	op := p.peek()
	switch op {
	case ">", ">=", "<", "<=", "==", "!=":
		p.i++
		b, ok := p.sum()
		if !ok {
			return 0, false
		}
		r := map[string]bool{">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b, "==": a == b, "!=": a != b}[op]
		return b2i(r), true
	}
	return a, true
}

func (p *condParser) sum() (int, bool) {
	v, ok := p.atom()
	for ok && (p.peek() == "+" || p.peek() == "-") {
		op := p.toks[p.i]
		p.i++
		w, ok2 := p.atom()
		if op == "+" {
			v += w
		} else {
			v -= w
		}
		ok = ok2
	}
	return v, ok
}

func (p *condParser) atom() (int, bool) {
	t := p.peek()
	if t == "" {
		return 0, false
	}
	p.i++
	if t == "(" {
		v, ok := p.or()
		if !ok || p.peek() != ")" {
			return 0, false
		}
		p.i++
		return v, true
	}
	if n, err := strconv.Atoi(t); err == nil {
		return n, true
	}
	if v, ok := p.vars[t]; ok && v >= 0 {
		return v, true
	}
	return 0, false
}

func codingSpecs(smoke bool) []Spec {
	var out []Spec
	for _, cs := range codingCases {
		cs := cs
		out = append(out, Spec{ID: cs.id, Scenario: "coding-agent", New: func() Task { return newCodingTask(cs, cs.id) }})
		if smoke {
			break
		}
	}
	return out
}

// interleavedSpecs are coding tasks meant to run alternately, each in its own session.
func interleavedSpecs(smoke bool) []Spec {
	cases := []codingCase{codingCases[0], codingCases[1], codingCases[2]}
	if smoke {
		cases = cases[:2]
	}
	var out []Spec
	for i, cs := range cases {
		cs, id := cs, fmt.Sprintf("s%d-%s", i+1, cs.id)
		out = append(out, Spec{ID: id, Scenario: "interleaved", New: func() Task { return newCodingTask(cs, id) }})
	}
	return out
}

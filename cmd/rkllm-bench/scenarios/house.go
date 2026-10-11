package scenarios

import (
	"fmt"
	"math"
	"strings"
)

// Port of /opt/rkllm-131/toolbench3.py (batch / procedure suite): a simulated
// house with preconditions, end-state scoring, order and conditional checks.
// Differences from the Python original: history is real (assistant tool_calls
// and matching tool_call_ids are sent back), and the turn cap is 12.

var houseTools = []any{
	fn("set_light", "Turn a light on or off, optionally set brightness percent", map[string]any{"room": tS, "on": tB, "brightness": tI}, []string{"room", "on"}),
	fn("garage_door", "Open or close the garage door", map[string]any{"action": enum("open", "close")}, []string{"action"}),
	fn("lock_door", "Lock or unlock a door", map[string]any{"door": tS, "action": enum("lock", "unlock")}, []string{"door", "action"}),
	fn("camera_record", "Start or stop recording on a camera", map[string]any{"camera": tS, "action": enum("start", "stop")}, []string{"camera", "action"}),
	fn("set_alarm", "Set the alarm system mode", map[string]any{"mode": enum("arm_away", "arm_night", "disarm")}, []string{"mode"}),
	fn("get_device_status", "Read the current state of a device", map[string]any{"device": enum("garage", "alarm", "front_door", "bath")}, []string{"device"}),
	fn("set_water_pressure", "Set bath water pressure", map[string]any{"level": enum("low", "medium", "high")}, []string{"level"}),
	fn("set_water_heater", "Turn the bath water heater on or off with a target temperature in Celsius", map[string]any{"on": tB, "target_c": tN}, []string{"on"}),
	fn("bath_tap", "Open or close the bath tap; fill_to_percent is the target fill level", map[string]any{"action": enum("open", "close"), "fill_to_percent": tI}, []string{"action"}),
	fn("set_thermostat", "Set a thermostat zone temperature in Celsius", map[string]any{"zone": tS, "temp": tN}, []string{"zone", "temp"}),
	fn("play_music", "Play a playlist in a room at a volume 0-100", map[string]any{"room": tS, "playlist": tS, "volume": tI}, []string{"room", "playlist"}),
}

const houseSystem = `You are the home controller. Execute every command the user gives, using the tools. Do all of the commands, do not skip any, do not do anything that was not asked.
You may call several tools in one reply.

PROCEDURES (always follow them exactly):
BATH: to run a bath you must do these steps in this order: (1) set_water_pressure level=low, (2) set_water_heater on=true with target_c (default 38 unless the user gives a temperature), (3) bath_tap action=open with fill_to_percent (default 80 unless the user gives a level). Never open the tap before the heater is on.
CONDITIONALS: if the user says "if X" you must check with get_device_status first, then act only if the condition is true.
When finished, reply with one short sentence summarising what you did.`

type light struct {
	on         bool
	brightness any
}

type music struct {
	playlist string
	volume   any
}

type houseState struct {
	lights     map[string]light
	garage     string
	doors      map[string]string
	recording  []string
	alarm      string
	pressure   any
	heaterOn   bool
	heaterT    any
	tapOpen    bool
	tapFill    any
	thermostat map[string]any
	music      map[string]music
}

type logEntry struct {
	name string
	args map[string]any
}

type house struct {
	s          houseState
	log        []logEntry
	precondErr int
	unknown    int
	dupes      int
	seen       map[string]bool
	names      map[string]bool
}

func newHouse(garage string) *house {
	return &house{
		s: houseState{lights: map[string]light{}, garage: garage, doors: map[string]string{}, alarm: "disarmed",
			tapFill: 0.0, thermostat: map[string]any{}, music: map[string]music{}},
		seen: map[string]bool{}, names: toolNames(houseTools),
	}
}

type missingArg string

func need(a map[string]any, k string) any {
	v, ok := a[k]
	if !ok {
		panic(missingArg(k))
	}
	return v
}

func (h *house) run(name string, a map[string]any) (res any) {
	key := jsonStr([]any{name, a})
	if h.seen[key] && name != "get_device_status" {
		h.dupes++
	}
	h.seen[key] = true
	h.log = append(h.log, logEntry{name, a})
	defer func() {
		if r := recover(); r != nil {
			if k, ok := r.(missingArg); ok {
				res = map[string]any{"error": fmt.Sprintf("missing argument '%s'", string(k))}
				return
			}
			panic(r)
		}
	}()
	s := &h.s
	switch name {
	case "set_light":
		room := strings.ToLower(pyStr(need(a, "room")))
		s.lights[room] = light{on: truthy(need(a, "on")), brightness: a["brightness"]}
	case "garage_door":
		if need(a, "action") == "close" {
			s.garage = "closed"
		} else {
			s.garage = "open"
		}
	case "lock_door":
		door := strings.ToLower(pyStr(need(a, "door")))
		if need(a, "action") == "lock" {
			s.doors[door] = "locked"
		} else {
			s.doors[door] = "unlocked"
		}
	case "camera_record":
		c := strings.ToLower(pyStr(need(a, "camera")))
		act := need(a, "action")
		idx := -1
		for i, x := range s.recording {
			if x == c {
				idx = i
			}
		}
		if act == "start" && idx < 0 {
			s.recording = append(s.recording, c)
		}
		if act == "stop" && idx >= 0 {
			s.recording = append(s.recording[:idx], s.recording[idx+1:]...)
		}
	case "set_alarm":
		s.alarm = pyStr(need(a, "mode"))
	case "get_device_status":
		d := pyStr(need(a, "device"))
		switch d {
		case "bath":
			return map[string]any{"pressure": s.pressure, "heater": map[string]any{"on": s.heaterOn, "target_c": s.heaterT},
				"tap": map[string]any{"open": s.tapOpen, "fill": s.tapFill}}
		case "garage":
			return map[string]any{d: s.garage}
		case "alarm":
			return map[string]any{d: s.alarm}
		case "front_door":
			v, ok := s.doors["front door"]
			if !ok {
				v, ok = s.doors["front_door"]
				if !ok {
					v = "unlocked"
				}
			}
			return map[string]any{d: v}
		default:
			panic(missingArg(d))
		}
	case "set_water_pressure":
		s.pressure = need(a, "level")
	case "set_water_heater":
		on := truthy(need(a, "on"))
		if on && s.pressure != "low" {
			h.precondErr++
			return map[string]any{"error": "precondition failed: set_water_pressure level=low must be done before the heater"}
		}
		s.heaterOn, s.heaterT = on, a["target_c"]
	case "bath_tap":
		open := need(a, "action") == "open"
		if open && !s.heaterOn {
			h.precondErr++
			return map[string]any{"error": "precondition failed: the water heater is off, turn it on first"}
		}
		fill, ok := a["fill_to_percent"]
		if !ok {
			if open {
				fill = 80.0
			} else {
				fill = s.tapFill
			}
		}
		s.tapOpen, s.tapFill = open, fill
	case "set_thermostat":
		s.thermostat[strings.ToLower(pyStr(need(a, "zone")))] = need(a, "temp")
	case "play_music":
		room := strings.ToLower(pyStr(need(a, "room")))
		s.music[room] = music{playlist: strings.ToLower(pyStr(need(a, "playlist"))), volume: a["volume"]}
	default:
		h.unknown++
		return map[string]any{"error": "unknown tool"}
	}
	return map[string]any{"ok": true}
}

// ---- check helpers (python lambdas) ----

type check struct {
	label string
	f     func(s *houseState) bool
}

func roomOn(s *houseState, name string, brightness *float64) bool {
	for r, v := range s.lights {
		if strings.Contains(r, name) && v.on && (brightness == nil || numEq(v.brightness, *brightness)) {
			return true
		}
	}
	return false
}

func roomOff(s *houseState, name string) bool {
	for r, v := range s.lights {
		if strings.Contains(r, name) && !v.on {
			return true
		}
	}
	return false
}

func doorLocked(s *houseState, name string) bool {
	for d, v := range s.doors {
		if strings.Contains(d, name) && v == "locked" {
			return true
		}
	}
	return false
}

func recording(s *houseState, name string) bool {
	for _, c := range s.recording {
		if strings.Contains(c, name) {
			return true
		}
	}
	return false
}

func thermo(s *houseState, name string, t float64) bool {
	for z, v := range s.thermostat {
		if f, ok := pyFloat(v); ok && strings.Contains(z, name) && math.Abs(f-t) < 0.1 {
			return true
		}
	}
	return false
}

func fptr(f float64) *float64 { return &f }

func checksBath(temp, fill float64) []check {
	return []check{
		{"pressure=low", func(s *houseState) bool { return s.pressure == "low" }},
		{fmt.Sprintf("heater on %gC", temp), func(s *houseState) bool {
			f, ok := pyFloat(s.heaterT)
			return s.heaterOn && s.heaterT != nil && ok && math.Abs(f-temp) < 0.6
		}},
		{fmt.Sprintf("tap open fill %g", fill), func(s *houseState) bool { return s.tapOpen && numEq(s.tapFill, fill) }},
	}
}

type houseScenario struct {
	id, garage, prompt string
	checks             []check
	order              [][2]string
	needStatusBefore   string
	forbidCall         string
	ref                []Reply
}

func c(name string, kv ...any) RefCall {
	a := map[string]any{}
	for i := 0; i+1 < len(kv); i += 2 {
		a[kv[i].(string)] = kv[i+1]
	}
	return RefCall{Name: name, Args: a}
}

var bathOrder = [][2]string{{"set_water_pressure", "set_water_heater"}, {"set_water_heater", "bath_tap"}}

const goodnight = "Goodnight routine: turn the living room light off, turn the hallway light on at 20 percent, lock the front door, if the garage is open then close it, start recording on the driveway camera, arm the alarm for night, set the bedroom thermostat to 19, and run a bath at 40 degrees filled to 60 percent."

func goodnightChecks(garageLabel string) []check {
	return append([]check{
		{"living off", func(s *houseState) bool { return roomOff(s, "living") }},
		{"hallway 20%", func(s *houseState) bool { return roomOn(s, "hallway", fptr(20)) }},
		{"front door locked", func(s *houseState) bool { return doorLocked(s, "front") }},
		{garageLabel, func(s *houseState) bool { return s.garage == "closed" }},
		{"recording driveway", func(s *houseState) bool { return recording(s, "driveway") }},
		{"alarm arm_night", func(s *houseState) bool { return s.alarm == "arm_night" }},
		{"bedroom 19", func(s *houseState) bool { return thermo(s, "bed", 19) }},
	}, checksBath(40, 60)...)
}

func goodnightRef(closeGarage bool) []Reply {
	calls := []RefCall{c("set_light", "room", "living room", "on", false), c("set_light", "room", "hallway", "on", true, "brightness", 20.0),
		c("lock_door", "door", "front door", "action", "lock")}
	if closeGarage {
		calls = append(calls, c("garage_door", "action", "close"))
	}
	calls = append(calls, c("camera_record", "camera", "driveway", "action", "start"), c("set_alarm", "mode", "arm_night"),
		c("set_thermostat", "zone", "bedroom", "temp", 19.0), c("set_water_pressure", "level", "low"),
		c("set_water_heater", "on", true, "target_c", 40.0), c("bath_tap", "action", "open", "fill_to_percent", 60.0))
	return []Reply{{Calls: []RefCall{c("get_device_status", "device", "garage")}}, {Calls: calls}, {Content: "Goodnight routine done."}}
}

var houseScenarios = []houseScenario{
	{
		id: "S1_five", garage: "open",
		prompt: "Turn on the kitchen light, close the garage door, start recording on the front_door camera, set the alarm to arm_away, and get the bathtub warmed up and filling.",
		checks: append([]check{
			{"kitchen light on", func(s *houseState) bool { return roomOn(s, "kitchen", nil) }},
			{"garage closed", func(s *houseState) bool { return s.garage == "closed" }},
			{"recording front_door", func(s *houseState) bool { return recording(s, "front") }},
			{"alarm arm_away", func(s *houseState) bool { return s.alarm == "arm_away" }},
		}, checksBath(38, 80)...),
		order: bathOrder,
		ref: []Reply{{Calls: []RefCall{c("set_light", "room", "kitchen", "on", true), c("garage_door", "action", "close"),
			c("camera_record", "camera", "front_door", "action", "start"), c("set_alarm", "mode", "arm_away"),
			c("set_water_pressure", "level", "low"), c("set_water_heater", "on", true, "target_c", 38.0),
			c("bath_tap", "action", "open", "fill_to_percent", 80.0)}}, {Content: "All five commands done and the bath is filling."}},
	},
	{id: "S2a_goodnight_open", garage: "open", prompt: goodnight, checks: goodnightChecks("garage closed"), order: bathOrder,
		needStatusBefore: "garage_door", ref: goodnightRef(true)},
	{id: "S2b_goodnight_closed", garage: "closed", prompt: goodnight, checks: goodnightChecks("garage still closed"), order: bathOrder,
		forbidCall: "garage_door", ref: goodnightRef(false)},
	{
		id: "S3_ten", garage: "open",
		prompt: "Right, I am heading out and the kids need a bath running at 36 degrees to 50 percent before I go. Please: kitchen light off, porch light on, lock the front door and the back door, close the garage, record the driveway camera and the front_door camera, play the jazz playlist in the living room at volume 30, set the hallway thermostat to 17, and arm the alarm to arm_away.",
		checks: append([]check{
			{"kitchen off", func(s *houseState) bool { return roomOff(s, "kitchen") }},
			{"porch on", func(s *houseState) bool { return roomOn(s, "porch", nil) }},
			{"front locked", func(s *houseState) bool { return doorLocked(s, "front") }},
			{"back locked", func(s *houseState) bool { return doorLocked(s, "back") }},
			{"garage closed", func(s *houseState) bool { return s.garage == "closed" }},
			{"rec driveway", func(s *houseState) bool { return recording(s, "driveway") }},
			{"rec front", func(s *houseState) bool { return recording(s, "front") }},
			{"jazz living 30", func(s *houseState) bool {
				for r, v := range s.music {
					if strings.Contains(r, "living") && strings.Contains(v.playlist, "jazz") && numEq(v.volume, 30) {
						return true
					}
				}
				return false
			}},
			{"hallway 17", func(s *houseState) bool { return thermo(s, "hall", 17) }},
			{"alarm arm_away", func(s *houseState) bool { return s.alarm == "arm_away" }},
		}, checksBath(36, 50)...),
		order: bathOrder,
		ref: []Reply{{Calls: []RefCall{c("set_light", "room", "kitchen", "on", false), c("set_light", "room", "porch", "on", true),
			c("lock_door", "door", "front door", "action", "lock"), c("lock_door", "door", "back door", "action", "lock"),
			c("garage_door", "action", "close"), c("camera_record", "camera", "driveway", "action", "start"),
			c("camera_record", "camera", "front_door", "action", "start"), c("play_music", "room", "living room", "playlist", "jazz", "volume", 30.0),
			c("set_thermostat", "zone", "hallway", "temp", 17.0), c("set_alarm", "mode", "arm_away"),
			c("set_water_pressure", "level", "low"), c("set_water_heater", "on", true, "target_c", 36.0),
			c("bath_tap", "action", "open", "fill_to_percent", 50.0)}}, {Content: "Done, house secured and bath running."}},
	},
}

// houseTask is one run of a house scenario.
type houseTask struct {
	sc          houseScenario
	h           *house
	statusIssue bool
	final       string
}

func (t *houseTask) ID() string         { return t.sc.id }
func (t *houseTask) Scenario() string   { return "house-batch" }
func (t *houseTask) Tools() []any       { return houseTools }
func (t *houseTask) MaxTokens() int     { return 1200 }
func (t *houseTask) Reference() []Reply { return t.sc.ref }
func (t *houseTask) Initial() []Message {
	return []Message{{Role: "system", Content: houseSystem}, {Role: "user", Content: t.sc.prompt}}
}

func (t *houseTask) OnAssistant(turn int, content string, calls []Call) bool {
	if len(calls) == 0 {
		t.final = content
	}
	return false
}

func (t *houseTask) Exec(cl Call) string {
	if !t.h.names[cl.Name] {
		t.h.unknown++
		return jsonStr(map[string]any{"error": "unknown tool"})
	}
	if cl.Name == t.sc.needStatusBefore {
		seen := false
		for _, e := range t.h.log {
			if e.name == "get_device_status" && pyStr(e.args["device"]) == "garage" {
				seen = true
			}
		}
		if !seen {
			t.statusIssue = true
		}
	}
	args := cl.Args
	if cl.Bad {
		args = map[string]any{"__bad__": true}
	}
	return jsonStr(t.h.run(cl.Name, args))
}

func orderOK(log []logEntry, pairs [][2]string) bool {
	idx := func(n string) int {
		for i, e := range log {
			if e.name == n {
				return i
			}
		}
		return -1
	}
	for _, p := range pairs {
		ia, ib := idx(p[0]), idx(p[1])
		if ia >= 0 && ib >= 0 {
			if ia > ib {
				return false
			}
		} else if ib >= 0 && ia < 0 {
			return false
		}
	}
	return true
}

func (t *houseTask) eval() Score {
	var missing []string
	ok := 0
	for _, ck := range t.sc.checks {
		if ck.f(&t.h.s) {
			ok++
		} else {
			missing = append(missing, ck.label)
		}
	}
	ordOK := orderOK(t.h.log, t.sc.order)
	forbid := false
	if t.sc.forbidCall != "" {
		for _, e := range t.h.log {
			if e.name == t.sc.forbidCall {
				forbid = true
			}
		}
	}
	condOK := !t.statusIssue
	state := math.Round(float64(ok)/float64(len(t.sc.checks))*100) / 100
	pass := len(missing) == 0 && ordOK && !forbid && condOK && t.h.unknown == 0 && t.h.precondErr == 0
	calls := make([]string, len(t.h.log))
	for i, e := range t.h.log {
		calls[i] = e.name
	}
	if missing == nil {
		missing = []string{}
	}
	return Score{Pass: pass, Score: state, Detail: map[string]any{
		"state_score": state, "missing": missing, "order_ok": ordOK, "forbidden_call_made": forbid,
		"checked_status_first": condOK, "precond_errors": t.h.precondErr, "unknown_tools": t.h.unknown,
		"dupes": t.h.dupes, "calls": calls, "final": strings.ReplaceAll(truncate(t.final, 200), "\n", " "),
	}}
}

func (t *houseTask) Solved() bool { return t.eval().Pass }
func (t *houseTask) Score() Score { return t.eval() }

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func houseSpecs(smoke bool) []Spec {
	var out []Spec
	for _, sc := range houseScenarios {
		sc := sc
		out = append(out, Spec{ID: sc.id, Scenario: "house-batch", New: func() Task { return &houseTask{sc: sc, h: newHouse(sc.garage)} }})
		if smoke {
			break
		}
	}
	return out
}

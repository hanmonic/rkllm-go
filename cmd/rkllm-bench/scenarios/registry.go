package scenarios

import (
	"fmt"
	"strings"
)

// Names lists the scenarios in default run order.
var Names = []string{"house-batch", "coding-agent", "hard-multistep", "interleaved"}

// Interleaved reports whether a scenario's tasks run alternately as one group.
func Interleaved(name string) bool { return name == "interleaved" }

// Specs returns the tasks of a scenario (one short case each when smoke).
func Specs(name string, smoke bool) ([]Spec, error) {
	switch name {
	case "house-batch":
		return houseSpecs(smoke), nil
	case "coding-agent":
		return codingSpecs(smoke), nil
	case "hard-multistep":
		return hardSpecs(smoke), nil
	case "interleaved":
		return interleavedSpecs(smoke), nil
	}
	return nil, fmt.Errorf("unknown scenario %q (have %v)", name, Names)
}

// ReferenceReply returns the scripted next assistant turn for a conversation,
// for the fake server. Tasks are matched on the initial system+user messages;
// tasks sharing a prompt (e.g. S2a/S2b) are told apart by replaying the
// reference calls and comparing the tool results with those in msgs.
func ReferenceReply(msgs []Message) (Reply, bool) {
	if len(msgs) < 2 {
		return Reply{}, false
	}
	var history []string
	turn := 0
	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			turn++
		case "tool":
			history = append(history, m.Content)
		}
	}
	for _, n := range Names {
		specs, _ := Specs(n, false)
	candidates:
		for _, s := range specs {
			t := s.New()
			ini := t.Initial()
			if len(ini) != 2 || ini[0].Content != msgs[0].Content || ini[1].Content != msgs[1].Content {
				continue
			}
			ref := t.Reference()
			var outs []string
			for i := 0; i < turn && i < len(ref); i++ {
				var calls []Call
				for j, rc := range ref[i].Calls {
					calls = append(calls, Call{ID: fmt.Sprintf("r%d_%d", i, j), Name: rc.Name, Args: rc.Args, Raw: jsonStr(rc.Args)})
				}
				if t.OnAssistant(i, ref[i].Content, calls) {
					break
				}
				for _, cl := range calls {
					outs = append(outs, t.Exec(cl))
				}
			}
			if len(outs) != len(history) {
				continue
			}
			for i := range outs {
				h := history[i]
				if k := strings.Index(h, "\n[output truncated:"); k >= 0 {
					if !strings.HasPrefix(outs[i], h[:k]) {
						continue candidates
					}
				} else if outs[i] != h {
					continue candidates
				}
			}
			if turn >= len(ref) {
				return Reply{Content: "Done."}, true
			}
			return ref[turn], true
		}
	}
	return Reply{}, false
}

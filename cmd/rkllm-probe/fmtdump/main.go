// fmtdump prints internal/prompt renderings used by the probe's template
// comparison (run on any host: go run ./cmd/rkllm-probe/fmtdump).
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"rkllm-go/internal/prompt"
)

const weatherTool = `{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`

func main() {
	q := "What is the capital of France? Answer in one word."
	cases := map[string][]prompt.Message{
		"one":   {{Role: "user", Content: q}},
		"two":   {{Role: "user", Content: "My name is Zorblax. Reply with just OK."}, {Role: "assistant", Content: "OK."}, {Role: "user", Content: "What is my name?"}},
		"tools": {{Role: "system", Content: "You are a helpful assistant."}, {Role: "user", Content: "What's the weather in Paris?"}},
	}
	out := map[string]string{}
	for _, f := range []prompt.Family{prompt.FamilyQwen3, prompt.FamilyQwen35, prompt.FamilyGemma4} {
		t, _ := prompt.Get(f)
		for name, msgs := range cases {
			for _, think := range []bool{false, true} {
				var tools []json.RawMessage
				if name == "tools" {
					tools = []json.RawMessage{json.RawMessage(weatherTool)}
				}
				out[fmt.Sprintf("%s/%s/think=%v", f, name, think)] = t.Format(msgs, tools, prompt.Options{EnableThinking: think})
			}
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	enc.SetEscapeHTML(false)
	enc.Encode(out)
}

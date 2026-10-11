"""Render the upstream HF chat templates (testdata/hf/*.jinja) for testdata/cases.json
the way transformers does, writing testdata/golden/<family>/<case>.txt.
usage: python render_hf.py <testdata dir>"""
import json, sys, os
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols

td = sys.argv[1]
def raise_exception(m): raise Exception(m)
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters['tojson'] = tojson
env.globals['raise_exception'] = raise_exception
files = {'qwen3': 'qwen3.jinja', 'qwen3.5': 'qwen3.5.jinja', 'gemma3': 'gemma3n.jinja', 'gemma4': 'gemma4.jinja'}
d = json.load(open(os.path.join(td, 'cases.json')))
for c in d['cases']:
    for fam in c['hf']:
        tpl = env.from_string(open(os.path.join(td, 'hf', files[fam])).read())
        msgs = json.loads(json.dumps(c['messages']))
        for m in msgs:
            for tc in m.get('tool_calls', []):
                if fam != 'qwen3':
                    tc['function']['arguments'] = json.loads(tc['function']['arguments'])
        kw = dict(messages=msgs, add_generation_prompt=True, bos_token='<bos>', eos_token='<eos>')
        if c.get('tools'): kw['tools'] = d['tools']
        kw['enable_thinking'] = bool(c.get('thinking'))
        out = tpl.render(**kw)
        os.makedirs(os.path.join(td, 'golden', fam), exist_ok=True)
        open(os.path.join(td, 'golden', fam, c['name'] + '.txt'), 'w').write(out)
        print(fam, c['name'], len(out))

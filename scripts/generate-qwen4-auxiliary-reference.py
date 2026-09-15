"""Tiny MTP/three-axis position oracles from the supplied mlx-vlm source.
Run with mlx-vlm/.venv/bin/python. Does not open any real checkpoint.
"""
import json
import runpy
import sys
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import Qwen4ExpMTPDraftModel
sys.argv = ['generate-qwen4-reference.py', '--bf16', '--paged']
g = runpy.run_path('scripts/generate-qwen4-reference.py')
model, c, out, weights = (g[x] for x in ['model', 'c', 'out', 'weights'])
mx.random.seed(271)
head = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig(text_config=TextConfig.from_dict(c)))
head.set_dtype(mx.bfloat16)
flat = dict(tree_flatten(head.parameters()))
mx.eval(flat)
for name, value in flat.items():
    if '.switch_mlp.' in name:
        if name.endswith('.gate_proj.weight'):
            prefix = name.split('.switch_mlp.')[0]
            weights['mtp.'+prefix+'.experts.gate_up_proj'] = mx.concatenate([value, flat[prefix+'.switch_mlp.up_proj.weight']], axis=1)
        elif name.endswith('.down_proj.weight'):
            weights['mtp.'+name.replace('.switch_mlp.down_proj.weight','.experts.down_proj')] = value
    else:
        weights['mtp.'+name] = value
mx.save_safetensors(str(out/'model.safetensors'), weights)
tokens = g['tokens']
cache = model.make_cache()
draft_cache = head.make_cache()
previous = None
rows = []
for i, token in enumerate(tokens):
    ids = mx.array([[token]],dtype=mx.int32)
    sink = []
    model.model(ids, cache=cache, hidden_sink=sink, capture_layer_ids=[])
    hidden = sink[0]
    if previous is not None:
        head._next_position = i-1
        mixed, draft_hidden = head._forward_hidden(model.model.embed_tokens(ids), previous, ids, draft_cache)
        logits = model.lm_head(mixed)
        mx.eval(hidden, draft_hidden, logits)
        rows.append({'token':token,'position':i-1,'input_hidden':previous.astype(mx.float32).reshape(-1).tolist(),
                     'hidden':draft_hidden.astype(mx.float32).reshape(-1).tolist(), 'logits':logits.astype(mx.float32).reshape(-1).tolist()})
    previous = hidden
(out/'mtp-oracle.json').write_text(json.dumps({'rows':rows,'source':'mlx-vlm qwen4_exp_mtp; bf16; seed 271; trusted target hidden shift'},indent=2)+'\n')
# Synthetic media embeddings with distinct temporal/height/width positions.
# This isolates language-side MRoPE and indexer block-start rotation.
cache = model.make_cache()
media_tokens = tokens[:16]
axes = [[i,i,i] if i<4 else ([4,(i-4)//2+4,(i-4)%2+4] if i<12 else [i-4]*3) for i in range(16)]
embeddings = model.model.embed_tokens(mx.array([media_tokens], dtype=mx.int32))
rows=[]
for i, token in enumerate(media_tokens):
    ids=mx.array([[token]],dtype=mx.int32)
    positions=mx.array(axes[i],dtype=mx.int32).reshape(3,1,1)
    hidden=model.model(ids,inputs_embeds=embeddings[:,i:i+1],cache=cache,position_ids=positions)
    logits=model.lm_head(hidden)
    mx.eval(logits)
    rows.append(logits.astype(mx.float32).reshape(-1).tolist())
(out/'mrope-oracle.json').write_text(json.dumps({'tokens':media_tokens,'positions':axes,'delta':-4,'logits':rows,
    'source':'mlx-vlm qwen4_exp; bf16; interleaved media positions across sparse-indexer threshold'},indent=2)+'\n')
print('Wrote MTP and media-position oracles')

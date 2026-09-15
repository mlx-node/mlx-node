"""Regenerate the tiny qwen4_exp oracle using the supplied mlx-vlm checkout.
Run with mlx-vlm/.venv/bin/python scripts/generate-qwen4-reference.py.
No real checkpoint is loaded. The synthetic model has fewer than 1M parameters.
"""
import json
import sys
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel

out = Path(__file__).resolve().parents[1] / 'crates/mlx-core/tests/fixtures/qwen4-exp'
if '--bf16' in sys.argv:
    out = out / 'bf16'
if '--paged' in sys.argv:
    out = out / 'paged'
c = dict(model_type='qwen4_exp_text', hidden_size=32, num_hidden_layers=4,
         num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=64,
         num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
         shared_expert_intermediate_size=16, linear_num_key_heads=2,
         linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
         linear_conv_kernel_dim=4, hc_count=2, hc_lowrank=8, rms_norm_eps=1e-6,
         max_position_embeddings=128, full_attention_interval=4, indexer_n_heads=2,
         indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
         ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4,
         ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=19,
         split_ngram_parts=2, make_ngram_vocab_size_divisible_by=8, eos_token_id=2,
         rope_parameters={'type':'default','rope_theta':10000.0,'partial_rotary_factor':0.25,
                          'mrope_section':[1,1,0],'mrope_interleaved':True})
if '--paged' in sys.argv:
    c['head_dim'] = 32
    c['indexer_head_dim'] = 16
    c['rope_parameters']['mrope_section'] = [2, 1, 1]
mx.random.seed(104)
model = LanguageModel(TextConfig.from_dict(c))
model.set_dtype(mx.bfloat16 if '--bf16' in sys.argv else mx.float32)
flat = dict(tree_flatten(model.parameters()))
assert sum(v.size for v in flat.values()) < 1_000_000
mx.eval(flat)
weights = {}
for name, value in flat.items():
    if name.startswith('model.'):
        name = 'model.language_model.' + name[len('model.'):]
    if '.switch_mlp.' in name:
        if name.endswith('.gate_proj.weight'):
            prefix = name.split('.switch_mlp.')[0]
            original = prefix.replace('model.language_model.', 'model.')
            weights[prefix + '.experts.gate_up_proj'] = mx.concatenate([
                value, flat[original + '.switch_mlp.up_proj.weight']], axis=1)
        elif name.endswith('.down_proj.weight'):
            weights[name.replace('.switch_mlp.down_proj.weight', '.experts.down_proj')] = value
        continue
    if '.conv1d.weight' in name:
        value = value.transpose(0,2,1)
    name = name.replace('.ngram_embedding.shards.0.', '.ngram_embedding.shard_0.')
    name = name.replace('.ngram_embedding.shards.1.', '.ngram_embedding.shard_1.')
    weights[name] = value
out.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(out/'model.safetensors'), weights)
(out/'config.json').write_text(json.dumps({'model_type':'qwen4_exp','text_config':c},indent=2)+'\n')
cache=model.make_cache()
tokens=[3,9,5,2,7,11,12,8,10,9,13,7,4,3,11,8,6,5]
logits=[]
for token in tokens:
    h=model.model(mx.array([[token]],dtype=mx.int32),cache=cache)
    y=model.lm_head(h)
    mx.eval(y, [a.state for a in cache])
    logits.append(y.astype(mx.float32).reshape(-1).tolist())
(out/'oracle.json').write_text(json.dumps({'tokens':tokens,'logits':logits,
    'source':f'mlx-vlm/mlx_vlm/models/qwen4_exp/language.py; seed 104; {model.lm_head.weight.dtype}; singleton decode'},indent=2)+'\n')
print('Wrote tiny oracle:', sum(v.nbytes for v in weights.values()), 'bytes')

# A second oracle exercises the GGUF converter's norm offsets, split indexer,
# tiled GDN value heads and combined PLE table, without quantization loss.
import struct
import numpy as np
suffixes = {
    'attn_hyper_connection.hc_norm.weight':'hc_attn_norm.weight',
    'attn_hyper_connection.input_mix_weight_down.weight':'hc_attn_down.weight',
    'attn_hyper_connection.input_mix_weight_up.weight':'hc_attn_up.weight',
    'attn_hyper_connection.block_inject_weight.weight':'hc_attn_inject.weight',
    'mlp_hyper_connection.hc_norm.weight':'hc_ffn_norm.weight',
    'mlp_hyper_connection.input_mix_weight_down.weight':'hc_ffn_down.weight',
    'mlp_hyper_connection.input_mix_weight_up.weight':'hc_ffn_up.weight',
    'mlp_hyper_connection.block_inject_weight.weight':'hc_ffn_inject.weight',
    'linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight',
    'linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight',
    'linear_attn.A_log':'ssm_a','linear_attn.dt_bias':'ssm_dt.bias',
    'linear_attn.conv1d.weight':'ssm_conv1d.weight','linear_attn.norm.weight':'ssm_norm.weight',
    'linear_attn.out_proj.weight':'ssm_out.weight',
    'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight',
    'self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
    'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
    'self_attn.indexer.q_layernorm.weight':'indexer.q_norm.weight',
    'self_attn.indexer.k_layernorm.weight':'indexer.k_norm.weight',
    'mlp.gate.weight':'ffn_gate_inp.weight','mlp.shared_expert_gate.weight':'ffn_gate_inp_shexp.weight',
    'mlp.shared_expert.gate_proj.weight':'ffn_gate_shexp.weight',
    'mlp.shared_expert.up_proj.weight':'ffn_up_shexp.weight',
    'mlp.shared_expert.down_proj.weight':'ffn_down_shexp.weight',
    'mlp.experts.down_proj':'ffn_down_exps.weight',
    'ple.key_proj.weight':'ple_key.weight','ple.value_proj.weight':'ple_value.weight',
    'ple.norm_key.weight':'ple_norm_key.weight','ple.norm_query.weight':'ple_norm_query.weight',
    'ple.norm_conv.weight':'ple_norm_conv.weight','ple.conv1d.weight':'ple_conv1d.weight',
}
gg = {}
for name, v in weights.items():
    name=name.removeprefix('model.language_model.')
    if name=='lm_head.weight': gg['output.weight']=v;continue
    if name=='embed_tokens.weight': gg['token_embd.weight']=v;continue
    if name.startswith('hyper_connection_mixer.'):
        dest={'hc_norm.weight':'output_hc_norm.weight','input_mix_weight_down.weight':'output_hc_down.weight','input_mix_weight_up.weight':'output_hc_up.weight'}[name.split('.',1)[1]]
        gg[dest]=v.astype(mx.float32)+1 if name.endswith('norm.weight') else v;continue
    _, layer, suffix=name.split('.',2)
    prefix=f'blk.{layer}.'
    if '.ple_embedding.' in suffix: continue
    if suffix=='mlp.experts.gate_up_proj':
        gg[prefix+'ffn_gate_exps.weight']=v[:,:c['moe_intermediate_size']]
        gg[prefix+'ffn_up_exps.weight']=v[:,c['moe_intermediate_size']:];continue
    if suffix=='self_attn.indexer.index_qk_proj.weight':
        n=c['indexer_n_heads']*c['indexer_head_dim']
        gg[prefix+'indexer.q_proj.weight']=v[:n];gg[prefix+'indexer.k_proj.weight']=v[n:];continue
    if suffix.startswith('linear_attn.'):
        kh=c['linear_num_key_heads'];vh=c['linear_num_value_heads'];kd=c['linear_key_head_dim'];vd=c['linear_value_head_dim']
        def reorder(v,dim,hd):
            shape=list(v.shape); tmp=shape[:dim]+[kh,vh//kh,hd]+shape[dim+1:]
            perm=list(range(len(tmp)));perm[dim],perm[dim+1]=perm[dim+1],perm[dim]
            return v.reshape(tmp).transpose(perm).reshape(shape)
        if 'in_proj_qkv' in suffix:
            v=mx.concatenate([v[:2*kh*kd],reorder(v[2*kh*kd:],0,vd)],axis=0)
        elif 'in_proj_z' in suffix: v=reorder(v,0,vd)
        elif 'in_proj_a' in suffix or 'in_proj_b' in suffix: v=reorder(v,0,1)
        elif suffix.endswith(('A_log','dt_bias')): v=reorder(v[:,None],0,1).reshape(-1)
        elif 'conv1d' in suffix:
            v=v.squeeze();v=mx.concatenate([v[:2*kh*kd],reorder(v[2*kh*kd:],0,vd)],axis=0)
        elif 'out_proj' in suffix: v=reorder(v,1,vd)
    if suffix.endswith('A_log'): v=-mx.exp(v.astype(mx.float32))
    elif (suffix.endswith('norm.weight') and suffix!='linear_attn.norm.weight') or suffix.startswith(('ple.norm_key.','ple.norm_query.','ple.norm_conv.','self_attn.indexer.q_layernorm.','self_attn.indexer.k_layernorm.')):
        v=v.astype(mx.float32)+1
    if 'conv1d.weight' in suffix or suffix=='mlp.shared_expert_gate.weight':v=v.squeeze()
    gg[prefix+suffixes[suffix]]=v
base='model.language_model.layers.1.ple.ple_embedding.'
gg['per_layer_token_embd.weight']=mx.concatenate([weights[base+f'ngram_embedding.shard_{i}.weight'] for i in range(c['split_ngram_parts'])])
mapping={'hidden_size':'embedding_length','num_hidden_layers':'block_count','num_attention_heads':'attention.head_count','num_key_value_heads':'attention.head_count_kv','head_dim':'attention.key_length','num_experts':'expert_count','num_experts_per_tok':'expert_used_count','moe_intermediate_size':'expert_feed_forward_length','shared_expert_intermediate_size':'expert_shared_feed_forward_length','linear_num_key_heads':'ssm.group_count','linear_num_value_heads':'ssm.time_step_rank','linear_key_head_dim':'ssm.state_size','linear_conv_kernel_dim':'ssm.conv_kernel','hc_count':'hyper_connection.count','hc_lowrank':'hyper_connection.low_rank','max_position_embeddings':'context_length','full_attention_interval':'full_attention_interval','indexer_n_heads':'attention.indexer.head_count','indexer_head_dim':'attention.indexer.key_length','indexer_budget':'attention.indexer.top_k','ple_conv_kernel_size':'ple.conv_kernel','ngram_size':'ple.ngram_size','heads_per_ngram':'ple.heads_per_ngram','eos_token_id':'ple.eos_token_id'}
meta={'general.architecture':(8,'qwen4exp')}
meta.update({'qwen4exp.'+dest:(4,c[src]) for src,dest in mapping.items()})
meta.update({'qwen4exp.ssm.inner_size':(4,c['linear_num_value_heads']*c['linear_value_head_dim']),
 'qwen4exp.attention.layer_norm_rms_epsilon':(6,c['rms_norm_eps']),
 'qwen4exp.rope.freq_base':(6,10000.0),'qwen4exp.rope.dimension_count':(4,4),
 'qwen4exp.attention.compress_ratios':(9,(4,[0,0,0,4])), 'qwen4exp.ple.layers':(9,(4,[1]))})
for dest,source in [('layer_multipliers','layer_multipliers'),('head_offsets','ngram_heads_offsets'),('head_vocab_sizes','ngram_heads_vocab_sizes')]:
    meta['qwen4exp.ple.'+dest]=(9,(11,weights[base+source].tolist()))
def string(v):
    b=v.encode();return struct.pack('<Q',len(b))+b
def value(t,v):
    if t==8:return string(v)
    if t==9:
        typ,items=v;return struct.pack('<IQ',typ,len(items))+b''.join(value(typ,x) for x in items)
    return struct.pack('<'+{4:'I',6:'f',11:'q'}[t],v)
header=b'GGUF'+struct.pack('<IQQ',3,len(gg),len(meta))
for k,(t,v) in meta.items():header+=string(k)+struct.pack('<I',t)+value(t,v)
data=bytearray()
for name,v in gg.items():
    mx.eval(v)
    if v.dtype==mx.bfloat16:typ=30;payload=np.array(v.view(mx.uint16)).tobytes()
    else:typ=0;payload=np.array(v.astype(mx.float32)).tobytes()
    while len(data)%32:data+=b'\0'
    header+=string(name)+struct.pack('<I',v.ndim)+b''.join(struct.pack('<Q',n) for n in reversed(v.shape))+struct.pack('<IQ',typ,len(data))
    data+=payload
while len(header)%32:header+=b'\0'
(out/'model.gguf').write_bytes(header+data)
print('Wrote GGUF oracle:', len(header)+len(data),'bytes')

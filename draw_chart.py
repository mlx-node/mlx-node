import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / 'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
BG, INK, MUTED, GRID = '#F5F7FC', '#17243B', '#58667C', '#E4E9F2'
colors = {'mlx-node':'#6554D9','llama.cpp':'#149D92'}
fig=plt.figure(figsize=(15,9),facecolor=BG)
fig.text(.055,.941,'MLX-NODE  /  PERFORMANCE TRACKER',fontsize=10,weight='bold',color=MUTED)
fig.text(.945,.941,'10 SEP 2026',ha='right',fontsize=10,color=MUTED)
fig.text(.055,.879,'Gemma 4 12B',fontsize=31,weight='bold',color=INK)
fig.text(.055,.837,'QAT Q4_0  ·  Q6_K head  ·  Recorded coding-agent prompts',fontsize=14,color=MUTED)
fig.text(.055,.788,'Apple M5 Max · 40 GPU cores · 128 GB     |     3 runs per cell · 256 output tokens',fontsize=11,color=MUTED)
legend=[Patch(facecolor=colors[rt],label=rt) for rt in colors]
fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.945,.889),frameon=False,ncols=1,fontsize=12,labelspacing=.8)

panels=[(.05,.445,'Prefill','prefillTokensPerSecond',1500,'−7.9% / +35.0% / +65.6%','Prompt processing · tokens/second'),
        (.535,.415,'Decode','decodeTokensPerSecond',60,'+3.8% / +12.9% / +18.2%','Token generation · tokens/second')]
for left,width,title,key,limit,headline,subtitle in panels:
    card=FancyBboxPatch((left,.19),width,.54,boxstyle='round,pad=0.01,rounding_size=0.014',
                       linewidth=0,facecolor='white',transform=fig.transFigure,zorder=0)
    fig.add_artist(card)
    fig.text(left+.025,.677,title,fontsize=22,weight='bold',color=INK)
    fig.text(left+width-.025,.681,headline,ha='right',fontsize=10.5,weight='bold',color=colors['mlx-node'])
    fig.text(left+.025,.64,subtitle,fontsize=10.5,color=MUTED)
    ax=fig.add_axes([left+.079,.265,width-.102,.32],facecolor='white')
    ys=np.array([2,1,0],dtype=float)
    for rt,offset in [('mlx-node',.17),('llama.cpp',-.17)]:
        vals=np.array([r[rt][key]['median'] for r in rows])
        mins=np.array([r[rt][key]['min'] for r in rows])
        maxs=np.array([r[rt][key]['max'] for r in rows])
        ax.barh(ys+offset,vals,height=.27,color=colors[rt],zorder=3)
        ax.errorbar(vals,ys+offset,xerr=np.vstack([vals-mins,maxs-vals]),fmt='none',
                    ecolor=INK,elinewidth=1.1,capsize=3,capthick=1,zorder=4)
        for y,v in zip(ys+offset,vals):
            ax.text(v*.63,y,f'{v:.1f}' if key.startswith('prefill') else f'{v:.2f}',
                    color='white',va='center',ha='center',fontsize=11.5,weight='bold',zorder=5)
    ax.set_yticks(ys,labels=['7,733','40,528','66,904'])
    ax.tick_params(axis='y',length=0,pad=10,labelsize=11,colors=INK)
    ax.tick_params(axis='x',length=0,pad=8,labelsize=10,colors=MUTED)
    ax.set_xlim(0,limit);ax.set_ylim(-.55,2.55)
    ax.set_xticks([0,300,600,900,1200,1500] if key.startswith('prefill') else [0,10,20,30,40,50,60])
    ax.grid(axis='x',color=GRID,lw=.8,zorder=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xlabel('Tokens per second  →',labelpad=12,color=MUTED,fontsize=10)
    fig.text(left+.025,.213,'Prompt tokens',fontsize=9.5,color=MUTED)

fig.text(.055,.137,'Bars show the median; whiskers show the observed min–max across three runs. Higher is faster.',fontsize=10.5,color=INK)
fig.text(.055,.099,'Active desktop · No other inference or compilation · Calibration included · Greedy outputs can differ',fontsize=10,color=MUTED)
fig.text(.055,.065,'Identical prompt IDs · BF16 K/V · 512-token prefill chunks · No prefix reuse or speculation · Separate chart scales',fontsize=9.5,color=MUTED)
for suffix in ['png','svg']:
    fig.savefig(OUT/f'gemma4-12b-qat-q4-0-benchmark.{suffix}',dpi=200,facecolor=BG)
print(OUT/'gemma4-12b-qat-q4-0-benchmark.png')

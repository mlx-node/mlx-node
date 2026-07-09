// 第 10 章（LM head 与权重共享）子章节「logit lens」正文的简体中文译文，与
// 英文版 ../lm-head.tsx 平行维护：同名导出 LmHeadLogitLensSection。术语保留英文
// （logit lens、token、layer、logits、W_U、Qwen3.5、Jacobian lens、tuned lens、
// nostalgebraist、unembedding、residual、softmax、off-distribution），只翻译外围
// 讲解。诚实约束（绑定）：本页只跑 PLAIN logit lens，不跑 Jacobian/tuned lens；
// fitted-lens 那条线是 Anthropic 在他们自己模型上的工作，本页既不运行也不复现。
// 同一个 <LogitLensLive/> widget（已双语，按 useLocale() 切换文案）在此渲染。

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { LogitLensLive } from '../../widgets/jlens/LogitLensLive';

export function LmHeadLogitLensSection() {
  return (
    <Prose>
      <h1>Logit lens</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 10 章 LM head——把本章刚搭好的那一次 matmul 沿着 stack 往下滑：从每一个深度读出词表，而不只是最后一层。
      </p>
      <p>
        <ChapterLink chapterId="lm-head">LM head</ChapterLink> 把下一个 token 的分布从<em>最后</em>那个 hidden state 里读出来——办法就是让它经过共享的
        unembedding <code>W_U</code>。可这次 matmul 本身，并不要求那个 hidden state 一定是最后一层。于是就有了这个一句话的问题，它开启了一整条
        interpretability 的支线：如果把<em>同一支</em> <code>W_U</code> 对准中间层，会读出什么？这就是{' '}
        <strong>logit lens</strong>——由 <strong>nostalgebraist</strong> 于 2020 年提出（「interpreting GPT: the logit lens」，LessWrong）。
      </p>

      <h2>核心想法：一种读法，读遍每个深度</h2>
      <p>
        一个 decoder LLM 为每个 token 只携带一支 residual 向量 <code>h_ℓ</code>，沿着 stack 一个边界一个边界地精修。LM head 永远只对这叠东西的<em>顶端</em>做
        unembedding。logit lens 就是把同样的事在中间每一个边界上都做一遍：取 <code>h_ℓ</code>，直接推过 <code>W_U</code>，做 softmax，读出 top token——相当于在每个深度上粗略地问一句「如果<em>现在</em>就得作答，模型会说什么？」。把这些读数按 layer 排开，你就能看着一个预测一层层拼装成形。
      </p>

      <h2>实时看它跑起来</h2>
      <p>
        下面这个 widget，正是把上面这套办法跑在你浏览器里加载的 Qwen3.5-0.8B 上：一次 forward pass，然后在十来个 residual 边界上做那次共享的{' '}
        <code>W_U</code> 读取。挑一个短提示，运行它。看每一层的 top token 如何从 stack 中段的乱码里爬出来、在靠近顶端处收敛到真正的答案——再看一个被 pin 住的答案
        token（比如「The capital of France is」里的 <code>Paris</code>）如何只在靠后的 layer 里，才在全词表 rank 上一路往上爬。
      </p>

      <LogitLensLive />

      <p>
        有两点值得留意。中间那些行大多是噪声——模型还没「拿定主意」，所以那里的 top token 常常与答案毫不相干。而且答案并不是平滑淡入的；它往往是在最后寥寥几层里「啪」地一下定型。这是 logit lens 上反复出现的观察：在这个模型上，面向词表的那个决定是很晚才做出的，而更早的那些 layer 仍在做一些 unembedding 读不干净的事。
      </p>

      <h2>为什么它只是近似——那块盲区</h2>
      <p>
        logit lens 是一个<em>粗糙的探针</em>，不是 ground truth，值得把原因说清楚。<code>W_U</code> 被训练来读的分布只有一种：<em>最后一层</em>的那种。一个中间的{' '}
        <code>h_ℓ</code> 活在表示空间里另一片区域——对 unembedding 而言它是 <strong>off-distribution</strong> 的——所以把它经过 <code>W_U</code> 去读，是一条可能错得很离谱的捷径，而且在越靠前的 layer 越糟。在一个浅浅的 0.8B 模型上，可解释的那条带子又薄又靠后：那些嘈杂的中段读数，是这套方法诚实的签名，而不是本 demo 的缺陷。
      </p>

      <h2>怎么修：一个 fitted lens（以及本页不主张什么）</h2>
      <p>
        盲区是有修法的：不用裸的 <code>W_U</code>，而是先让每个 <code>h_ℓ</code> 经过一个小小的、学出来的映射，让读法去匹配那一层真正编码的东西。<em>tuned lens</em> 为每一层学一个 affine 探针；<strong>Jacobian lens</strong> 则让每个{' '}
        <code>h_ℓ</code> 经过网络自身的平均 Jacobian 来读。这条 fitted-lens 的线——尤其是 Jacobian lens——是 <strong>Anthropic</strong> 在<em>他们自己</em>的模型上做并评估的工作，它锐化的，正是你刚才看着迟迟不肯收敛的那些模糊的中间层。
      </p>
      <p className="text-muted-foreground">
        把范围说清楚：本页只跑 <strong>plain logit lens</strong>。它<em>不</em>跑 Jacobian lens，也不跑 tuned lens，更不主张复现 Anthropic 的结果——它只是把 fitted lens 想去补上的那块盲区，实时摆给你看。
      </p>
    </Prose>
  );
}

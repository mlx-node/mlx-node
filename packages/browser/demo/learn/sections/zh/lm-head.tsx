// 第 10 章（LM head 与权重共享）子章节「logit lens」正文的简体中文译文，与
// 英文版 ../lm-head.tsx 平行维护：同名导出 LmHeadLogitLensSection。术语保留英文
// （logit lens、token、layer、logits、W_U、Qwen3.5、Jacobian lens、tuned lens、
// nostalgebraist、unembedding、residual、softmax、off-distribution），只翻译外围
// 讲解。诚实约束（绑定）：本页只跑 PLAIN logit lens，不跑 Jacobian/tuned lens；
// tuned lens 是 Belrose 等人/EleutherAI（2023），Jacobian lens 是 Anthropic 在
// 他们自己模型上的工作，本页两种 fitted lens 都不运行也不复现。
// 同一个 <LogitLensLive/> widget（已双语，按 useLocale() 切换文案）在此渲染。

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { JacobianLensLive } from '../../widgets/jlens/JacobianLensLive';
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
        盲区是有修法的：不用裸的 <code>W_U</code>，而是先让每个 <code>h_ℓ</code> 经过一个小小的、学出来的映射，让读法去匹配那一层真正编码的东西。<em>tuned lens</em> 为每一层学一个 affine 探针（Belrose 等人，EleutherAI，2023）；<strong>Jacobian lens</strong> 则让每个{' '}
        <code>h_ℓ</code> 经过网络自身的平均 Jacobian 来读。其中 <em>Jacobian lens</em> 是 <strong>Anthropic</strong> 在<em>他们自己</em>的模型上做并评估的工作，它锐化的，正是你刚才看着迟迟不肯收敛的那些模糊的中间层。
      </p>
      <p className="text-muted-foreground">
        把范围说清楚：本页只跑 <strong>plain logit lens</strong>。它<em>不</em>跑 Jacobian lens，也不跑 tuned lens，更不主张复现 Anthropic 的结果——它只是把 fitted lens 想去补上的那块盲区，实时摆给你看。
      </p>
    </Prose>
  );
}

// 第 10 章子章节「Jacobian lens」正文的简体中文译文，与英文版 ../lm-head.tsx
// 平行维护：同名导出 LmHeadJacobianLensSection。术语保留英文（logit lens、
// Jacobian lens、tuned lens、token、layer、rank、argmax、top-K、W_U、unembedding、
// residual、Qwen3.5、AUC、WikiText、softmax、off-distribution），只翻译外围讲解。
// 诚实约束（绑定，会被复核）：本页跑的是我们自己的 fit——J_1..J_23 在这颗
// Qwen3.5-0.8B 上、用 100 条 WikiText 提示拟合（约 11 小时）。方法是 Anthropic 的；
// fit 和本页所有数字都是我们在小模型上得到的。Anthropic 在前沿规模上的结果一律
// 框成「Anthropic 在 Claude 上看到的」并附引用，绝不冒充为我们的。下面的 eval 数字
// 逐字来自 eval-results-v1.json，band 数字来自 band-report.json。同一个
// <JacobianLensLive/> widget（已双语）在此渲染。
export function LmHeadJacobianLensSection() {
  return (
    <Prose>
      <h1>Jacobian lens</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 10 章 LM head——用来修那块 logit lens 读不动的模糊中间层。而且和那一页不同，这一页跑的是一个真正<em>拟合过</em>的 lens，就在你浏览器里的 Qwen3.5-0.8B 上。
      </p>
      <p>
        <ChapterLink chapterId="lm-head">logit lens</ChapterLink> 那一页最后停在一块诚实的盲区上：一个中间的 <code>h_ℓ</code> 对共享的 unembedding <code>W_U</code> 而言是 <strong>off-distribution</strong> 的，所以把它直接推过 <code>W_U</code>，在中段读出来的是噪声——而中段恰恰是你最想读的地方。修法是：别再裸读 <code>h_ℓ</code>，先把它<em>搬运</em>到 unembedding 被训练去读的那个坐标系里。这就是 <strong>Jacobian lens</strong>；而且和 logit lens 那一页不同，本页把这个拟合过的 lens 实时跑起来：把它打开，看那些原本迟迟不肯收敛的层如何变得可读。
      </p>

      <h2>什么是 Jacobian lens</h2>
      <p>
        保留 LM head 那一次 matmul，只在它前面塞进一个逐层的映射 <code>J_ℓ</code>：
      </p>
      <MathDisplay latex={String.raw`\operatorname{lens}(h_\ell) = \operatorname{softmax}\!\left(W_U \cdot \operatorname{norm}(J_\ell \, h_\ell)\right)`} />
      <p>
        <code>J_ℓ</code> 是一个逐层的、在语料上<em>取平均</em>的一阶（Jacobian）映射，刻画在 layer <code>ℓ</code> 处一个小扰动如何推动<em>最后</em>那个 residual。用上它，就把 <code>h_ℓ</code> <em>先</em>搬运进最后一层的坐标系，<em>再</em>让共享的 unembedding 去读——于是读法匹配的是 stack 顶端所期待的，而不是 layer <code>ℓ</code> 裸露出来的样子。两个邻居能帮你定位它：<strong>logit lens</strong> 就是这个公式里 <MathDisplay inline latex={String.raw`J_\ell = I`} />（裸读 residual）；<strong>tuned lens</strong> 则把 <code>J_ℓ</code> 换成一个端到端学出来的逐层 affine 探针（Belrose 等人，EleutherAI，2023）。
      </p>

      <h2>不是 tuned lens——目标不一样</h2>
      <p>
        值得把它和 tuned lens 的区别说准。tuned lens 的优化目标是让它的读数<em>和下一个 token 吻合</em>——它最小化的是与模型自身输出分布之间的 KL。Jacobian lens 什么都不优化成这样：它靠把每一层经过网络自身的平均 Jacobian 搬运，来读出那一层<em>真正编码</em>的东西。「和下一个 token 最吻合」与「对中间层读得最好」是<em>两个不同的目标</em>；当它们不一致时，那是特性而不是缺陷——两种 lens 回答的本就是两个不同的问题。
      </p>

      <h2>我们拟合了什么——又没拟合什么</h2>
      <p>
        来源要说得直白。我们把 23 个映射 <code>J_1..J_23</code> 在 <strong>100 条 WikiText 提示</strong>（wikitext-103-raw-v1，原始文本，不套 chat template）上、在<em>这颗</em> Qwen3.5-0.8B 上拟合——在笔记本上约 <strong>11 小时</strong>——拟合目标设为 final norm <em>之前</em>的那个最后 residual。输出边界 <code>ℓ24</code> 按构造保持 <MathDisplay inline latex={String.raw`J = I`} />，所以这个拟合的 lens 只作用在中间层。<strong>方法是 Anthropic 的</strong>；fit 和本页每一个数字都是<strong>我们的</strong>，来自一颗小小的开源模型。拟合脚本和那些 vendored 的 eval 套件都在仓库里（<code>packages/browser/scripts/jlens/</code>），Apache-2.0 许可，所以整件事从头到尾可复现。
      </p>

      <h2>实时看它跑起来</h2>
      <p>
        在一个精选提示上切换 <strong>LOGIT | JACOBIAN</strong>。baked 帧无需下载、瞬间渲染；「在你的设备上实时计算」会在浏览器里的模型上重算它（JACOBIAN 模式还会一次性加载拟合的 pack）。看一个被 logit lens 一直压在 rank 999+ 的概念，如何在拟合的 <code>J</code> 下、于 stack 中段一下子蹿进 top 几名。
      </p>

      <JacobianLensLive />

      <h2>这个拟合的 lens 真的有用吗？</h2>
      <p>
        实时 demo 可以挑好看的说。所以这里给出聚合结果，来自我们六个 vendored eval 套件（论文 §methods-comparison 的那些提示）。headline 指标是一个归一化的 log-<code>k</code> pass@<code>k</code> <strong>AUC</strong>——越高表示目标/中间概念变得可读得<em>越早、在越多深度上可读</em>——其中 rank 取<em>在拟合域上的 min</em> <code>ℓ1..23</code>。在这一次运行里，Jacobian lens 在 <strong>六个套件里赢了六个</strong>：
      </p>

      <div className="not-prose my-5 overflow-x-auto rounded-md border border-border bg-background">
        <table className="w-full border-collapse text-[13px]">
          <caption className="px-3 py-2 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
            J-lens vs logit-lens · headline AUC（min over ℓ1..23）· 来自 eval-results-v1.json
          </caption>
          <thead>
            <tr className="border-y border-border text-[11px] uppercase tracking-wider text-muted-foreground">
              <th className="px-3 py-2 text-left font-medium">Suite</th>
              <th className="px-3 py-2 text-right font-medium">J-lens AUC</th>
              <th className="px-3 py-2 text-right font-medium">logit-lens AUC</th>
              <th className="px-3 py-2 text-right font-medium">J 胜</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {[
              { suite: 'typo', j: '0.781', logit: '0.432' },
              { suite: 'order-ops', j: '0.638', logit: '0.242' },
              { suite: 'multihop', j: '0.530', logit: '0.262' },
              { suite: 'multilingual', j: '0.484', logit: '0.265' },
              { suite: 'association', j: '0.053', logit: '0.003' },
              { suite: 'poetry', j: '0.039', logit: '0.031' },
            ].map((r) => (
              <tr key={r.suite} className="border-b border-border/50 last:border-b-0">
                <td className="px-3 py-1.5 text-left text-foreground/90">{r.suite}</td>
                <td className="px-3 py-1.5 text-right font-semibold text-foreground">{r.j}</td>
                <td className="px-3 py-1.5 text-right text-muted-foreground">{r.logit}</td>
                <td className="px-3 py-1.5 text-right text-primary">✓</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        诚实地读这张表。AUC 更高不等于「模型答对了」——它的意思是：在拟合的 lens 下，答案或某个中间概念变得<em>可读</em>得更早、跨越更多层。而且两个最难的套件——<strong>poetry</strong> 和 <strong>association</strong>——对<em>两种</em> lens 都贴着地板（J-AUC 0.039 和 0.053；logit-AUC 0.031 和 0.003）：Jacobian lens 仍然赢，但在这种对 0.8B 来说很难的任务上，两种 lens 都读不出多少。用这条命令复现整张表：<code>JLENS_PACK=lens-pack-v1.safetensors JLENS_OUT=eval-results-v1.json oxnode packages/browser/scripts/jlens/eval.mts</code>。
      </p>

      <h2>增益到底在哪儿</h2>
      <p>
        这份优势并不是沿 stack 均匀铺开的——它是一条<strong>中段的带子</strong>。四个衡量 band 结构的 proxy 检测器全部触发（4/4）；而直接测量它的那个——在给定边界上，J-lens rank 优于 logit rank 的 eval 中间概念所占的比例——从早段带子里在 <code>ℓ6-7</code> 附近抬起来，并在 <strong>ℓ17 达到峰值</strong>（比例 <strong>0.595</strong>，对比早段带子的 <strong>0.158</strong>）。要留意 headline AUC 是什么、不是什么：它是<em>在整个拟合域上取 min</em> <code>ℓ1..23</code>，而不是一个限制在带子内的分数——band 的故事解释的是这个 min <em>大概</em>来自哪里，并不改变这个数字是怎么算出来的。
      </p>
      <p>
        widget 里那个法语 headline 提示，就是这件事的定性面孔：对 <code>La saison après l&rsquo;été est l&rsquo;</code>，抽象概念 <code>season</code> 和 <code>summer</code> 在拟合的 <code>J</code> 下、于边界 16–17 附近浮到 rank 1–2 左右，而 plain logit lens 在同样的深度上把它们死死压在 rank 999+。
      </p>

      <h2>Anthropic 在前沿规模上看到的</h2>
      <blockquote>
        在<em>他们自己</em>的模型（Claude）上，Anthropic 报告 Jacobian lens 能读出丰富、可言说的中间内容——workspace 带子大约 <code>k~25</code> 层宽，读数在 ≤10% 的方差内稳定。那些数字是 <strong>Sonnet-4.5 的</strong>，在 Anthropic 的模型上，出自他们的论文 <em>Verbalizable Representations Form a Global Workspace in Language Models</em>（transformer-circuits.pub，2026）。他们的 causal、swap、tuned-lens 实验我们既不运行也不复现。一颗 0.8B 是否也有那么丰富，<strong>明确未知</strong>——上面的 6/6 是在一颗小开源模型上、只做 readout 的部分复现，并不是对他们结果的主张。
      </blockquote>

      <div className="not-prose my-5 space-y-2 rounded-md border border-border bg-muted/30 p-4 text-[13px] text-foreground/85">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">诚实地读这些读数</div>
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            每一条 rank 轨迹追踪的都是一个<strong>单 token 的表层形式</strong>（带前导空格的那个 token）；一个被切成多个 token 的概念，只追踪它的第一片。
          </li>
          <li>
            一个 readout 是<strong>一袋没有绑定的概念</strong>——它告诉你某个方向存在，而不是它如何被组合、如何绑定到某个角色上。要预期有些读数本身就难以干净地解释。
          </li>
          <li>
            对两种 lens 来说，stack 的前三分之一都是<strong>有噪声的</strong>；默认视图正因如此隐藏了 <code>ℓ1..5</code>。
          </li>
          <li>
            输出边界 <code>ℓ24</code> 按构造是 <MathDisplay inline latex={String.raw`J = I`} />，所以它的 Jacobian 读数和 logit 读数完全相同——拟合的映射只作用在中间层。
          </li>
          <li>
            <code>k~25</code> 的宽度和 ≤10% 的方差是 <strong>Sonnet-4.5 的，不是我们的</strong>；一颗 0.8B 是否带有可比的结构，未知。
          </li>
          <li>
            我们的 fit 用了 <strong>100 条 WikiText 提示、不套 chat template</strong>；它是一次小而诚实的拟合，不是一个生产级的 lens。
          </li>
        </ul>
        <p className="text-[11px] text-muted-foreground/80">
          拟合配方和那六个 vendored eval 套件是 Apache-2.0（上游：<code>anthropics/jacobian-lens</code>）；见{' '}
          <code>packages/browser/scripts/jlens/data/NOTICE</code>。
        </p>
      </div>
    </Prose>
  );
}

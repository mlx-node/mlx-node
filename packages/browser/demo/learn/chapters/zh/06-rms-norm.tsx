import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { LearnedGainInspector } from '../../widgets/LearnedGainInspector';
import { SaturatedSoftmax } from '../../widgets/SaturatedSoftmax';

/**
 * Scaffolding metadata for chapter 6 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[5].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'rmsnorm',
  objective: '解释 RMSNorm 对隐藏向量做了什么，以及为什么 pre-norm（前置归一化）让深层 Transformer 变得可训练。',
  problem:
    '每一层都把注意力和 MLP 的输出加回同一条残差流——如果没有归一化，幅值会在 24 层中不断漂移，softmax 会饱和，训练随之崩溃。RMSNorm 的全部职责就是踩住刹车。',
  minutes: 7,
  glossary: [
    {
      term: 'RMSNorm',
      definition: '把 x 除以 sqrt(mean(x^2) + eps)，再逐元素乘上可学习的增益 g。不减均值，也没有偏置项。',
    },
    {
      term: 'LayerNorm',
      definition:
        '它的前辈：先减去 mean(x)，再除以标准差，同时带有增益和偏置。RMSNorm 去掉了减均值和偏置这两步。',
    },
    {
      term: 'pre-norm',
      definition: '一种架构选择：在每个子模块之前对残差流做归一化。残差流本身保持未归一化。',
    },
    {
      term: 'post-norm',
      definition: '2017 年原版 Transformer 的模式：先做残差相加，最后归一化。更容易解读，但深层训练困难得多。',
    },
    {
      term: 'variance / std',
      definition:
        '方差（variance）= 与均值的平方距离的平均值。标准差（std）= 方差的平方根。LayerNorm 除以标准差；RMSNorm 则把均值完全跳过。',
    },
    {
      term: 'L2 norm',
      definition:
        '向量各元素平方和的平方根。一个标准的幅值度量；RMSNorm 让每个 token 的 L2 落在 sqrt(hidden_dim) 附近。',
    },
    {
      term: 'learned gain (g)',
      definition: 'RMSNorm 在除以 RMS 之后施加的逐特征缩放。归一化器携带的唯一可学习参数。',
    },
    {
      term: 'gradient path / identity path',
      definition:
        '训练信号在网络中反向传播所走的路线。残差高速公路是一条恒等路径（identity path），因为它把信号原样传过去（乘以 1），所以信号不会在多层传播中逐渐缩小到零（消失）。',
    },
  ],
  takeaways: [
    'RMSNorm 在下一个子模块读取输入之前，把输入幅值压到大约 sqrt(hidden_dim)——对 Qwen3.5-0.8B 来说约为 32。',
    'pre-norm 让残差流保持未归一化，梯度因此能沿一条恒等路径流动；这正是 24 层堆叠能训得动的原因。',
    '去掉 LayerNorm 的减均值和偏置在质量上基本无损，计算上还略微更省——每个现代开源 LLM 都这么做。',
  ],
  exercise: {
    prompt:
      '自动运行结束后，用 Layer 选择器在第 0 层和第 22 层之间来回切换。比较 "Input RMSNorm input" 与 "Input RMSNorm output" 的 L2/tok 数值。随着深度增加各自如何变化？哪一个大致保持不变？',
    answer:
      '输入侧的 L2/tok 随深度显著增长（残差相加不断累积），而输出侧的 L2/tok 始终停留在同一量级（接近 sqrt(1024) ≈ 32，并被可学习增益调制）。这种跨深度的稳定性正是归一化的职责所在。',
  },
  quiz: [
    {
      id: 'q1-formula-diff',
      prompt: '与更早的 LayerNorm 相比，RMSNorm 去掉了什么？',
      options: [
        {
          id: 'a',
          label: '可学习增益 g。',
        },
        {
          id: 'b',
          label: '减均值这一步和加性偏置——RMSNorm 只除以 RMS，再施加一个可学习增益。',
        },
        {
          id: 'c',
          label: '整个除法步骤；它只做重新缩放。',
        },
      ],
      correctId: 'b',
      explanation: 'RMSNorm = LayerNorm 去掉减均值、再去掉偏置。经验上质量不相上下，速度上小有收益。',
    },
    {
      id: 'q2-pre-vs-post-norm',
      prompt: '为什么 pre-norm 是 20 层以上 Transformer 的标准选择？',
      options: [
        {
          id: 'a',
          label: '梯度沿未归一化的残差路径以干净的恒等映射流动，深层堆叠因此保持可训练。',
        },
        {
          id: 'b',
          label: 'pre-norm 比 post-norm 用的参数更少。',
        },
        {
          id: 'c',
          label: 'pre-norm 不需要残差连接。',
        },
      ],
      correctId: 'a',
      explanation:
        'pre-norm 让残差高速公路保持未归一化——归一化只作用于每个子模块的输入。这样就保留了一条贯穿深度、畅通无阻的梯度路径。',
    },
    {
      id: 'q3-output-magnitude',
      prompt: '经过 RMSNorm 和可学习增益之后，一个 token 隐藏向量的 L2 大致落在什么量级？',
      options: [
        {
          id: 'a',
          label: '大约 1——RMSNorm 把每个向量归一化为单位长度。',
        },
        {
          id: 'b',
          label: '大约 sqrt(hidden_dim)，再乘上可学习增益——对 1024 维隐藏状态来说约为 32。',
        },
        {
          id: 'c',
          label: '保持输入原本的幅值，不做任何改变。',
        },
      ],
      correctId: 'b',
      explanation:
        'RMSNorm 使 mean(x^2) ≈ 1，因此 x 的 L2 ≈ sqrt(hidden_dim)（可学习增益能把它放大或缩小）。增益按特征调制这个值，但绝不会让它偏离几个数量级。',
    },
  ],
};

export function RmsNormChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>RMSNorm：让激活值保持可控</h1>
        <p>
          <strong>一句话：</strong>Qwen3.5 的 24 层中，每一层都把注意力和 MLP 的输出加回同一条残差流——如果没有什么东西约束幅值，隐藏向量会一直增长，直到
          softmax 饱和、梯度更新失去意义。RMSNorm 就是那个便宜、几乎无状态、负责踩刹车的小技巧。这里的图表展示了一个真实
          prompt 的情况：残差流仍随深度爬升大约一个数量级（下方可见的那条线），但每个子模块的输入被 RMSNorm 按平在
          √hidden_dim 附近（粉色线）。
        </p>
        <p>
          在看公式之前，先感受一下刹车要解决的问题。&ldquo;softmax 饱和&rdquo;听起来很抽象——下面这个小部件把它变得具体。把一个
          token 的幅值往上推，看看下游的 softmax 如何从一个分散的分布坍缩到单一选项上，对其余选项视而不见：
        </p>

        <SaturatedSoftmax />

        <h2>公式，只有两行</h2>
        <p>
          RMSNorm 把每个 token 的隐藏向量除以它的均方根（root-mean-square），再用可学习的逐特征增益 <code>g</code>{' '}
          重新缩放：
        </p>
        <MathDisplay
          latex={String.raw`\begin{aligned} \text{RMS}(x) &= \sqrt{\text{mean}(x^2) + \varepsilon} \\ y_i &= \frac{x_i}{\text{RMS}(x)} \cdot g_i \end{aligned}`}
        />
        <p>
          用四个数亲手算一遍（ε 小到可以忽略，训练前 g = 1）。取 <code>x = [2, −1, 3, 0]</code>。逐项平方：{' '}
          <code>x² = [4, 1, 9, 0]</code>。求均值：<code>(4 + 1 + 9 + 0) / 4 = 3.5</code>。于是{' '}
          <code>RMS = √3.5 ≈ 1.87</code>，相除得到 <code>y ≈ [1.07, −0.53, 1.60, 0]</code>
          。大的元素被压小，零仍是零，向量保持原来的形状——改变的只是大小。
        </p>
        <p>
          这里的 <code>ε</code>（epsilon）是一个极小的常数，保证永远不会除以零；可学习增益 <code>g</code>{' '}
          是归一化器携带的唯一参数——每个特征一个标量，长度等于 <code>hidden_dim</code>。先除以 RMS，让向量的平均平方幅值落在
          1。正是这一步把向量的 <strong>L2 范数</strong>——L2 范数就是向量的长度，即各元素平方和的平方根——锚定在
          √hidden_dim 附近：
        </p>
        <ul>
          <li>经过 RMSNorm 后，各元素平方的均值 ≈ 1；</li>
          <li>
            因此这 1024 个元素的平方和 ≈ 1024（这里 <code>Σ</code> 表示&ldquo;对所有元素求和&rdquo;，所以{' '}
            <code>Σx²</code> ≈ 1024）；
          </li>
          <li>
            因此向量长度 <code>L2 = √(Σx²) ≈ √1024 = 32</code>——&ldquo;大约 32&rdquo;就是这么来的。
          </li>
        </ul>
        <p>
          然后让 <code>g</code> 按模型的需要给每个特征重新加权。初始时每个特征的 <code>g = 1</code>，所以训练前
          RMSNorm 只是把幅值标准化；训练中模型学到哪些特征更重要，并把这些信息写进 <code>g</code>。
        </p>

        <LearnedGainInspector />

        <p>和更早的 LayerNorm 摆在一起，简化之处一目了然：</p>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-[12px] leading-relaxed">
          <code>{`# RMSNorm — what every modern LLM uses
y = x / sqrt(mean(x²) + ε) * g
                              ↑
                       single learned vector

# LayerNorm — what the 2017 transformer used
y = (x - mean(x)) / sqrt(var(x) + ε) * g + b
     ↑—————————                  ↑
   subtract mean first        plus learned bias`}</code>
        </pre>
        <p>
          顺带解释 LayerNorm 那一行里的两个词，因为演示面板的统计框也会用到它们：<code>var(x)</code>——
          <strong>方差</strong>——是与均值的平方距离的平均值，而 <strong>std</strong>
          （标准差）是它的平方根。去掉减均值和偏置就是全部差别。Llama、Mistral、Qwen、DeepSeek——你能遇到的每个开源 LLM
          都在用 RMSNorm；更简单的公式在质量上经验性地不相上下，运行起来还略快一些。
        </p>

        <h2>pre-norm 与 post-norm</h2>
        <p>
          下面的示意图展示了归一化在一层内部的位置。现代模型堆叠全都采用 <strong>pre-norm</strong>
          （前置归一化）：残差高速公路保持未归一化，梯度因此能沿一条干净的恒等路径流过它。post-norm（后置归一化，2017
          年的原版做法）把归一化放在残差高速公路<em>上</em>——6 层没问题，24 层就很痛苦。
        </p>
        <NormFlowDiagram />

        <p>
          为什么 post-norm 在深层会变得痛苦？归一化不只重新缩放前向流动的激活值——它同样会重新缩放训练时<em>反向</em>
          流动的学习信号。在 post-norm 里，这个信号在 24 层中的每一层都要穿过一次归一化，24
          次小的缩放会累乘成一次大的缩放。pre-norm 的恒等路径把这 24 次全部绕开。
        </p>

        <h2>演示展示了什么</h2>
        <p>
          <strong>L2/tok across layers</strong>（跨层 L2/tok）图表（演示面板顶部）是本章的点睛之笔——一条曲线持续上漂，一条曲线被按住不动。下方的单层明细让你抽查任意一层的统计量。底部的
          <em>尺度不变性游乐场</em>则是一个合成的 256 维向量，你可以把它拉伸
          0.1×–10×，亲自确认输出幅值确实不依赖输入的尺度。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

/**
 * Side-by-side flow diagram showing where the norm sits inside a transformer
 * sub-block. The accent on the RMSNorm box highlights the placement
 * difference; the dashed line traces the residual highway so the "gradient
 * path" claim in the prose has a visual referent.
 */
function NormFlowDiagram() {
  return (
    <div className="my-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
      <NormFlowBox variant="post" />
      <NormFlowBox variant="pre" />
    </div>
  );
}

function NormFlowBox({ variant }: { variant: 'pre' | 'post' }) {
  const W = 220;
  const H = 200;
  const isPre = variant === 'pre';

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">
        {isPre ? 'Pre-norm（Qwen3.5 与现代 LLM）' : 'Post-norm（2017 年原版）'}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block h-auto w-full max-w-[260px]"
        role="img"
        aria-label={isPre ? 'pre-norm 子模块示意图' : 'post-norm 子模块示意图'}
      >
        <defs>
          <marker
            id={`arrow-${variant}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor" opacity={0.55} />
          </marker>
        </defs>
        {/* input label */}
        <text x={W / 2} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          x
        </text>
        {isPre ? (
          <>
            {/* split: norm path on the left, residual highway on the right (dashed) */}
            <line x1={W / 2} y1={18} x2={W / 2} y2={28} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={W / 2} y1={28} x2={70} y2={48} stroke="currentColor" strokeOpacity={0.4} />
            <line
              x1={W / 2}
              y1={28}
              x2={W / 2}
              y2={148}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
            <text x={W / 2 + 6} y={90} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              残差
            </text>
            <text x={W / 2 + 6} y={102} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              （未归一化）
            </text>
            {/* RMSNorm box (accented) */}
            <rect
              x={35}
              y={48}
              width={70}
              height={28}
              rx={5}
              className="fill-primary/15"
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={66} fontSize={11} textAnchor="middle" className="fill-primary">
              RMSNorm
            </text>
            <line x1={70} y1={76} x2={70} y2={94} stroke="currentColor" strokeOpacity={0.4} />
            {/* attn / mlp */}
            <rect
              x={35}
              y={94}
              width={70}
              height={28}
              rx={5}
              fill="currentColor"
              fillOpacity={0.06}
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={112} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              attn / mlp
            </text>
            {/* connect to + node */}
            <line x1={70} y1={122} x2={70} y2={148} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={70} y1={148} x2={W / 2 - 10} y2={148} stroke="currentColor" strokeOpacity={0.4} />
            <circle cx={W / 2} cy={148} r={10} fill="none" stroke="currentColor" strokeOpacity={0.5} />
            <text x={W / 2} y={152} fontSize={12} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              +
            </text>
            <line
              x1={W / 2}
              y1={158}
              x2={W / 2}
              y2={178}
              stroke="currentColor"
              strokeOpacity={0.55}
              markerEnd={`url(#arrow-${variant})`}
            />
            <text x={W / 2} y={194} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
              x_out
            </text>
          </>
        ) : (
          <>
            {/* residual stream on right (still un-normed inside the block) */}
            <line x1={W / 2} y1={18} x2={W / 2} y2={28} stroke="currentColor" strokeOpacity={0.4} />
            <line x1={W / 2} y1={28} x2={70} y2={48} stroke="currentColor" strokeOpacity={0.4} />
            <line
              x1={W / 2}
              y1={28}
              x2={W / 2}
              y2={114}
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeDasharray="3 3"
            />
            <text x={W / 2 + 6} y={80} fontSize={9} fill="currentColor" fillOpacity={0.5}>
              残差
            </text>
            {/* attn / mlp first */}
            <rect
              x={35}
              y={48}
              width={70}
              height={28}
              rx={5}
              fill="currentColor"
              fillOpacity={0.06}
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={70} y={66} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              attn / mlp
            </text>
            <line x1={70} y1={76} x2={70} y2={114} stroke="currentColor" strokeOpacity={0.4} />
            {/* + node */}
            <line x1={70} y1={114} x2={W / 2 - 10} y2={114} stroke="currentColor" strokeOpacity={0.4} />
            <circle cx={W / 2} cy={114} r={10} fill="none" stroke="currentColor" strokeOpacity={0.5} />
            <text x={W / 2} y={118} fontSize={12} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
              +
            </text>
            {/* RMSNorm sits on the residual highway */}
            <line x1={W / 2} y1={124} x2={W / 2} y2={140} stroke="currentColor" strokeOpacity={0.4} />
            <rect
              x={W / 2 - 35}
              y={140}
              width={70}
              height={28}
              rx={5}
              className="fill-primary/15"
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            <text x={W / 2} y={158} fontSize={11} textAnchor="middle" className="fill-primary">
              RMSNorm
            </text>
            <line
              x1={W / 2}
              y1={168}
              x2={W / 2}
              y2={178}
              stroke="currentColor"
              strokeOpacity={0.55}
              markerEnd={`url(#arrow-${variant})`}
            />
            <text x={W / 2} y={194} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
              x_out
            </text>
          </>
        )}
      </svg>
      <div className="mt-1 text-[11px] text-muted-foreground">
        {isPre
          ? '归一化只作用于子模块的输入。残差高速公路保持未归一化——梯度沿干净的恒等路径流动。24 层以上依然稳定。'
          : '归一化位于残差高速公路上。梯度在每一层都要多挤过一道归一化。6 层训练没问题；超过 20 层就很痛苦。'}
      </div>
    </div>
  );
}

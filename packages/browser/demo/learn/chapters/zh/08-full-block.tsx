// 中文（简体）版第 8 章正文 — 完整的 Transformer 块。
// Demo 组件保持英文，仍从英文模块 ../08-full-block.tsx 渲染；本文件只翻译
// 正文 Body 与 learning 数据。术语表的 term 字段保持英文原样。

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { GdnTeaser } from '../../widgets/GdnTeaser';
import { ResidualStream } from '../../widgets/ResidualStream';
import { StackCollapsedView } from '../../widgets/StackCollapsedView';

// Qwen3.5-0.8B; matches the model's hidden_size.
const HIDDEN_DIM = 1024;

/**
 * Scaffolding metadata for chapter 8 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[7].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'full-block',
  objective: '端到端读懂一个 transformer 层：前置归一化、注意力、残差、前置归一化、MLP、残差，如此重复。',
  problem: '每个子块此前都是单独介绍的；你还需要看清它们是如何接线组合的，以及整个堆叠是如何叠起来的。',
  minutes: 8,
  glossary: [
    {
      term: 'decoder layer',
      definition:
        '一个 pre-norm 块：x_attn = x + Attention(RMSNorm(x))；x_out = x_attn + MLP(RMSNorm(x_attn))。Qwen3.5-0.8B 堆叠了 24 个这样的层。',
    },
    {
      term: 'residual stream',
      definition: '贯穿整座塔全部深度、未经归一化的隐藏状态。每个块都从它读取，并把一个增量写回去。',
    },
    {
      term: 'sub-block',
      definition: '注意力或 MLP。每一层恰好包含两个子块，每个子块都包在自己的归一化 + 残差里。',
    },
    {
      term: 'hybrid attention',
      definition: 'Qwen3.5 的配方：每第四层使用 softmax 注意力，其余层使用线性变体（GatedDeltaNet）。',
    },
    {
      term: 'linear attention',
      definition: '一种近似注意力，每个 token 占用固定的内存。用一部分精确回忆能力，换取极其便宜的长上下文解码。',
    },
    {
      term: 'capture point',
      definition:
        '层内一个有名字的位置（pre_attn_input、mlp_output 等），inspector 在这里把激活统计量读出来。',
    },
  ],
  takeaways: [
    '每一层的形状完全相同——两组"前置归一化 + 子块 + 残差"——这个形状堆叠了 24 次。',
    '残差流的幅度随深度增长，尽管每个单独子块的输出始终很小。',
    'Qwen3.5 并非纯 transformer——24 层中只有 6 层使用 softmax 注意力，其余运行线性注意力。',
  ],
  exercise: {
    prompt:
      "自动运行结束后，拖动 3D 塔正对自己，然后分别点击第 3 层和第 4 层。'Layer 3 - Full attention' 标签与 'Layer 4 - Linear attention' 有什么不同？暖色与冷色的环在整座塔上又是如何分布的？",
    answer:
      "第 3 层（以及第 7、11、15、19、23 层）显示 'Full attention' 标签，使用从橙到红的暖色渐变；其余层显示 'Linear attention'，使用从靛蓝到紫的冷色渐变。颜色梯度还显示出残差流幅度随塔向上攀升——这正是这条\"高速公路\"不断累积贡献的直观画面。",
  },
  quiz: [
    {
      id: 'q1-block-order',
      prompt: '在 Qwen3.5 的一层（pre-norm）中，操作的实际顺序是什么？',
      options: [
        {
          id: 'a',
          label: 'RMSNorm -> Attention -> 加回残差；RMSNorm -> MLP -> 加回残差。',
        },
        {
          id: 'b',
          label: 'Attention -> RMSNorm -> MLP -> RMSNorm -> 最后才加残差。',
        },
        {
          id: 'c',
          label: 'Attention -> MLP -> 最后只做一次 RMSNorm。',
        },
      ],
      correctId: 'a',
      explanation: 'Pre-norm 把每个子块单独包裹起来。每个子块都把结果写回未经归一化的残差流。',
    },
    {
      id: 'q2-hybrid-full',
      prompt: 'Qwen3.5-0.8B 的 24 层中，有多少层使用你在第 4 章读到的 softmax 注意力？',
      options: [
        { id: 'a', label: '全部 24 层。' },
        { id: 'b', label: '6 层——每四层一个。' },
        { id: 'c', label: '12 层——每隔一层。' },
      ],
      correctId: 'b',
      explanation: '第 3、7、11、15、19、23 层是全量 softmax 注意力。其余 18 层运行 GatedDeltaNet 线性注意力。',
    },
    {
      id: 'q3-residual-grows',
      prompt: '为什么沿塔向上扫视时，残差流的逐 token L2 会增长？',
      options: [
        {
          id: 'a',
          label: '每一层都把子块输出加进残差流；尽管每次写入都很小，幅度仍会不断累积。',
        },
        {
          id: 'b',
          label: '更深的层在 RMSNorm 中使用更大的可学习增益。',
        },
        {
          id: 'c',
          label: '模型在更深的层把数值精度提升到更高位数。',
        },
      ],
      correctId: 'a',
      explanation: '残差流是一系列加法的总和，从不被覆盖。即便是很小的贡献，跨过 24 层也会复利式累积。',
    },
  ],
};

export function FullBlockChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>完整的 Transformer 块</h1>
        <p>
          前面五章把各个组成部分——注意力、多头与 GQA、RoPE、RMSNorm、门控 MLP——逐一单独介绍，仿佛它们彼此孤立。
          事实并非如此。Qwen3.5-0.8B 的 24 个解码器层中的每一层，都按<em>同一种</em>固定模式把它们拼接在一起，然后模型把这个模式堆叠
          24 次。本章拉远视角：先看完整的一层，再看由这些层堆成的一座“塔”。
        </p>

        <h2>Pre-norm：归一化、子块、残差</h2>
        <p>
          Qwen3.5 采用现代的 <strong>pre-norm</strong>（前置归一化）方案。每一层内部有两个子块（注意力和
          MLP），它们的包裹方式完全相同：先归一化输入，对归一化后的副本运行子块，再把结果加回残差流。
        </p>
        <MathDisplay
          latex={String.raw`\begin{aligned} x_\text{attn} &= x + \text{Attention}\bigl(\text{RMSNorm}(x)\bigr) \\ x_\text{out} &= x_\text{attn} + \text{MLP}\bigl(\text{RMSNorm}(x_\text{attn})\bigr) \end{aligned}`}
        />
        <p>
          两次 RMSNorm、两次残差相加、一次注意力、一次 MLP——这就是完整的一层。下面的 3D
          组件把它画成一座又高又窄的塔：残差流从底部贯穿到顶部，塔中你看到的每一个“环”就是这 24 层中的一层。
        </p>

        <h2>每个部件各自贡献什么</h2>
        <ul>
          <li>
            <strong>注意力</strong>（第 4 章）——整层中唯一让信息<em>跨 token</em>{' '}
            流动的地方。没有它，每个位置都会各自独立演化。
          </li>
          <li>
            <strong>多头与 GQA</strong>（第 5 章）——注意力由许多头并行运行，这些头共享一个更小的 K/V
            投影池，模型因此能同时关注多种模式，又不会让 KV 缓存膨胀（KV 缓存保存每个 token 的 key 和
            value，避免每一步都重新计算——后面有专门一章）。
          </li>
          <li>
            <strong>RoPE</strong>（第 6 章）——在注意力内部施加在 Q 和 K 上的旋转。模型正是靠它知道 token 5 在 token 3
            之后，而不需要单独的位置嵌入。
          </li>
          <li>
            <strong>RMSNorm</strong>（第 7 章）——施加在每个子块<em>之前</em>。它把输入的幅度压到大约{' '}
            <code>sqrt(hidden_dim)</code> ≈ {Math.sqrt(HIDDEN_DIM).toFixed(0)}，让注意力分数和 MLP 激活不会爆炸。
          </li>
          <li>
            <strong>MLP</strong>（第 8 章）——使用 SwiGLU 门控非线性的逐 token
            前馈网络。它占据了模型参数中最大的几块之一，是逐 token 特征计算发生的地方。
          </li>
        </ul>

        <StackCollapsedView />

        <h2>残差流</h2>
        <p>
          解码器 LLM 中最重要却最少被点名的概念是<strong>残差流</strong>：一个逐 token
          的向量，从嵌入进入层堆叠，被每个子块<em>写入</em>，最终在顶部被 LM head——把向量变成下一个 token
          分数的最终投影（下一章）——读取。层内没有任何操作会<em>替换</em>它——注意力和 MLP 的结果都是<em>加</em>
          回去的。每一步操作实际上都是 <code>h := h + Δ</code>。
        </p>
        <p>
          具体来说，用一个只有 4 个槽位的玩具残差流：设 <code>h = [1.2, −0.4, 0.8, 0.3]</code>，注意力子块算出{' '}
          <code>Δ = [0.1, 0.3, −0.1, 0.0]</code>。新的残差流是 <code>h + Δ = [1.3, −0.1, 0.7, 0.3]</code>。
          这次写入只轻轻推动了几个槽位——最后一个原封不动地通过，其余槽位也保留了大部分旧值。
        </p>
        <p>
          下面的动画把这件事演示得很具体：一个隐藏向量、两层的写入（注意力和 MLP 各两次）、顶部的 LM head
          读取累积的总和。柱子不断变长，因为每次写入都是严格的加法——残差流从不变短。
        </p>

        <ResidualStream />

        <p>
          由此带来两个结果。第一，<strong>梯度</strong>：即使在 24
          层的堆叠里，反向传播也能沿恒等路径直接穿过，这正是深层 transformer
          能训练起来的原因。没有残差时，梯度（调整权重的训练信号——训练一章会讲）会被每一层的变换（它的雅可比矩阵——这里先不必纠结这个词）反复相乘，跨过这么深的堆叠后
          <em>往往</em>会消失或爆炸（在向后穿过许多层的过程中缩小到几乎为零，或者越来越大）；残差的恒等路径正是让它得以存活的关键。第二，
          <strong>幅度</strong>：残差流的 L2（向量的长度——各分量平方和的平方根）随深度增长，因为每一层都加入非零的贡献，而每个
          <em>单独</em>子块的输出始终很小。右侧的 3D 组件按层输出的逐 token L2
          给每个环着色——沿塔向上扫视，可以看到幅度总体在攀升（趋势向上，个别层可能回落），大致呼应动画里不断生长的柱子。
        </p>
        <p className="text-muted-foreground">
          <strong>心智模型：</strong>把残差流想成一块由全部 24 层共享的工作内存。每一层并不是<em>变换</em>
          这份表示，而是向一个共享的累加和<em>贡献</em>自己的部分。“思考”就是这个累加过程。
        </p>

        <h2>把这座塔读成一次“含义”的攀升</h2>
        <p>
          3D 组件按原始幅度（逐 token L2）给每个环着色，而这确实也是 inspector
          唯一测量的东西。但在同一座塔之上，还叠着一个更柔软、更难精确指认的故事：残差流<em>携带</em>
          的内容，往往会随着你向上攀升而发生变化。对标准 transformer
          的探针研究提示出一个粗略趋势——靠前的层偏向局部、表层的结构（哪个 token
          挨着哪个、基本语法），靠后的层则偏向更抽象、句子层面的含义。请把它当成一种宽松的直觉，而非一次测量：这是人们从训练好的网络里读出的统计倾向，而不是盖在某个具体层上的标签。在这里它甚至更模糊，因为我们的
          24 层里有 18 层是 <em>GatedDeltaNet</em>，而不是那类研究大多考察的 softmax
          注意力堆叠——所以"靠前→语法、靠后→抽象"这幅图景，对这个特定模型而言充其量是个手势，并不是逐层的保证。
        </p>
        <p>
          这次攀升你在上一章其实已经见过。
          <ChapterLink chapterId="attention">注意力一章里的多义性演示</ChapterLink>喂入了三条 prompt，它们都以字节完全相同的
          token <code>·bank</code> 结尾——金融、航空、河畔——所以 embedding 查表在那个最后位置交给层堆栈的是<em>同一个</em>
          起点向量，可模型对下一个 token 的预测却彻底分了岔。这正是同一种分岔，如今换个角度读成一次沿塔的攀升：三份副本在底部进入时一模一样，随着上下文不断累积，每一层的写入都把它们再往外推开一点点，直到塔顶它们分别指向
          <code>·account</code>、<code>·angle</code> 和一个平实的句末。这里并没有提出任何新主张——它只是你已经见过的那次分岔的“深度视角”。
        </p>

        <h2>混合注意力：并非每一层都是“全量”的</h2>
        <p>
          Qwen3.5 是一个<strong>混合</strong>堆叠。大多数层使用一种叫 <em>GatedDeltaNet</em>{' '}
          的快速线性注意力变体；只有每第四层（0.8B 模型上索引为 3、7、11、15、19、23，从 0 计数）使用第 4 章的经典{' '}
          <code>softmax(QKᵀ/√d)</code> 公式。线性注意力层在长上下文紧张的内存预算下完成几乎全部的跨 token
          混合；全量注意力层按固定间隔出现，负责线性注意力做不到的重活——例如精确回看某个早先的 token；第 12
          章会展示一个线性层恰好在这件事上失败。3D
          组件用更暖的颜色高亮全量注意力层，让你一眼看出这种交错。第 12
          章将深入讲解这套混合配方的原理，以及它为什么成了高吞吐开源 LLM 的新默认。
        </p>

        <GdnTeaser />

        <h2>这个组件展示什么</h2>
        <p>
          按下 <em>Run</em>，即可为一次前向传播捕获每层全部七个隐藏状态统计量。3D 塔渲染 24
          个堆叠的片层——每层一个——按层输出的逐 token L2
          着色。拖动可旋转，滚轮可缩放，点击某一层可在侧边面板查看它的统计量。面板里的迷你柱状图沿所选层内的七个捕获点逐一走过，让你追踪信号从层输入、穿过两个子块、再从顶部输出的全过程。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

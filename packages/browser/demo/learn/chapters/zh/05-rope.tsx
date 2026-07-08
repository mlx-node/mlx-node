import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { PermutationInvariance } from '../../widgets/PermutationInvariance';
import { RopeRelativeDial } from '../../widgets/RopeRelativeDial';

/**
 * 第 6 章——位置编码（RoPE）的中文正文。
 *
 * 纯 JS：给定模型配置后 RoPE 的数学是确定性的，所以本章直接可视化数学本身，
 * 不调用 worker。交互演示（RopeDemo）保持英文版本，由英文模块渲染。
 */

// Qwen3.5-0.8B；见 crates/mlx-core/src/models/qwen3_5/config.rs 以及模型
// config.json 中的 `text_config.rope_parameters` 块。RoPE 只旋转每个 256 维
// 头的前 `head_dim * partial_rotary_factor` = 64 个特征，即 32 对。
//
// 脚注：真实配置其实是多模态/分段的 RoPE（mRoPE，mrope_section [11, 11, 10]），
// 因为基座是一个还要给图像切块定位的视觉语言模型。对纯一维文本生成来说，
// 该变体退化为本章可视化的普通一维 RoPE，所以我们只展示一维核心、略去多模态分段。
const HEAD_DIM = 256;
const PARTIAL_ROTARY_FACTOR = 0.25;
const ROPE_DIMS = HEAD_DIM * PARTIAL_ROTARY_FACTOR; // 64 个被旋转的维度
const ROPE_THETA = 10_000_000;
const MAX_POSITION = 262_144;

const DOT_QUERY_POS = 50;

/**
 * 第 6 章的脚手架元数据中文译文——驱动 `<ChapterFrame>` 渲染的页眉、
 * 术语表、要点、练习与随堂测验。
 * `chapterId` 必须与 `learn/chapters.ts` 中 `CHAPTERS[5].id` 一致。
 */
export const learning: ChapterLearningData = {
  chapterId: 'rope',
  objective: '解释 RoPE 如何通过逐对旋转把 token 顺序注入注意力，而无需可学习的位置嵌入。',
  problem: "自注意力是置换不变的——没有位置信号，模型无法区分 'cat sat on mat' 和 'mat sat on cat'。",
  minutes: 8,
  glossary: [
    {
      term: 'RoPE',
      definition:
        '旋转位置嵌入（Rotary positional embedding）。在点积之前，把 Q 和 K 的每个 (x_2i, x_2i+1) 维度对旋转 m * theta_i 的角度。',
    },
    {
      term: 'pair frequency (theta_i)',
      definition: '每个维度对的旋转速率：theta_i = base^(-2i / rope_dims)。i 小转得快，i 大转得慢。',
    },
    {
      term: 'rope_theta (base)',
      definition: '频率指数的底数。Qwen3.5 用 1e7，远高于最初的 1e4——把频谱拉长以支持长上下文。',
    },
    {
      term: 'partial rotary factor',
      definition: 'RoPE 旋转每个头维度的比例。Qwen3.5 用 0.25，所以 256 个头维度中只有 64 个被旋转；其余原样通过。',
    },
    {
      term: 'relative-position property',
      definition: 'RoPE(q,m) 与 RoPE(k,n) 的点积只取决于 m-n，所以模型只需要学习相对偏移。',
    },
    {
      term: 'unitary rotation',
      definition: '旋转保持向量范数不变——RoPE 改变向量指向的方向，从不改变它们的大小。',
    },
  ],
  takeaways: [
    'RoPE 把位置编码成旋转而不是加法——向量大小保持恒定，只有角度携带位置信号。',
    '低序号的维度对是高频的（精细的局部位置）；高序号的维度对是低频的（粗粒度的全局位置）。',
    'RoPE 之后的点积只取决于 m - n，这正是带 RoPE 的模型能外推到训练中从未见过的长度的原因。',
  ],
  exercise: {
    prompt:
      "把 'Token position m' 滑块从 0 拖到 100。对比高频旋转面板（pair 0）和低频面板（pair 28）。哪一个先转完一整圈？当 m=100 时，另一个转了多少？",
    answer:
      'pair 0 到 m=100 时已经转过很多整圈（在 Qwen3.5 的频谱里它的频率大约是 1 rad/token）。pair 28 几乎没动——它的频率小到指针还停在 3 点钟方向附近。这个差距正是使用一整组频率谱的全部意义：一次操作同时编码短程位置和长程位置。',
  },
  quiz: [
    {
      id: 'q1-why-rotate',
      prompt: 'RoPE 解决了纯自注意力解决不了的什么问题？',
      options: [
        { id: 'a', label: '自注意力的输出幅度太大。' },
        {
          id: 'b',
          label: '自注意力是置换不变的——在有什么东西把顺序注入进来之前，它对 token 的先后一无所知。',
        },
        {
          id: 'c',
          label: '自注意力无法表示负值。',
        },
      ],
      correctId: 'b',
      explanation:
        '把输入打乱（输出也相应打乱），注意力算出的东西不变。位置必须来自某个地方——RoPE 把它注入 Q 和 K。',
    },
    {
      id: 'q2-low-vs-high',
      prompt: '在 RoPE 下，最终编码“粗粒度全局位置”的是哪些维度对？',
      options: [
        {
          id: 'a',
          label: '高序号的维度对——它们的频率最低，旋转得足够慢，在很长的上下文中仍能保持可区分。',
        },
        {
          id: 'b',
          label: '低序号的维度对，因为它们的频率最高。',
        },
        {
          id: 'c',
          label: '是随机的——RoPE 在每次训练中随机分配角色。',
        },
      ],
      correctId: 'a',
      explanation:
        'theta_i = base^(-2i/d)，i 越大 theta_i 越小。这些维度对即使跨越数万个 token 也几乎不旋转，这正是“粗粒度全局位置”的样子。',
    },
    {
      id: 'q3-relative-pos',
      prompt: '它为什么被称为相对位置性质？',
      options: [
        {
          id: 'a',
          label: '经过 RoPE 的 Q 和 K 的点积只取决于偏移量 (m - n)，与绝对位置 m 和 n 无关。',
        },
        {
          id: 'b',
          label: 'RoPE 的位置是相对某个选定的锚点 token 存储的。',
        },
        {
          id: 'c',
          label: '旋转总是把两个相邻的 token 联系起来。',
        },
      ],
      correctId: 'a',
      explanation:
        '把代数算一遍，对位置的依赖就坍缩成 m - n。这就是在 4k token 上训练的模型能外推到更长上下文的原因。',
    },
  ],
};

export function RopeChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>位置编码：模型如何知道 token 的顺序</h1>
        <p>
          自注意力有一个奇怪的性质：它是<strong>置换不变（permutation-invariant）</strong>的。如果把输入 token
          打乱，每个 token 的注意力输出也会跟着一起被打乱，但注意力层计算出的关系本身不会变。没有外力帮助，Transformer
          真的分不清 <em>"the cat sat on the mat"</em> 和 <em>"mat the on sat cat the"</em>
          。必须有什么东西把顺序重新注入进来。
        </p>
        <p>
          先亲眼看看。下面是一个由三个 token
          组成的玩具集合，没有附带任何位置信息——随意重新排列它们，注意力权重纹丝不动：
        </p>
        <PermutationInvariance />

        <h2>两个更早的想法</h2>
        <ul>
          <li>
            <strong>可学习的位置嵌入。</strong>早期的 GPT、BERT 等模型给 0..N-1 的每个位置都配一个可学习的向量，并把它
            <em>加</em>到 token
            嵌入上。简单，但模型无法外推到训练中从未见过的位置，而且必须事先定死最大长度。
          </li>
          <li>
            <strong>正弦位置嵌入。</strong>2017 年最初的 Transformer 用不同频率的固定正弦波，同样加到 token
            嵌入上。理论上可以外推，但实践中超过训练窗口后质量会退化。
          </li>
        </ul>

        <h2>RoPE 的想法：旋转，而不是相加</h2>
        <p>
          <strong>旋转位置嵌入</strong>（Rotary Positional Embedding，RoPE）不去动嵌入本身，而是把位置直接烘焙进注意力的数学里。诀窍是在点积计算
          <em>之前</em>，把 query 和 key 向量按一个取决于 token 位置的角度<strong>旋转</strong>。
        </p>
        <p>
          具体来说，RoPE 把相邻的特征维度配成对——<code>(x_0, x_1)</code>、<code>(x_2, x_3)</code>、…
          ——并把每一对当作 2D 平面上的一个点。对维度对序号 <code>i</code> 和<strong>旋转维度</strong>{' '}
          <code>d</code>（RoPE 实际旋转的特征数——不一定是完整的头维度；下文马上细说），频率是：
        </p>
        <MathDisplay latex={String.raw`\theta_i = \text{base}^{-2i / d}`} />
        <p>
          用文字读这个式子：随着维度对序号 <code>i</code> 增大，指数 <code>−2i/d</code> 越来越负，所以{' '}
          <MathDisplay latex={String.raw`\theta_i`} inline />{' '}
          越来越小。低序号的对转得快（追踪精细的局部位置）；高序号的对转得慢（追踪粗粒度的长程位置）。
        </p>
        <p>
          在 token 位置 <code>m</code>，这一对被旋转 <MathDisplay latex={String.raw`m \cdot \theta_i`} inline />{' '}
          的角度：
        </p>
        <p>
          <strong>旋转矩阵</strong>做的事就是把一个 2-D 点——想象一根钟表指针——旋转角度{' '}
          <MathDisplay latex={String.raw`\theta`} inline />
          ；下面的四个三角函数项只是计算旋转后落点的配方。再往下的钟面表盘展示的正是这种旋转。这里的角度用
          <strong>弧度（radian）</strong>度量——在这个单位里一整圈是{' '}
          <MathDisplay latex={String.raw`2\pi \approx 6.28`} inline />，所以“1 rad/token”
          的速率大约是每步六分之一圈。
        </p>
        <MathDisplay
          latex={String.raw`\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}`}
        />
        <p>
          亲手把这个矩阵算一遍，就用 pair 0——它的频率恰好是 <code>θ₀ = base⁰ = 1</code> 弧度/token，所以 pair 0
          每步转 1 rad。取位置 <code>m = 2</code> 处的 <code>(1, 0)</code>：它旋转 2 rad，落在{' '}
          <code>(cos 2, sin 2) ≈ (−0.42, 0.91)</code>。长度不变，方向变了——而这正是你将在下方表盘
          0（高频钟面）上看到的旋转。
        </p>
        <p>
          这里为了直观把维度画成相邻配对（<code>x_0</code> 配 <code>x_1</code>）；Qwen3.5
          实际如何排列它们是一个实现细节，第一遍阅读可以跳过（见下方<strong>进阶</strong>）。
        </p>

        {/* 可选的深入阅读：rotate-half 的配对布局是一个实现细节，会打断初学者
            “旋转而不是相加” 的主线——无论哪种布局，数学、角度和频率都完全一样。
            默认折叠、可跳过。用原生 <details> 保持键盘可聚焦；样式仿照项目的
            Glossary 折叠面板（与 04-multihead-gqa.tsx 一致）。 */}
        <details className="group not-prose my-4 rounded-md border border-border bg-muted/30 px-4 py-3 text-sm">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-foreground/90 [&::-webkit-details-marker]:hidden">
            <span className="font-medium">
              进阶：维度对实际是怎么排列的（rotate-half）{' '}
              <span className="text-muted-foreground">· 可选，给好奇的读者</span>
            </span>
            <span
              aria-hidden="true"
              className="text-xs text-muted-foreground transition-transform group-open:rotate-90"
            >
              ▸
            </span>
          </summary>
          <div className="mt-3 space-y-2 text-[13px] leading-relaxed text-foreground/80">
            <p>
              这里为了直观把维度画成相邻配对（<code>x_0</code> 配 <code>x_1</code>），但 Qwen3.5——和大多数
              Llama/HF 风格的模型一样——用的是数学上等价的 <em>rotate-half</em> 布局：维度 <code>i</code>{' '}
              与旋转块后一半中的对应维度 <code>i + d/2</code>{' '}
              配对。同样的角度、同样的频率、同样的相对位置性质；不同的只是<em>哪</em>两个维度共享一次旋转。
            </p>
          </div>
        </details>

        <p>
          旋转是<strong>酉变换（unitary）</strong>——它保持向量范数不变——所以 RoPE 不改变 Q 和 K
          的大小，只改变它们指向的方向。位置信息骑在角度上，而不是大小上。
        </p>

        <h2>宽广的频率范围</h2>
        <p>
          低序号的维度对（<code>i</code> 小）拿到最高的频率、转得最快——几个 token
          就转完一整圈。高序号的维度对频率小到几乎为零，即使跨越数万个 token 也几乎不动。直觉是：高频对编码
          <strong>精细的局部位置</strong>（“我离邻居是不是 3 个 token？”），低频对编码
          <strong>粗粒度的全局位置</strong>（“我在文档的前半部分还是后半部分？”）。
        </p>
        <p>
          Qwen3.5 的头维度是 <code>{HEAD_DIM}</code>，部分旋转因子（partial rotary factor）是{' '}
          <code>{PARTIAL_ROTARY_FACTOR}</code>（所以每个头只有前 <code>{ROPE_DIMS}</code>{' '}
          个特征被旋转——其余原样通过），RoPE 底数则大得不寻常：<code>{ROPE_THETA.toExponential(0).replace('e+', 'e')}</code>
          。大底数把频谱拉长，使最低频的维度对在模型 <code>{MAX_POSITION.toLocaleString()}</code> token
          的上下文窗口内几乎不旋转——这是长上下文外推的关键配料之一。
        </p>
        <p>
          为什么只旋转头的四分之一？因为未被旋转的那 {HEAD_DIM - ROPE_DIMS}{' '}
          个维度完全不携带位置——它们做纯粹的内容匹配（“这个 key 的含义是不是我的 query
          正在找的？”）。位置只需要骑在向量的一部分上；其余部分可以只关心语义本身。
        </p>
        <p>
          还有一个 Qwen3.5 混合架构特有的细节：这个 RoPE 旋转<strong>只施加在 6 个全量注意力层上</strong>
          （如上所述，这些层旋转 <code>{HEAD_DIM}</code> 个头维度中的前 <code>{ROPE_DIMS}</code> 个）。其余 18
          层是线性（GatedDeltaNet）层，通过循环再加一段短因果卷积来隐式地编码位置——完全不用 RoPE。
        </p>

        <h2>相对位置性质</h2>
        <p>
          这是让 RoPE 真正“通”的部分。取 <code>RoPE(q, m)</code> 与 <code>RoPE(k, n)</code> 的点积，结果
          <em>只取决于差值</em> <code>m - n</code>，与 <code>m</code> 和 <code>n</code>{' '}
          各自的取值无关。模型永远不需要学习“位置 1,427”是什么意思——它只需要学习注意力在<em>相对</em>
          偏移上应该如何表现，而这天然地泛化到它从未见过的长度。
        </p>
        <p>
          右侧第三个面板把这一点变得具体。把 query 位置固定在 <code>m = {DOT_QUERY_POS}</code>、变动 key 位置{' '}
          <code>n</code>，旋转后的点积在 <code>n - m = 0</code>{' '}
          处出现尖锐峰值，并随着距离拉远向两侧（带着涟漪地）衰减。这种衰减正是让注意力天然“偏爱”近处 token
          胜过远处 token 的归纳偏置——而且完全不需要逐位置的参数。
        </p>
        <p>你可以直接感受这种只依赖偏移量的性质。把两个位置一起平移，分数纹丝不动：</p>
        <RopeRelativeDial />

        <p className="mt-6 text-muted-foreground">
          本章直接可视化 RoPE 的数学——不需要任何模型推理。想看看它在真实注意力分数上的效果？打开
          <em>自注意力</em>（第 4 章），观察分数图如何偏向对角线。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

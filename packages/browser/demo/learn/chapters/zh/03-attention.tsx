// 第 4 章「自注意力」的简体中文译文。与英文版 ../03-attention.tsx 平行维护：
// 正文组件与 learning 数据同名导出；AttentionDemo（右侧实时演示）保留在英文
// 模块中渲染，因此这里不重复导出，也不引入仅供演示使用的依赖。

import { findChapter } from '../../chapters';
import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { SubChapterNav } from '../../scaffolding/SubChapterNav';
import { CausalMaskToggle } from '../../widgets/CausalMaskToggle';
import { CausalMaskVisual } from '../../widgets/CausalMaskVisual';
import { PolysemyDivergence } from '../../widgets/PolysemyDivergence';
import { ScaledSoftmax } from '../../widgets/ScaledSoftmax';
import { ToyAttentionExample } from '../../widgets/ToyAttentionExample';

export const learning: ChapterLearningData = {
  chapterId: 'attention',
  objective: '看懂一张 softmax(QK^T/sqrt(d))V 注意力图，并解释其中每个格子、每一行、每一列的含义。',
  problem: '没有注意力，Transformer 就没有任何途径让一个 token 的表示依赖于它前面的内容。',
  minutes: 9,
  glossary: [
    {
      term: 'query (Q)',
      definition: '由 token 经学习投影得到的向量，表达“我在找什么？”——每个 token、每个头各有一个 query 向量。',
    },
    {
      term: 'key (K)',
      definition: '表达“如果你看向我，我能提供什么？”的学习投影——通过点积与 Q 比较。',
    },
    {
      term: 'value (V)',
      definition: '携带实际内容的学习投影：当一个 token 被注意到时，它真正贡献出去的就是这个向量。',
    },
    {
      term: 'attention pattern',
      definition: 'softmax(QK^T/sqrt(d)) 的一行：单个 query token 在更早位置上的概率分布。',
    },
    {
      term: 'scaled dot product',
      definition: 'QK^T 除以 sqrt(d_head)。这个缩放让 softmax 保持在梯度不会随 d_head 增大而消失的区间。',
    },
    {
      term: 'causal mask',
      definition: '把分数矩阵的上三角设为 -inf，使位置 i 只能注意位置 0..i。',
    },
    {
      term: 'self-attention vs cross-attention',
      definition:
        '自注意力（本章）的 Q、K、V 都来自同一条运行中的序列——token 互相看彼此。cross-attention 是它来自机器翻译的同胞机制：query 来自一条序列，key/value 来自另一条。我们这条纯文本 decoder 路径只会做自注意力。',
    },
  ],
  takeaways: [
    '热力图的每一行是一个 query token 的注意力分布；softmax 之后每行的和恒为 1。',
    '下三角形状不是装饰——它是因果掩码，是让训练保持诚实的唯一机制，也是所有位置能够并行训练的原因。',
    '除以 sqrt(d_head) 的缩放，让注意力在头变宽时不会令 softmax 坍缩到单个格子上。',
  ],
  exercise: {
    prompt:
      '在默认提示词 “The cat sat on the” 上点击 Run，然后在热力图里看最下面一行（预测之前的最后一个 token）。哪个更早的 token 拿到的注意力最多？这说明你正在看的这个头在做什么？',
    answer:
      '多数头会把最亮的格子放在前一个 token（“ the”）或 “ cat” 上——说明这个头在做“关注上一个 token”与“关注我正要接续其动词的那个名词”的某种混合。不同的层/头组合看起来会截然不同——靠前层的 head 0 常常近似一条对角线，而更深的头会把注意力撒得更远。',
  },
  quiz: [
    {
      id: 'q1-row-sum',
      prompt: '注意力热力图中单独一行的数值加起来等于多少？',
      options: [
        { id: 'a', label: '大约是 d_head，因为有 sqrt(d_head) 缩放。' },
        { id: 'b', label: '恰好为 1——softmax 是逐行应用的。' },
        { id: 'c', label: '取决于未归一化的点积碰巧是多少。' },
      ],
      correctId: 'b',
      explanation: '对每一行做 softmax 会得到一个概率分布：softmax 之后矩阵的每一行之和都是 1。',
    },
    {
      id: 'q2-scale-by-sqrt-d',
      prompt: '为什么分数矩阵在 softmax 之前要除以 sqrt(d_head)？',
      options: [
        {
          id: 'a',
          label: '点积会随 d_head 增大而变大，不缩放的话 softmax 会饱和、梯度会消失。',
        },
        {
          id: 'b',
          label: '这是一个归一化步骤，让权重之和为 1。',
        },
        {
          id: 'c',
          label: '为了让结果不依赖于词表的选择。',
        },
      ],
      correctId: 'a',
      explanation: '除以 sqrt(d_head) 使分数的方差在头维度增大时大致不变，softmax 因此保持在信息量充足的区间。',
    },
    {
      id: 'q3-why-triangular',
      prompt: '为什么注意力热力图是下三角，而不是完整的方阵？',
      options: [
        {
          id: 'a',
          label: '空白格子对应分词器从未产生过的 token。',
        },
        {
          id: 'b',
          label: '因果掩码在 softmax 之前把上三角设为负无穷，所以位置 i 的 token 只能看位置 0..i。',
        },
        {
          id: 'c',
          label: '跳过上半部分可以节省 GPU 显存。',
        },
      ],
      correctId: 'b',
      explanation: '掩码的存在是为了让模型在训练时无法偷看未来的 token——否则“预测下一个 token”就变得毫无难度。',
    },
  ],
};

export function AttentionChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>自注意力：每个 token 如何看到其他所有 token</h1>
        <p>
          到本章为止，我们已经把文本变成了 token（第 2 章），又把 token 变成了向量（第 3 章）。现在有趣的问题是：一个
          token <em>是怎么知道句子其余部分的？</em>现代 LLM 的答案是<strong>自注意力（self-attention）</strong>。
        </p>

        <h2>直觉</h2>
        <p>
          直观地说，每个 token 会扫过它前面的所有
          token，把自己需要的信息拉进来——就像重读一遍句子，把注意力集中在对接下来的内容最要紧的那几个词上。
        </p>
        <p>
          想象你是模型，有人递给你提示词 <code>"The cat sat on the ___"</code>，让你猜下一个词。要猜得好，你必须回看前面的
          token：<code>cat</code> 告诉你主语是一只动物，<code>sat</code> 告诉你它正停在某个东西上，<code>on the</code>{' '}
          告诉你接下来要出现一个名词。自注意力就是这种“回看”的可学习、可被训练调节的版本：对序列中的每一个
          token，模型决定它应该对其他每个 token 投入<em>多少</em>注意力。右侧面板运行的正是这条提示词，并展示它的注意力模式。
        </p>

        <h2>同一个词，同一个起点向量</h2>
        <p>
          要看清这种“回看”<em>为什么</em>必不可少，最干净的例子是多义词。第 3 章的 embedding 查表是不看上下文的：它只是一张按
          token id 索引的表，所以无论句子讲的是金钱、飞机转弯还是河边野餐，单词 <code>bank</code> 取到的都是{' '}
          <code>embed_tokens.weight</code> 的<em>同一</em>行。在最后一个 token 进入层堆栈的那一刻，模型完全无法区分这些含义——三个起点向量逐字节相同。
        </p>
        <p>
          把这些含义拉开，正是上面各层的全部工作；而你可以在流水线的另一端看到结果：给这个模型三条都以同一个 token
          结尾的 prompt，它对下一个 token 的押注会完全分岔。有一点要说在前面——在这个模型里，消歧义的工作由 6 个全注意力层<em>和</em>
          18 个 GatedDeltaNet 线性注意力层（第 12 章）共同分担，所以“是注意力做到的”只是“是混合上下文的层堆栈做到的”的简写。
        </p>
        <PolysemyDivergence />

        <h2>Q、K、V——同一个向量的三个投影</h2>
        <p>
          每个 token 以一个向量 <code>x</code> 的形式进来（它的嵌入，经过前面各层加工后的结果）。模型用三种不同的方式对{' '}
          <code>x</code> 做线性投影（这里的“投影”只是指用一个学习到的矩阵乘以该向量，得到一个新向量）：
        </p>
        <ul>
          <li>
            <code>Q = x · Wq</code>——<strong>query（查询）</strong>：这个 token 在找什么？
          </li>
          <li>
            <code>K = x · Wk</code>——<strong>key（键）</strong>：如果你看向这个 token，它能提供什么？
          </li>
          <li>
            <code>V = x · Wv</code>——<strong>value（值）</strong>：当这个 token 被注意到时，它应该贡献什么？
          </li>
        </ul>
        <p>
          三者都是长度为 <code>d_head</code> 的向量（即每个头的维度——原始论文 “Attention Is All You Need” 称之为{' '}
          <code>d_k</code>，模型配置里叫 <code>head_dim</code>）。关键在于，这并不是三个不同的 token——而是同一个 token
          的三种不同视角，分别独立学习，让注意力能做出比“直接比较嵌入”更有意思的事。在这个模型里：<code>x</code> 是
          1,024 个数 → 8 个长度 256 的 query 向量，外加 2 组长度 256 的 key/value（8 对 2 的安排是
          GQA，下一章讲）。
        </p>
        <p>
          初学者还必须搞清楚一点：同样的三个矩阵 <code>Wq</code>、<code>Wk</code>、<code>Wv</code> 应用于<em>每个</em>
          位置上的<em>每个</em> token——它们只学习一次、整条序列共享，所以投影本身不携带 token 位于<em>哪里</em>
          的任何信息；位置信息是单独注入的（由 RoPE 完成，第 6 章）。
        </p>
        <p className="text-muted-foreground">
          趁热说一句命名：因为 <code>Q</code>、<code>K</code>、<code>V</code> 都来自同一条运行中的序列，整章讲的都是
          <strong>自</strong>注意力。它的同胞机制是 <em>cross</em>-attention——query 来自一条序列，key/value
          来自另一条；最初的用途是机器翻译，由 decoder 的 query 去读 encoder 的 key。不过 cross-attention
          在这整个模型家族里都不会出现，哪怕是在完整的多模态 checkpoint 里也不会：视觉信息是通过<em>早期融合（early
          fusion）</em>接入的——投影后的图像 patch token 会被直接拼接进和文本 token 相同的那条自注意力序列里，而它在
          <ChapterLink chapterId="architecture">本演示里是关闭的</ChapterLink>——这里 6 个全注意力层没有一个在做
          cross-attention。
        </p>

        <h2>公式</h2>
        <p>对单个头来说，核心运算就是一行数学：</p>
        <MathDisplay
          latex={String.raw`\text{attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_\text{head}}}\right) V`}
        />
        <p>
          我们从左往右读。<code>Q · Kᵀ</code>（ᵀ 表示行列互换——含义是“拿每个 query 与每个 key 做点积”）是一个{' '}
          <code>[seq_len, seq_len]</code> 矩阵：格子 <code>(i, j)</code> 是 token <em>i</em> 的 query 与 token{' '}
          <em>j</em> 的 key 的点积——一个衡量“<em>i</em> 的提问与 <em>j</em> 的供给有多匹配”的原始分数。
        </p>
        <p>
          除以 <code>√d_head</code> 是出于数值上的原因：随着 <code>d_head</code> 增大，点积平均也会变大；不做缩放，softmax
          就会饱和（一个格子是 1.0，其余接近
          0.0），梯度也会消失（梯度是调节权重的训练信号——训练一章会讲；现在觉得模糊没关系）。除以随机点积的标准差（≈{' '}
          <code>√d_head</code>）能让分数保持在 softmax 仍有信息量的区间。
        </p>
        <p>
          下面的小部件把这句话变得可感。拖动头维度，观察同一组 key 的对齐情况：不缩放时，分布随着 <code>d_head</code>{' '}
          增大坍缩到单个 key 上；加上 <code>÷√d_head</code> 缩放后，它在任何宽度下都保持完全一致。
        </p>

        <ScaledSoftmax />
        <p>
          <code>softmax</code> 把分数矩阵的每一行变成一个概率分布：每行之和为 1。这就是每个 token 的
          <em>注意力模式（attention pattern）</em>。再用这个 <code>[seq_len, seq_len]</code> 分布乘以 <code>V</code>
          ，就是按该模式加权地把各个 value 混合起来——这一混合结果，就是注意力层输出的 token <em>i</em> 的新表示。
        </p>
        <p>
          有一点很容易想错：这个混合后的 value 向量并不是对 token 的整体替换。它是一个小小的修正，会被<em>加回</em>到
          token 带进来的那个向量上——注意力是往运行中的表示上写一个增量（delta），而不是把它覆盖掉。这个运行中的表示有个名字，叫做
          <em>残差流（residual stream）</em>；“只加不替换”这条规则，是这个模型里每一个 block 都围绕着接线的那道接缝，
          <ChapterLink chapterId="full-block">完整 Transformer block</ChapterLink> 一章会展示这套接线。
        </p>

        <p>
          这个 <code>[seq_len, seq_len]</code>{' '}
          矩阵也正是注意力变贵的地方。三篇可选的深入阅读再往下挖一层——讲它的内存开销、怎么给这份开销一个具体的数字，以及绕开开销的技巧。第一遍阅读都可以跳过。
        </p>

        <SubChapterNav chapterId="attention" items={findChapter('attention')?.sections} />

        <h2>因果掩码——以及它为何是 decoder LLM 的承重墙</h2>
        <p>
          到目前为止定义的自注意力是对称的：每个 token 都能看到其他所有 token。对编码器（想想 BERT——完形填空类任务）来说没问题。但对生成式
          LLM 是一场灾难。训练的全部意义在于根据前文预测下一个 token。如果模型在训练时被允许看<em>后面</em>
          的内容，它直接照抄就能绕过整个问题。
        </p>
        <p>
          所以在 softmax 之前，我们把 <code>QKᵀ</code> 的上三角设为 <code>-∞</code>（等价地，加上一个对角线及以下为{' '}
          <code>0</code>、对角线以上为 <code>-∞</code> 的掩码矩阵，再逐行 softmax）。softmax 之后这些格子恰好变成{' '}
          <code>0</code>——token <em>i</em> 只能注意 key <em>0 … i</em>。右侧热力图呈<strong>下三角</strong>
          正是这个原因。最下面一行——提示词的最后一个 token——是唯一能看到整条提示词的一行，所以下一个 token 的预测由
          <em>那一行</em>产生。
        </p>
        <p>
          下面的开关小部件把这一点变得具体：把掩码关掉，看模型对第一个 token 的注意力伸向序列的未来。
        </p>

        <CausalMaskToggle />

        <p>
          一个微妙但重要的推论：正是这个掩码让<strong>并行训练</strong>
          成为可能。掩码就位后，你可以把整条序列一次性塞进模型，在一次前向传播里同时计算每个位置的损失——因为谁也看不到前方，每个位置的预测彼此独立。没有掩码，你只能一个一个
          token 地喂。这种三角形的稀疏模式，是让 decoder-only 训练在大规模下可行的那个唯一的架构决策。
        </p>

        <h2>热力图里每个格子的含义</h2>
        <p>
          右侧热力图正是某一层、某一个头的 <code>softmax(QKᵀ/√d)</code> 矩阵：
        </p>
        <ul>
          <li>
            行是 <strong>query</strong>（发起注意的 token）
          </li>
          <li>
            列是 <strong>key</strong>（被注意的 token）
          </li>
          <li>
            <code>(i, j)</code> 处的亮格子表示“在计算 token <em>i</em> 的新表示时，模型大量提取了 token <em>j</em> 的
            value”
          </li>
          <li>每行之和为 1；由于因果掩码，上三角为 0</li>
        </ul>
        <p>
          看看热力图下方扇出图画出的那些长边：一次全注意力就能把很靠后的 token
          的内容直接送到当前位置，无论中间隔着多少 token。在这里距离是免费的——只有相关性才有代价，因为分数是点积，并不是两个
          token 间距的函数。（这种“免费触达”说的是 6 个全注意力层；18 个 GatedDeltaNet 层则改为通过一份滚动状态往回触达，
          <ChapterLink chapterId="kv-cache">KV 缓存一章</ChapterLink>会讲为什么模型两者都需要。）
        </p>

        <h2>为什么要多个头、多个层</h2>
        <p>
          不同的头最终会各有分工——一个可能跟踪“上一个
          token”，一个跟踪“句子的开头”，另一个跟踪“句法上相关的名词”。堆叠注意力层让后面的层把这些模式组合起来：第 5
          章看头，第 9 章看完整的堆叠。
        </p>
        <p>
          关于<em>这个</em>模型的一个细节：它是混合架构。只有每第 4 层——24 层中的 6 层——使用你一直在读的完整 softmax
          自注意力。其余 18 层使用更便宜的线性注意力变体（GatedDeltaNet），它维护固定大小的滚动状态，而不是对所有过去的
          token 做注意力（KV 缓存一章会讲）。右侧的实时热力图反映了这一点：把层滑块拨到 6
          个全量注意力层之一就能看到真实分数；落在线性层上时，面板只会提示该层不输出 softmax 注意力矩阵。
        </p>
        <p className="text-muted-foreground">
          关于头滑块的说明：这个模型有 <strong>8 个 query 头</strong>，但这些头共享的 <strong>key/value 组只有 2 个</strong>
          ——这就是分组查询注意力（GQA），下一章展开。现在只需把每个头当作各自独立的注意力模式来读。
        </p>

        <h2>手算一个微型例子</h2>
        <p>
          矩阵代数配上具体数字更容易找到感觉。下面的小部件带着两个 token——“The” 和
          “cat”——走完注意力公式的每一步。所有数字都由固定的 Q、K、V
          算出；逐步点击，看分数矩阵、缩放、掩码、softmax 和最终输出依次出现。
        </p>

        <ToyAttentionExample />

        <h2>因果掩码，逐格看</h2>
        <p>
          同一个掩码的第二种视角：一个 5×5 网格，点击任意格子，看“<em>i</em> 能看到 <em>0..i</em>
          ”对这条提示词具体意味着什么。
        </p>

        <CausalMaskVisual />

        <p className="mt-6 text-muted-foreground">
          在右侧点击 <em>Run</em>。模型预测的下一个 token
          会带着一道彩虹流光出现在提示词末尾，热力图展示的是产生该预测的那次前向传播中真实的 softmax
          后注意力分数。编辑提示词，或者换一个层/头，观察模式（和预测）如何变化。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

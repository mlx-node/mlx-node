import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { AttentionPipeline } from '../../widgets/AttentionPipeline';
import { GqaGroupSweep } from '../../widgets/GqaGroupSweep';
import { HeadSplitDiagram } from '../../widgets/HeadSplitDiagram';
import { MhaVsGqaCacheBar } from '../../widgets/MhaVsGqaCacheBar';
import { MultiheadConcat } from '../../widgets/MultiheadConcat';
import { PatternDetectorPanel } from '../../widgets/PatternDetectorPanel';

const QWEN_NUM_HEADS = 8;
const QWEN_NUM_KV_HEADS = 2;
const QWEN_HEAD_DIM = 256;
// Qwen3.5-0.8B 的 hidden_size。仅在正文中用来说明 head_dim 与 hidden / num_heads
// 是解耦的（1024 / 8 = 128 ≠ head_dim = 256）。
const QWEN_HIDDEN_DIM = 1024;

/**
 * 第 4 章（multi-head-gqa）的脚手架元数据中文译文——驱动 `<ChapterFrame>`
 * 渲染的页眉、术语表、要点、练习与随堂测验。
 * `chapterId` 必须与 `learn/chapters.ts` 中 `CHAPTERS[3].id` 一致。
 */
export const learning: ChapterLearningData = {
  chapterId: 'multi-head-gqa',
  objective: '解释注意力为什么要拆成多个头并行运行，以及分组查询注意力（GQA）如何缩小 KV 缓存。',
  problem: '单个注意力头一次只能建模一种模式，而给每个 query 头都配一份独立的 K/V，会让长上下文下的缓存内存爆炸。',
  minutes: 10,
  glossary: [
    {
      term: 'head',
      definition: '注意力的一个并行副本，拥有自己的 Q/K/V 切片。每个头可以专注于一种不同的关系。',
    },
    {
      term: 'num_heads (H)',
      definition: '每层并行运行多少个 query 头。Qwen3.5-0.8B 用 8 个。',
    },
    {
      term: 'd_model (hidden)',
      definition:
        '在层与层之间流动的残差流的宽度。Qwen3.5-0.8B 用 1024——W_O 把拼接后的各头输出（8 × 256 = 2048）投影回这个宽度。',
    },
    {
      term: 'num_kv_heads (G)',
      definition: '一共有多少个 K/V 头。在 GQA 下，每个 K/V 头被 H/G 个 query 头共享。Qwen3.5-0.8B 用 2 个。',
    },
    {
      term: 'group_size',
      definition: 'H / G——多少个 query 头共享同一个 K/V 头。对 Qwen3.5-0.8B 来说是 4。',
    },
    {
      term: 'GQA',
      definition: '分组查询注意力：把 H 个 query 头划分成 G 个 K/V 组，使缓存缩小 H/G 倍，而质量损失很小。',
    },
    {
      term: 'MQA',
      definition:
        '多查询注意力：GQA 的极端形式，即 num_kv_heads=1。缓存最小，但各头能寻找的不同模式数量也被削减得最狠。',
    },
    {
      term: 'output projection (W_O)',
      definition:
        '把拼接后的各头映射回残差宽度的可学习矩阵（这里是 2048 × 1024）。它的每个 256 宽切片就是一个头写回残差流的专属“笔”——先拼接再乘 W_O，与把 8 份逐头贡献相加是同一套算术。',
    },
  ],
  takeaways: [
    '同一 K/V 组里的头共享 key 和 value——但它们的 Q 投影各自独立，所以仍然可以关注不同的内容。',
    'GQA 把 KV 缓存缩小 H/G 倍，质量几乎不受影响；对 Qwen3.5-0.8B 来说，相比完整的 MHA 是 4x 的节省。',
    '在长上下文下，占满内存、主导解码延迟的是 KV 缓存，而不是权重。',
  ],
  exercise: {
    prompt:
      '自动运行结束后，通道图默认选中 Q0，并排对比把它和同组的 Q1 放在一起。点一点同组里其余的 Q0–Q3 芯片，对比它们的热力图。这些模式是完全相同、彼此相似，还是差异很大？',
    answer:
      '它们共享同一份 K 和 V，所以模式“押韵”但并不相同——两者仍会点亮相似的 key 位置，但每行的权重不同，因为 Q0 和 Q1 各有自己的 Wq。这正是 GQA 保留下来的东西：同样的 key，不同的问题。',
  },
  quiz: [
    {
      id: 'q1-share-kv',
      prompt: '在 H=8、G=2 的 GQA 下，Q0、Q1、Q2、Q3 之间共享的究竟是什么？',
      options: [
        { id: 'a', label: 'Wq 权重矩阵。' },
        {
          id: 'b',
          label: '它们的 K 和 V 投影——这四个 query 头都和同一个 K/V 头 0 的缓存做点积。',
        },
        { id: 'c', label: '它们最终的输出投影。' },
      ],
      correctId: 'b',
      explanation:
        'GQA 把 query 头划分成组；同组的每个头都读取同一份共享的 K/V。Wq 在每个 query 头上仍然各自独立。',
    },
    {
      id: 'q2-cache-savings',
      prompt: '对 Qwen3.5-0.8B（H=8，G=2），在固定序列长度下，KV 缓存比完整 MHA 小多少？',
      options: [
        { id: 'a', label: '约缩小 2x。' },
        { id: 'b', label: '约缩小 4x。' },
        { id: 'c', label: '约缩小 8x。' },
      ],
      correctId: 'b',
      explanation: '缓存大小与 num_kv_heads 成正比，所以比例是 H/G = 8/2 = 4x。本章的 KV 计数器展示的正是这个数字。',
    },
    {
      id: 'q3-mqa-vs-gqa',
      prompt: '为什么现代开源模型通常更偏爱 GQA 而不是 MQA？',
      options: [
        {
          id: 'a',
          label:
            'MQA 牺牲太多——只剩一个 K/V 头时，各头能寻找的不同模式大幅减少——而 G=2-8 的 GQA 既能拿回 MHA 的大部分质量，又保住了 MQA 的大部分缓存节省。',
        },
        {
          id: 'b',
          label: 'MQA 需要的 KV 缓存比 GQA 更多。',
        },
        {
          id: 'c',
          label: 'GQA 在所有硬件上的训练速度都比 MQA 快。',
        },
      ],
      correctId: 'a',
      explanation:
        'GQA 踩中了平衡点：相对 MHA 的质量损失极小，缓存节省又与 MQA 接近。Llama 3 和 Qwen3.5 都在用它。',
    },
  ],
};

export function MultiheadGqaChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>多头注意力与 GQA</h1>
        <p>
          第 4 章把注意力当作一个单一操作来介绍：<code>softmax(QKᵀ/√d)·V</code>（这里的 <code>√d</code> 是{' '}
          <code>√head_dim</code>，即每个头的宽度——对 Qwen3.5 来说 <code>√{QWEN_HEAD_DIM} = 16</code>
          ——而不是完整的隐藏宽度）。用文字描述就是：先给每个 token 的 query 与所有 token 的 key
          的匹配程度打分，除以 <code>√d</code> 以免向量变长时分数爆炸，softmax 把这些分数变成总和为 1
          的权重，再按这些权重混合各个 value。在真实的 LLM 里，这个操作会被<em>并行运行很多次</em>，每个
          <strong>头（head）</strong>各跑一份，最后把各头的结果拼接起来、投影回去。原因在于容量：单个头一次只能建模一种关系——比如“看上一个
          token”。多个头并行让模型可以同时关注不同的模式：一个头追上一个 token，另一个头找匹配的括号，还有一个头处理远距离的主谓一致。
        </p>

        <p>
          可以把每个头看作比较 token 的一面“透镜”：它学会留意一种特定的关系。一个头可能追踪哪个形容词属于哪个名词；另一个头追踪代词指代前文的哪个词。多面透镜并行运行，模型就能同时盯住许多种关系。
        </p>

        <p>
          在深入头和 KV 缓存之前，先看一张把整个注意力模块画在一起的图——本章和上一章分开讲的每个阶段，端到端连成一条线。先逐步走一遍，再切换开关，看看
          Qwen3.5 的<em>线性</em>层如何用一个循环状态取代整条二次复杂度的流水线。
        </p>

        <AttentionPipeline />

        <h2>标准多头注意力（MHA）</h2>
        <p>
          对于隐藏宽度 <code>d</code> 和 <code>H</code> 个头（<code>H</code> = 并行运行的 query 头数），每个头有自己的维度{' '}
          <code>d_head</code>（<code>d_head</code> = 每个头的向量宽度）——经典设定是 <code>d_head = d / H</code>
          ，让各个头恰好铺满整个隐藏宽度（很多模型，包括 Qwen3.5，把这两者解耦了——下文细说）。模型用三个权重矩阵把
          token 向量投影成形状为 <code>[H, seq, d_head]</code> 的 Q、K、V，再做 reshape（把{' '}
          <code>[H, seq, d_head]</code> 读成一个三维数组：H 个头 × seq 个位置 × 每个位置 d_head
          个数）。每个头在自己的 Q/K/V 切片上独立运行注意力；<code>H</code> 个输出被拼接起来，投影回一个{' '}
          <code>d</code> 宽的向量，继续流过这一层。
        </p>
        <p>
          下面把这个切分画了出来——一个 token 的向量进入，八条头带出来（还有那条更窄的 K/V 通路，正是本章 GQA
          部分要讲的）：
        </p>

        <HeadSplitDiagram />

        <p>
          推理时，每个 token 的 K 和 V 会跨生成步<strong>缓存</strong>下来——这正是从第二个 token
          起速度变快的原因。为什么缓存 K 和 V，却从不缓存 Q？解码时模型只需要<em>新</em> token 的 query——但这个
          query 必须和<em>每个历史 token</em> 的 key 做点积，并混合<em>每个历史 token</em> 的 value。所以历史的 K 和
          V 要保存下来、每一步都重新读取，而历史 token 的 Q 只在它自己那一步用过一次，之后再也用不到。缓存大小是
          <em>每层</em> <code>2 × H × d_head × seq_len</code> 个浮点数。对一个 32k
          上下文、层数深、隐藏宽度大的模型来说，这就是 GB 量级——而且与权重不同，它会随每个新 token 持续增长。
          <strong>在长上下文下占满内存的是 KV 缓存，不是权重。</strong>
        </p>

        <MultiheadConcat />

        <h2>分组查询注意力（GQA）</h2>
        <p>
          GQA 是如今缩小缓存的标准技巧。它不再给每个 query 头单独配一个 K/V 头，而是把 query 头划分成{' '}
          <code>num_kv_heads</code> 组，让同组的每个头共享同一对 K/V：
        </p>
        <MathDisplay
          latex={String.raw`\begin{aligned} \text{num\_heads} \;\;&=\;\; H && \text{(e.g. 8 for Qwen3.5-0.8B)} \\ \text{num\_kv\_heads} \;\;&=\;\; G && \text{(e.g. 2 for Qwen3.5-0.8B)} \\ \text{group\_size} \;\;&=\;\; H / G && \text{(4 query heads per K/V head)} \end{aligned}`}
        />
        <p>
          Q 投影仍然有 <code>H</code> 个头（各自有自己的权重），但 K 和 V 投影只有 <code>G</code> 个头（
          <code>G</code> = key/value 组的数量，每组共享一份 K/V）。计算 query 头 <code>h</code>{' '}
          的注意力时，它和组 <code>floor(h / group_size)</code> 的 K 做点积——并读取同一组的 V。KV 缓存缩小{' '}
          <code>H/G</code> 倍，Q 的维度完全不变，实践中的精度损失极小。
        </p>

        <h2>Qwen3.5-0.8B 的具体配置</h2>
        <p>
          Qwen3.5-0.8B 使用 <code>num_heads = {QWEN_NUM_HEADS}</code>、<code>num_kv_heads = {QWEN_NUM_KV_HEADS}</code>、
          <code>head_dim = {QWEN_HEAD_DIM}</code>。每个 K/V 头被{' '}
          <code>{QWEN_NUM_HEADS / QWEN_NUM_KV_HEADS}</code> 个 query 头共享，所以它的 KV 缓存比等价的完整 MHA 模型
          <strong>小 {QWEN_NUM_HEADS / QWEN_NUM_KV_HEADS}×</strong>。更大的 Qwen3.5 变体保持同样的{' '}
          <code>group_size = 4</code> 比例。
        </p>
        <p>
          注意 Qwen3.5 把 <code>head_dim</code> 和 <code>hidden / H</code> <strong>解耦</strong>了：经典的默认值应该是{' '}
          <code>
            {QWEN_HIDDEN_DIM} / {QWEN_NUM_HEADS} = {QWEN_HIDDEN_DIM / QWEN_NUM_HEADS}
          </code>
          ，但它独立选择了 <code>head_dim = {QWEN_HEAD_DIM}</code>——每个头的宽度是一个自由的超参数。因此 query
          投影做的映射是{' '}
          <code>
            hidden = {QWEN_HIDDEN_DIM} → H · head_dim = {QWEN_NUM_HEADS} · {QWEN_HEAD_DIM} ={' '}
            {QWEN_NUM_HEADS * QWEN_HEAD_DIM}
          </code>
          ，而不是一个方形的 <code>d × d</code> 矩阵。（Qwen 还给这个投影加了几处小变化——见下方
          <strong>进阶</strong>。）
        </p>
        <p>
          Qwen3.5 还交替使用两种层，下方的层选择器只列出可以检查 softmax 分数的经典<em>全量注意力</em>
          层（另一种层在下方<strong>进阶</strong>中介绍）。只有每四层中的一层——<strong>24 层中的 6 层</strong>
          ——运行这种完整的 softmax 注意力并保有持续增长的 KV 缓存；其余 18 层是线性（GatedDeltaNet）层，携带的是固定大小的循环状态。所以下面的缓存数字只统计这
          6 层，而不是全部 24 层。
        </p>

        {/* 可选的深入阅读：Qwen-3.5 特有的复杂细节，对初学者来说会打断
            多头 → GQA 的主线——折叠进 q_proj 的逐头输出门，以及本章不另行
            展开的线性注意力层。默认折叠、可跳过。用原生 <details> 保持键盘
            可聚焦；样式仿照项目的 Glossary 折叠面板（与 14-architecture.tsx 一致）。 */}
        <details className="group not-prose my-4 rounded-md border border-border bg-muted/30 px-4 py-3 text-sm">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-foreground/90 [&::-webkit-details-marker]:hidden">
            <span className="font-medium">
              进阶：Qwen-3.5 特有细节（输出门、QK-norm、线性层）{' '}
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
              在 Qwen3.5 中，完整的 <code>q_proj</code> 权重实际上是两倍宽——
              <code>{2 * QWEN_NUM_HEADS * QWEN_HEAD_DIM}</code> 个输出——因为它在 query 旁边还输出一个逐头的
              <em>输出门</em>，正如上方流水线图所示；上文写的 <code>{QWEN_NUM_HEADS * QWEN_HEAD_DIM}</code> 是 query
              那一半。（Qwen 还在点积之前对 query 和 key 做逐头的 RMSNorm——<code>q_norm</code> /{' '}
              <code>k_norm</code>；架构一章会把两者都讲到。）
            </p>
            <p>
              Qwen3.5 的层是交替排布的：大多数层是<em>线性注意力</em>
              （一种循环变体，超出本章范围），每第四层才是可以检查 softmax 分数的经典<em>全量注意力</em>
              层。下方的层选择器只列出后者。
            </p>
          </div>
        </details>

        <h2>W_O 是怎么来的</h2>
        <p>
          流水线图里有一个矩阵值得回头多看一眼：最末端的输出投影 <code>W_O</code>。先退一步，暂时忽略注意力权重，只追踪
          <em>单个</em>头对残差流做了什么。它的 value 通路是两段线性映射拼起来的：<code>v_proj</code> 把{' '}
          <code>{QWEN_HIDDEN_DIM}</code> 宽的残差向量降到 <code>{QWEN_HEAD_DIM}</code> 维的 value，这个头在{' '}
          <code>W_O</code> 里那条 <code>{QWEN_HEAD_DIM}</code> 宽的切片再把结果升回 <code>{QWEN_HIDDEN_DIM}</code>{' '}
          维。这是一个<strong>低秩</strong>映射：它只能在一个 {QWEN_HEAD_DIM}{' '}
          维的子空间里移动残差向量，但代价比自由映射小得多。如果给每个头配一个满秩映射，那是一个{' '}
          <code>
            {QWEN_HIDDEN_DIM} × {QWEN_HIDDEN_DIM}
          </code>{' '}
          矩阵——<em>每个头</em>约 1.05M 个参数——而分解后的降/升一对只要{' '}
          <code>
            2 × ({QWEN_HIDDEN_DIM} × {QWEN_HEAD_DIM})
          </code>{' '}
          ≈ 0.52M，还没算上 GQA 对“降”那一半的共享（下文细说），成本已经减半。用四个不受约束的头的价钱买八个受约束的头，这就是架构做的交易。
        </p>
        <p>
          “先拼接、再乘 <code>W_O</code>”这套配方看上去也比实际更神秘。把这个{' '}
          <code>
            {QWEN_NUM_HEADS * QWEN_HEAD_DIM} × {QWEN_HIDDEN_DIM}
          </code>{' '}
          的矩阵横向切成 {QWEN_NUM_HEADS} 条{' '}
          <code>
            {QWEN_HEAD_DIM} × {QWEN_HIDDEN_DIM}
          </code>{' '}
          的带子，每头一条。拼接后的 <code>{QWEN_NUM_HEADS * QWEN_HEAD_DIM}</code> 维向量乘 <code>W_O</code>，与让每个头把自己的{' '}
          <code>{QWEN_HEAD_DIM}</code> 维输出穿过自己那条带子、再把 {QWEN_NUM_HEADS} 个结果<strong>相加</strong>，是
          <em>完全相同</em>的算术——拼接加一次大投影 ≡ 逐头贡献求和。所以比“把各头粘在一起”更好的心智模型是：每个头独立提出一份小小的低秩修改，这些修改被
          <em>加</em>到<ChapterLink chapterId="full-block">残差流</ChapterLink>
          上。没有哪个头会覆盖别的头；它们是累加的。
        </p>
        <p>
          对我们这个模型，GQA 给这幅图景添了一道褶皱。“降”的那一半按组共享：<code>v_proj</code> 只有{' '}
          {QWEN_NUM_KV_HEADS} 个 KV 头（
          <code>
            {QWEN_HIDDEN_DIM} → {QWEN_NUM_KV_HEADS * QWEN_HEAD_DIM}
          </code>
          ），所以同组的四个 query 头读到的是<em>同一份</em> {QWEN_HEAD_DIM} 维 value 向量。但“升”的那一半仍然完全逐头：
          <code>o_proj</code> 是{' '}
          <code>
            {QWEN_NUM_HEADS * QWEN_HEAD_DIM} → {QWEN_HIDDEN_DIM}
          </code>
          ，给全部 {QWEN_NUM_HEADS} 个 query 头各留了一条自己的{' '}
          <code>
            {QWEN_HEAD_DIM} → {QWEN_HIDDEN_DIM}
          </code>{' '}
          带子。同组的头用不同的注意力权重混合同一份
          value，再经由不同的投影把结果写进残差流——共享地读，各自地写。
        </p>

        <h2>取舍，以及 MQA 的位置</h2>
        <p>
          GQA 放弃了一点“各头能寻找多少种不同东西”的能力——同组的 query 头无法关注不同的 key，因为它们共享
          K。但它们<em>仍然</em>可以通过各自独立的 Q 投影，在这些共享的 key
          上学出不同的注意力模式（下面的并排对比就能看到）。内存上的收益很大，质量损失很小，更小的缓存带来的解码加速是实打实的。Llama
          3、Qwen3.5 这些中等规模的开源模型全都在用 GQA。
        </p>
        <p>
          把 GQA 推到极端——<code>num_kv_heads = 1</code>——就得到了
          <strong>多查询注意力（Multi-Query Attention，MQA）</strong>：所有 query 头共享一份
          K/V。早期面向推理优化的模型（PaLM、Falcon）用的是 MQA。如今的共识是：组大小适中（通常 4–8）的 GQA 能在保住
          MQA 大部分缓存节省的同时，拿回 MHA 的大部分质量。
        </p>
        <p>
          在下面把整个谱系扫一遍。从 MHA（8 个 KV 头）一路滑到 MQA（1 个），每层的 KV
          缓存线性缩小。每个变体都保留全部八个 query 头，所以 query 侧不受影响——缩小的是各头可以匹配的不同
          K/V 特征组的数量，因为同组的 query 头现在必须共享一份 K/V。
        </p>

        <GqaGroupSweep />

        <h2>不同的头最终学会做什么</h2>
        <p>
          你没法从权重矩阵上直接读出一个头的职责——必须看它在真实输入上产生的模式。下面是三种常见原型，用示意性的热力图画出来。
        </p>

        <PatternDetectorPanel />

        <h2>KV 缓存到底有多大？</h2>
        <p>
          数字能让节省变得具体。下面的柱状图比较了一个假想的 MHA 版 Qwen3.5-0.8B 与真实 GQA 布局在 8k
          上下文（一次聊天几轮就能达到的水平）下的全模型 KV 缓存。
        </p>

        <MhaVsGqaCacheBar />

        <p className="mt-6 text-muted-foreground">
          这里的小部件运行与第 4 章相同的 <code>"The cat sat on the"</code>{' '}
          提示词，并在末尾用彩虹流光“幽灵”出模型预测的下一个 token。上方的通道图展示哪些 query 头共享哪个 K/V
          头；下方的双联热力图对比同组的两个 query 头——相同的 K、不同的 Q，因此在同样的 key
          上呈现出不同的注意力模式。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

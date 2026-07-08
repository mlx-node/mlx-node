import { ChapterLink } from '../../scaffolding/ChapterLink';
import * as React from 'react';

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { ContextWindow } from '../../widgets/ContextWindow';
import { HallucinationDemo } from '../../widgets/HallucinationDemo';
import { ModelSizeLadder } from '../../widgets/ModelSizeLadder';
import { ParamBudgetDiagram } from '../../widgets/ParamBudgetDiagram';

/**
 * Chapter 16 — The whole model, end to end (and what it isn't). (zh)
 *
 * Simplified-Chinese translation of ../14-architecture.tsx. Code, numbers,
 * widget props, chapter ids and module names are byte-identical to the
 * English source; only display prose is translated.
 *
 * Two arcs in one capstone. First, a static, text-only, interactive architecture
 * POSTER of Qwen3.5-0.8B. No worker, no WASM, no inference — everything rendered
 * here is inline SVG + React state. Then a closing reflection on the honest
 * limits that fall out of next-token prediction — hallucination, no lookup
 * database, a finite context window, stochastic output.
 *
 * Every numeric fact below matches the Qwen3.5-0.8B checkpoint the browser
 * loads (qwen3.5-0.8b-mlx-bf16): vocab_size 248,320, SwiGLU intermediate
 * 3,584, 24 layers (18 linear + 6 full), hidden 1024, RoPE base 1e7.
 */

// --------------------------------------------------------------------------
// Verified model facts (single source of truth: the sibling chapters).
// --------------------------------------------------------------------------
const HIDDEN_DIM = 1024; // Qwen3.5-0.8B config: hidden_size
const VOCAB = 248_320; // Qwen3.5-0.8B config: vocab_size (image_token_id 248056 > 151936)
const INTERMEDIATE = 3584; // Qwen3.5-0.8B config: intermediate_size (the 0.6B model is 3072)
const NUM_LAYERS = 24;
const NUM_FULL = 6; // every 4th layer
const NUM_LINEAR = NUM_LAYERS - NUM_FULL; // 18
const MAX_POSITION = 262_144; // 05-rope, 10-kv-cache
const NUM_HEADS = 8; // 04-multihead-gqa
const NUM_KV_HEADS = 2; // 04-multihead-gqa
const HEAD_DIM = 256; // 04-multihead-gqa, 05-rope
const ROPE_DIMS = 64; // 05-rope (head_dim * 0.25)

const VOCAB_LABEL = VOCAB.toLocaleString();

export const learning: ChapterLearningData = {
  chapterId: 'architecture',
  objective:
    '在一张图上追踪一个 token（词元）穿过 Qwen3.5-0.8B 全程的路径，说清这套线性/全量注意力混合堆叠是如何拼装在一起的——然后说出从"预测下一个 token"中直接推导出来的核心局限：幻觉、没有可查询的数据库、有限的上下文、随机的输出。',
  problem:
    '前面每一章都在放大看单个组件——很容易只见树木、不见森林。这一章就是那张把每个零件放回原位的地图，最后再退后一步，正视整台机器的真实边界。',
  minutes: 12,
  glossary: [
    {
      term: 'residual stream',
      definition: '贯穿模型全部深度的未归一化隐藏向量（残差流）。每个块读取它、计算一个小的增量，再把结果加回去。',
    },
    {
      term: 'pre-norm block',
      definition:
        '按 x + sublayer(rmsnorm(x)) 排布的子块：先归一化输入、再做变换，然后加回到未归一化的残差主干上。这种写法让深层堆叠依然可训练。',
    },
    {
      term: 'hybrid attention',
      definition: '在同一个堆叠里混用两种序列混合器（混合注意力）。Qwen3.5 按固定排布交替使用线性注意力层和全量 softmax 注意力层。',
    },
    {
      term: 'Gated DeltaNet (linear attention)',
      definition:
        '一种 O(n) 的循环序列混合器，把历史压缩进一个固定大小的状态，而不是不断增长的 KV 缓存。24 层中有 18 层使用它。',
    },
    {
      term: 'grouped-query attention',
      definition:
        '多个查询头共享同一个 K/V 头的全量 softmax 注意力（分组查询注意力）。Qwen3.5 用 8 个查询头共享 2 个 KV 头，缓存缩小为原来的 1/4。',
    },
    {
      term: 'weight tying',
      definition:
        '把嵌入矩阵复用为 LM head（权重共享）：同一张表在底部把 token id 映射成向量，在顶部又把向量打分成 logits。',
    },
    {
      term: 'hallucination',
      definition: '流畅、自信却不真实的输出（幻觉）——当不存在真实答案时，模型仍然采样了“最像真的”那个 token。',
    },
    {
      term: 'grounding',
      definition: '把模型输出绑定到可验证的来源（例如检索到的文档）上；基础模型默认没有任何这样的依据。',
    },
    {
      term: 'context window',
      definition: `模型一次能关注到的最大 token 数（Qwen3.5-0.8B 为 ${MAX_POSITION.toLocaleString()}）。`,
    },
    {
      term: 'parametric memory',
      definition: '训练时隐式存进模型权重里的知识（参数化记忆），而不是存放在任何可查询的数据库中。',
    },
    {
      term: 'knowledge cutoff',
      definition: '一个时间点（知识截止日期）：在它之后发生的事模型一概不知，因为训练数据到此为止。',
    },
    {
      term: 'stochastic',
      definition: '默认带随机性——因为是采样，同一个提示词在不同的运行中可能产出不同的结果。',
    },
  ],
  takeaways: [
    'Qwen3.5-0.8B 就是 24 个 pre-norm 残差块；逐层唯一变化的只有序列混合器——18 层用线性的 Gated DeltaNet，6 层用全量注意力。',
    '全量注意力是全局的，但它的 KV 缓存随序列长度增长；线性注意力便宜且循环，只维护一个固定大小的状态。3:1 的排布用一小部分内存留住了绝大部分质量。',
    '嵌入矩阵和 LM head 是同一份权重（权重共享）——一张表既负责把 token 映射进来，也负责把向量打分出去。每一层还都带一个稠密的 SwiGLU FFN——全模型没有任何混合专家（MoE）。',
    '幻觉不是故障：模型采样的是最合理的 token，而“合理”只有在该事实在训练数据中充分出现过时才恰好等于“真实”。',
    '模型没有数据库，调用之间也没有记忆——它的知识固化在权重里（带知识截止日期），它的工作记忆只有当前的上下文窗口。',
    '采样让生成默认是随机的；除非贪心解码，否则相同提示词也可能给出不同答案。',
  ],
  exercise: {
    prompt:
      '在本页试两件事。（1）在海报中来回切换 GDN ↔ GQA，对照阅读右侧两个面板——说出全量注意力层（GQA）有、而线性层（Gated DeltaNet）没有的两样东西。（2）在下方的幻觉演示里切换到“没有真实答案”，把最高一根柱子的高度和“广为人知的事实”那一档比较——更平的分布会让模型放弃作答吗？',
    answer:
      '（1）全量注意力（GQA）有一个随序列长度增长的 KV 缓存，并使用部分 RoPE + 每头 Q/K 归一化，再对所有位置做 softmax；线性层（Gated DeltaNet）则只保留一个固定大小的循环状态加一个小小的卷积缓存，从不计算全位置两两的 softmax。（2）在事实题上最高柱约 0.93（非常确定）；在编造的论文题上约 0.16，概率摊到许多 token 上——模型变得不那么确定，却仍然押注在单个最合理的 token 上，给出一个自信的、捏造的答案。除非训练教过它说“我不知道”，低置信度不会自动变成“我不知道”。',
  },
  quiz: [
    {
      id: 'how-many-full',
      prompt: '24 层中有多少层使用全量注意力？',
      options: [
        { id: 'a', label: '全部 24 层' },
        { id: 'b', label: '6 层——每 4 层一个' },
        { id: 'c', label: '一层也没有' },
      ],
      correctId: 'b',
      explanation: '排布是 3 层线性 + 1 层全量，重复 6 次——所以 24 层里有 6 层是全量注意力。',
    },
    {
      id: 'weight-tying',
      prompt: '这里的权重共享（weight tying）指什么？',
      options: [
        { id: 'a', label: '嵌入矩阵和 LM head 是同一份权重' },
        { id: 'b', label: '所有层共享同一份权重' },
        { id: 'c', label: 'Q 投影和 K 投影共享权重' },
      ],
      correctId: 'a',
      explanation: '同一张表用了两次：输入端做 token id → 向量，输出端（其转置）做向量 → logits。',
    },
    {
      id: 'other-18',
      prompt: '其余 18 层用的序列混合器是什么？',
      options: [
        { id: 'a', label: '全量 softmax 注意力' },
        { id: 'b', label: 'Gated DeltaNet 线性注意力' },
        { id: 'c', label: '只是一个卷积' },
      ],
      correctId: 'b',
      explanation: '18 个非全量层是 Gated DeltaNet——带固定大小状态的循环线性注意力，不是普通卷积。',
    },
    {
      id: 'q1-why-hallucinate',
      prompt: 'LLM 为什么会产生幻觉？',
      options: [
        { id: 'a', label: '它在故意对用户撒谎。' },
        {
          id: 'b',
          label: '它采样最合理的下一个 token，而前向计算中没有任何环节把这个 token 与事实来源核对。',
        },
        { id: 'c', label: '它的训练数据全是假的。' },
      ],
      correctId: 'b',
      explanation:
        '模型输出的是“看起来最像”的 token；当不存在真实答案时，它仍会给出一个流畅、自信的 token。合理不等于真实。',
    },
    {
      id: 'q2-knowledge',
      prompt: '推理时，LLM 的事实性知识存放在哪里？',
      options: [
        { id: 'a', label: '在一个它每生成一个 token 就查询一次的数据库里。' },
        { id: 'b', label: '在训练时固化下来的权重里（参数化记忆）——并带有知识截止日期。' },
        { id: 'c', label: '在用户以前的对话里。' },
      ],
      correctId: 'b',
      explanation: '基础模型什么也不查；它的知识烤进了权重里，训练结束后就被冻结——这正是它有知识截止日期的原因。',
    },
    {
      id: 'q3-stochastic',
      prompt: '为什么同一个提示词在两次运行中可能给出两个不同的答案？',
      options: [
        { id: 'a', label: '模型在两次运行之间改写了自己的权重。' },
        {
          id: 'b',
          label: '生成是从概率分布中采样的，所以默认是随机的——除非用贪心解码（temperature 0）。',
        },
        { id: 'c', label: '上下文窗口的大小变了。' },
      ],
      correctId: 'b',
      explanation: '采样是有意引入的随机性；贪心解码（永远取最高分 token）是确定性的特例。',
    },
  ],
};

// ==========================================================================
// Chapter body — prose + the inline interactive poster.
// ==========================================================================

export function ArchitectureChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <div className="space-y-8">
        <Prose className="lg:max-w-none">
          <h1>一页看完整个模型</h1>
          <div className="grid grid-cols-1 gap-x-14 lg:grid-cols-2">
            <p>
              <strong>一句话总结：</strong>Qwen3.5-0.8B 就是 {NUM_LAYERS} 个几乎一模一样的 pre-norm
              残差块叠起来，底部是嵌入表，顶部是与之共享权重的 LM head——前面各章读到的内容几乎都住在其中某个块里。
              下面的海报就是那张地图：它把整个文本解码器铺成一条中央主干，左侧展开稠密 SwiGLU
              前馈网络，右侧展开两种序列混合器——线性的 Gated DeltaNet 和全量的门控 GQA（分组查询注意力）。
              悬停、点击或用 Tab 聚焦任意方块，下方面板会告诉你它是做什么的。
            </p>
            <p>
              关于“整个模型”要诚实说明一点：这条主干画的是<strong>文本解码器</strong>——也就是浏览器内演示真正运行的路径。
              完整的 Qwen3.5-0.8B checkpoint 是多模态的，所以海报也勾勒了可选的视觉编码器（左下角灰色部分）；
              在这个纯文本演示里它从未被用到。
            </p>
          </div>
        </Prose>

        <ArchitecturePoster />

        <div className="mx-auto w-full max-w-3xl">
          <Prose className="lg:max-w-none">
            <p>
              颜色容易盖住一件事：海报里的每一个方块，都是一坨在训练结束时就<em>冻结</em>的权重。真正在它们之间逐层
              <em>向上流动</em>的，只有你的文本——而到了这一步，它已经不再是词语或语义，只是一个{' '}
              <code>[你的 token 数 × {HIDDEN_DIM}]</code> 的、由普通实数组成的数组。整条前向通路，就是这一个张量在通往
              顶部的路上被不断地重排、相加、再重排。那些方块是程序，而张量是唯一在动的东西。
            </p>
          </Prose>
        </div>

        <Prose className="lg:max-w-none">
          <div className="grid grid-cols-1 gap-x-14 lg:grid-cols-2">
            <div>
              <h2>同一个块，重复 24 次</h2>
              <p>
                一个 token 以整数 id 进入，在嵌入矩阵里查表，变成一个 {HIDDEN_DIM} 维向量。这个向量就是
                <strong>残差流（residual stream）</strong>的起点——那条贯穿模型全部深度、不做归一化的主干道。
                {NUM_LAYERS} 层中的每一层都对它做同样的两件事：一个序列混合子块（某种注意力）和一个逐 token
                的前馈子块，二者都包在如今已成标配的 pre-norm + 残差模式里。
              </p>
              <p>
                关键在于，前馈这一半从不改变：每一层都带一个稠密的 <code>SwiGLU</code> MLP（{HIDDEN_DIM} →{' '}
                {INTERMEDIATE} → {HIDDEN_DIM}），这个 checkpoint 里没有任何混合专家（MoE）路由。归一化用的是
                RMSNorm，且只作用于子块的输入，所以残差主干本身保持未归一化，梯度可以一路传到最底层。最后一层之后，
                一个最终的 RMSNorm 加上 LM head 把隐藏状态变成词表中每个 token 一个的 logits（未归一化分数）。
              </p>
              <p>
                有人会问：到底为什么要把同一个块重复 {NUM_LAYERS}{' '}
                次？常见的解读是：深度换来的是“组合”——靠前的层可以先把一件小而局部的事敲定，比如确定某个代词指回前面
                几个 token 处的某个名字；靠后的层再借助这个已经解决的事实去判断更大的东西，比如这句话真正在说哪个分句。
                每个块只往残差流里加一个很小的微调，所以叠起来之后，这些微调就能累积成单个块绝无可能独自完成的那种多步
                塑形。这个直觉是有用的——但要松松地拿着：这幅图景大多来自对稠密的、纯注意力模型的研究，而它能多干净地
                迁移到这套混合堆叠上（这里 {NUM_LAYERS} 层中有 {NUM_LINEAR}{' '}
                层是线性循环、而非全量注意力），本身就并没有被很好地确立。
              </p>
            </div>
            <div>
              <h2>两种序列混合器</h2>
              <p>
                逐层<em>确实</em>在变的只有序列混合器。Qwen3.5 是一个<strong>混合体</strong>：它按固定的{' '}
                <code>[linear, linear, linear, full]</code> 排布重复六次，于是 {NUM_LAYERS} 层中有 {NUM_LINEAR}{' '}
                层使用线性注意力（Gated DeltaNet），其余 {NUM_FULL} 层使用全量分组查询注意力。全量注意力层是全局的——
                每个位置都能回看其他所有位置——但它的 KV 缓存随序列增长。线性层则是循环且便宜的：它们把历史折叠进一个
                固定大小的状态，所以在第 10 个 token 和第 100,000 个 token 时占用的内存一样多。
              </p>
              <p>
                3:1 的比例就是这笔交易。全量注意力是唯一能可靠地回头找回某个特定早期 token 的混合器，所以模型保留了
                少量几层；线性层则以一小部分缓存的代价承担了大部分语言流。结果是：质量接近纯注意力模型，而内存占用
                随上下文增长几乎不动。
              </p>
            </div>
          </div>
        </Prose>

        {/* Where the ~0.8B parameters live — a to-scale budget computed from
            the checkpoint's real tensor shapes. Centered at article width like
            the other reading sections. */}
        <div className="mx-auto w-full max-w-3xl">
          <Prose className="lg:max-w-none">
            <p>
              还有另一种看待这台机器的方式：约 0.8B（8 亿）个参数究竟<em>住在哪里</em>？逐块清点一遍，你会发现 FFN
              堆叠是每一层里最大的单一块——比两种序列混合器加起来还大：
            </p>
          </Prose>
          <ParamBudgetDiagram />
        </div>

        {/* Optional deep-dive: what the GDN sub-boxes in the poster actually
            mean, in words. Collapsed by default and skippable. */}
        <div className="mx-auto w-full max-w-3xl">
          <details className="group not-prose rounded-md border border-border bg-muted/30 px-4 py-3 text-sm">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-foreground/90 [&::-webkit-details-marker]:hidden">
              <span className="font-medium">
                进阶：Gated DeltaNet 层的内部 <span className="text-muted-foreground">· 选读，写给好奇的你</span>
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
                这里展开解释上方海报里那些青色 Gated DeltaNet 方块的含义。完全可以跳过——就使用模型而言，上面的 3:1
                交易已经是故事的全部。但在 {NUM_LINEAR} 个线性层中的任意一层内部，真正发生的是这些：
              </p>
              <ul className="list-disc space-y-1.5 pl-5">
                <li>
                  <strong>投影。</strong>线性映射把输入拆成 query/key、value 加一个 <em>z 门控</em>，以及一对原始的
                  beta/衰减信号（b, a）——sigmoid 把 b 变成 β，衰减公式把 a 变成 g，它们是下面那段循环要拧的旋钮。
                </li>
                <li>
                  <strong>一个短卷积。</strong>一个因果的<em>逐通道</em>卷积（kernel 4，所以每个通道只看最近 4 个
                  token），后接 SiLU 激活，在循环之前先加一点局部混合。
                </li>
                <li>
                  <strong>门控 delta 规则循环</strong>是它的心脏。它只携带一个固定大小的状态，逐 token 扫过序列：β
                  控制每个新 token 对状态的<em>覆写</em>程度，g（衰减）控制旧信息被<em>遗忘</em>的速度。这让它对序列长度是
                  O(n) 的——线性（开销随 token 数同步增长），而不是二次方（全量注意力的开销随 token 数的<em>平方</em>
                  增长——token 翻倍，工作量约变四倍）——而且状态大小永不增长。
                </li>
                <li>
                  <strong>内存。</strong>这一层不维护随每个 token 增长的 KV 缓存，只携带一个很小的卷积缓存（最近几个
                  token）加上那个固定大小的循环状态——所以它的内存<em>与上下文长度无关，是常量</em>。
                </li>
                <li>
                  <strong>收尾。</strong>一个 z 门控的 RMSNorm 加一个输出投影，把结果送回 {HIDDEN_DIM} 维的残差流。
                </li>
              </ul>
              <p>
                正是这种常量内存的循环，让 {NUM_LAYERS} 层中的 {NUM_LINEAR} 层可以不带增长的缓存运行，只剩{' '}
                {NUM_FULL} 个全量注意力层需要为缓存买单。
              </p>
            </div>
          </details>
        </div>

        {/* Model-size ladder — situates this 0.85B model among the rough scales
            of language models, and sets up the limitations arc: even a well-built
            model this size is ~100× smaller than frontier systems. Centered at
            article width like the other reading sections. */}
        <div className="mx-auto w-full max-w-3xl">
          <Prose className="lg:max-w-none">
            <p>
              在点出这些局限之前，还有一件事要放在尺度上看清：这是一个小模型。它的 {(852_985_920).toLocaleString()}{' '}
              个参数把它放在规模阶梯的底部附近，远低于大多数人说“大语言模型”时心里想的那些系统。这并不让它成为一个玩具
              ——但它确实意味着，下面有些局限部分是关于<em>规模</em>的，在一个大得多的模型上会减轻（却永不消失）。
            </p>
          </Prose>
          <ModelSizeLadder />
        </div>

        {/* The merged limitations arc is a reading section, not part of the
            full-width poster — center it at article width inside the wide
            (~1400px) body so it doesn't strand in the left third. */}
        <div className="mx-auto w-full max-w-3xl">
          <Prose className="lg:max-w-none">
            <h2>模型不是什么</h2>
            <p>
              到这里你已经端到端看完了整台机器——每一个块，以及它所在的位置。同样值得精确说清的，是它<em>没有</em>
              在做什么，因为几乎每一次"这 LLM 坏了"的惊讶，都来自期待一个这套架构从未拥有过的能力。下面这些局限没有
              一个是后来打上的补丁式缺陷——它们全部直接源自"预测下一个 token"本身。
            </p>

            <h2>它采样的是“合理”，不是“真实”</h2>
            <p>
              前向计算为每个候选的下一个 token 给出一个概率，然后我们从中采样。这条流水线里没有任何一步把事实与
              某个来源核对。当真实的续写在训练数据中很常见时，“合理”与“真实”恰好重合，模型看上去就像“知道”。
              而当不存在真实答案——或者模型压根没见过——它<em>仍然</em>会输出一个流畅、自信的 token。这就是
              <strong>幻觉</strong>：
            </p>
            <HallucinationDemo />
            <p>
              注意，它不是在撒谎（没有意图），也不是出了故障——它在精确地执行自己被造出来要做的事：产出一个看起来
              很像的下一个 token。分布越平，它越没把握，但除非训练教过它含糊其辞，它照样会押注一个答案。
            </p>

            <h2>没有可查询的数据库</h2>
            <p>
              基础模型的“知识”是<strong>参数化的</strong>：在训练时烤进权重，而不是存放在推理时可查询的表里。
              所以它无法可靠地引用一个自己没有实际记住的来源，并且带有<strong>知识截止日期</strong>——训练之后发生的事
              根本不在权重里。（检索和工具可以在上面外挂一个真正的数据库，但那是外加的；你研究的这个模型本身一个也没有。）
            </p>

            <h2>它的记忆就是上下文窗口——而且是有限的</h2>
            <ContextWindow />
            <p>
              模型唯一的工作记忆，是当前位于它的
              <ChapterLink chapterId="kv-cache">上下文窗口</ChapterLink>
              之内的那些 token。超过这个硬上限，最旧的 token 必须被丢弃腾位置，丢了就再也回不来。而在两次独立对话之间，
              它什么也不记得——每次调用都从一片空白开始，除了你在提示词里重新提供的文本。
            </p>

            <h2>相同输入，不同输出</h2>
            <p>
              因为我们在<em>采样</em>，生成默认是<strong>随机的</strong>：temperature 大于零时，同一个提示词在不同
              运行中可能产出不同答案。这对创意写作是特性，对任何需要复现的场景则是坑——想要确定性时就用贪心解码
              （temperature 0），正如<ChapterLink chapterId="sampling">采样一章</ChapterLink>所展示的那样。
            </p>

            <h2>这些都不是 bug</h2>
            <p>
              它们不是等待修复的缺陷——而是一个一次只预测一个合理 token 的系统的诚实形状。了解它们，正是"用好 LLM"
              与“被它打个措手不及”之间的分界线。这也是追完整条前向通路的真正回报：模型不再是魔法，而成为一个
              你能切实推理其长处与边界的工具。
            </p>
          </Prose>
        </div>
      </div>
    </ChapterFrame>
  );
}

// ==========================================================================
// Detail data — one entry per selectable block. The `sequence` entry is
// special: it swaps with the mixer toggle (see sequenceDetailFor).
// ==========================================================================

type RelatedLink = { chapterId: string; label: string };
type Detail = { title: string; facts?: string; body: string; related?: RelatedLink[] };

const DETAILS: Record<string, Detail> = {
  'start-here': {
    title: '从这里开始——一张图里的整个模型',
    facts: '文本从底部进入 · 下一个 token 的分数从顶部输出',
    body: `这就是 Qwen3.5-0.8B 的全貌。你的文本以 token id 的形式从底部进入，先变成向量（嵌入），然后向上流过 ${NUM_LAYERS} 个重复的层——每一层先跨序列混合信息（注意力），再对每个 token 做变换（SwiGLU 前馈）——最后一次归一化加 LM head 把顶部的向量变成词表中每个 token 一个的分数。点击、悬停或用 Tab 聚焦任意方块即可阅读它的说明。大多数层使用便宜的“线性”混合器，少数层使用全量注意力——用上方的切换按钮对比两者。`,
  },
  // --- central spine ---
  output: {
    title: '输出 logits',
    facts: `${VOCAB_LABEL} 个分数 · 词表中每个 token 一个`,
    body: '堆叠的最顶端：softmax 之前，词表中每个 token 对应一个实数。对它们取 argmax（贪心）或采样，就选出了下一个 token，随后它被追加到序列里再喂回模型。',
    related: [{ chapterId: 'sampling', label: '采样' }],
  },
  'lm-head': {
    title: '权重共享的 LM head',
    facts: `→ ${VOCAB_LABEL} 个 logits · 权重与嵌入共享`,
    body: `把最终的 ${HIDDEN_DIM} 维隐藏状态投影成词表中每个 token 一个的 logit。投影权重就是嵌入矩阵本身（权重共享）：底部把 id 映射成向量的那张表，到了顶部又把向量打分成 logits。`,
    related: [{ chapterId: 'lm-head', label: 'LM head 与权重共享' }],
  },
  'final-norm': {
    title: '最终 RMSNorm',
    body: '24 层全部走完之后的最后一次归一化，作用在输出投影之前，让 LM head 读到的是尺度良好的隐藏状态。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  decoder: {
    title: `解码器堆叠 ×${NUM_LAYERS}`,
    facts: `${NUM_LAYERS} 层 · ${NUM_LINEAR} 线性 + ${NUM_FULL} 全量 · hidden ${HIDDEN_DIM}`,
    body: `${NUM_LAYERS} 个 pre-norm 残差块。序列混合的位置按固定的 [GDN, GDN, GDN, full-attention] 排布重复 ${NUM_FULL} 次——所以全量注意力落在第 4、8、12、16、20、24 层，其余各层都是线性的。前馈那一半在每一层都是一个稠密的 SwiGLU MLP。`,
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'pre-norm': {
    title: 'RMSNorm（注意力前）',
    facts: `逐 token L2 → ≈ √${HIDDEN_DIM} ≈ 32`,
    body: '在进混合器之前先归一化残差流：除以均方根，再乘上一个可学习的逐特征增益。这让残差在深度方向不断累加时，数值幅度仍保持稳定。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'residual-1': {
    title: '把混合器输出加回去',
    body: '混合器的输出被加回残差流。未归一化的残差是一条恒等通路，让梯度能传到全部 24 层——这正是这套堆叠能训练起来的原因。',
    related: [{ chapterId: 'full-block', label: '完整的 Transformer 块' }],
  },
  'post-norm': {
    title: 'RMSNorm（MLP 前）',
    body: '第二次 pre-norm，这次放在前馈子块前面。运算与注意力前的归一化完全相同，作用在（刚被更新过的）残差流上。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  mlp: {
    title: '稠密 SwiGLU FFN',
    facts: `${HIDDEN_DIM} → ${INTERMEDIATE} → ${HIDDEN_DIM} · SwiGLU · 无 MoE`,
    body: `每一层都有的逐 token 前馈网络（稠密——没有混合专家）。gate 和 up 两个投影把维度从 ${HIDDEN_DIM} 拓宽到 ${INTERMEDIATE}，二者按 SiLU(gate) · up 结合，再由 down 投影把 ${INTERMEDIATE} 压回 ${HIDDEN_DIM}，加回残差流。左侧面板展开的正是这个块。`,
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },
  'residual-2': {
    title: '把 MLP 输出加回去',
    body: 'MLP 的输出被加回残差流。这个和就是本层的输出——下一层读到的输入。如此重复 24 次。',
    related: [{ chapterId: 'full-block', label: '完整的 Transformer 块' }],
  },
  embeddings: {
    title: 'token + 视觉嵌入',
    facts: `嵌入表 ${VOCAB_LABEL} × ${HIDDEN_DIM}`,
    body: `每个 token id 在嵌入矩阵里查表，变成一个 ${HIDDEN_DIM} 维向量——残差流的起点。在多模态 checkpoint 中，投影后的视觉 token 会被拼接进同一条序列；浏览器演示只用文本。`,
    related: [{ chapterId: 'embeddings', label: '词嵌入（Embeddings）' }],
  },
  input: {
    title: '输入序列',
    facts: `上下文最长 ${MAX_POSITION.toLocaleString()} 个 token`,
    body: '你的文本变成一串整数 token id；多模态模型还可以交错插入可选的图像和视频 token。每个 id 只是嵌入矩阵的一个行索引——纯粹的查表键，此刻还谈不上任何“理解”。',
    related: [{ chapterId: 'tokenization', label: '分词（Tokenization）' }],
  },

  // --- left: Dense SwiGLU FFN expanded ---
  'ffn-input': {
    title: 'FFN 输入（hidden 1024）',
    body: `归一化后的残差向量进入逐 token 前馈网络。整个堆叠里的隐藏宽度都是同样的 ${HIDDEN_DIM} 维。`,
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },
  'ffn-gate': {
    title: 'gate 投影',
    facts: `Linear ${HIDDEN_DIM} → ${INTERMEDIATE}`,
    body: '两条并行升维投影之一。它的输出经 SiLU 压缩后，作为乘性门控作用在 up 投影上。',
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },
  'ffn-up': {
    title: 'up 投影',
    facts: `Linear ${HIDDEN_DIM} → ${INTERMEDIATE}`,
    body: '第二条并行投影，保持线性。它承载被门控调制的那些值。',
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },
  'ffn-act': {
    title: 'SwiGLU 激活',
    facts: 'SiLU(gate) · up',
    body: '逐元素运算：SiLU(gate) 乘以 up。正是这个门控让 SwiGLU 在相同参数预算下比普通 MLP 更有表达力。',
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },
  'ffn-down': {
    title: 'down 投影',
    facts: `Linear ${INTERMEDIATE} → ${HIDDEN_DIM}`,
    body: `把宽达 ${INTERMEDIATE} 维的中间表示压回 ${HIDDEN_DIM} 维，结果才能加回残差流。`,
    related: [{ chapterId: 'mlp', label: 'MLP 模块' }],
  },

  // --- right top: Gated DeltaNet (linear attention) expanded ---
  'gdn-input': {
    title: 'GDN 隐藏输入',
    body: '归一化后的残差进入线性注意力块——24 层中有 18 层使用它。',
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'gdn-proj': {
    title: '输入投影 + 状态',
    facts: 'q,k · v,z · b,a  +  conv 与循环状态',
    body: '线性投影把输入拆成 query/key、value/z 门控、beta/衰减（b,a）三路，旁边还有从此前 token 携带下来的卷积状态和循环状态。',
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'gdn-conv': {
    title: '逐通道 Conv1D',
    facts: 'kernel 4 · SiLU · q/k/v 通道',
    body: '在 q/k/v 通道上做一个短的因果逐通道卷积（kernel 4），后接 SiLU 激活——让线性注意力层在进入循环之前先有一点局部混合。',
  },
  'gdn-recurrence': {
    title: '门控 delta 循环',
    facts: 'β、g 更新一个固定大小的状态',
    body: 'Gated DeltaNet 的心脏：一段 delta 规则循环，β 控制新信息对状态的覆写程度，g（衰减）控制遗忘速度。对序列是 O(n)，状态大小固定。',
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'gdn-cache': {
    title: '卷积缓存与循环缓存',
    body: 'GDN 不维护随序列增长的 KV 缓存，只携带一个很小的卷积缓存（最近几个 token）和一个固定大小的循环状态——无论上下文多长，内存都是常量。',
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'gdn-out': {
    title: 'RMSNormGated + out-proj',
    body: '一个 z 门控的 RMSNorm 加输出投影，把这个块的结果送回 1024 维残差流。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },

  // --- right bottom: Gated GQA (full attention) expanded ---
  'gqa-input': {
    title: 'GQA 隐藏输入',
    body: '归一化后的残差进入全量注意力块——24 层中的 6 层使用它（第 4、8、12、16、20、24 层）。',
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },
  'gqa-qproj': {
    title: 'Q 投影（+ 门控）',
    facts: `${NUM_HEADS} 个查询头 · head-dim ${HEAD_DIM}`,
    body: `投影出 ${NUM_HEADS} 个查询头，外加一路 sigmoid 输出门控信号。对注意力输出做门控，正是门控 GQA 中“门控”二字的来由。`,
    related: [{ chapterId: 'multi-head-gqa', label: '多头注意力与 GQA' }],
  },
  'gqa-kvproj': {
    title: 'K / V 投影',
    facts: `${NUM_KV_HEADS} 个 KV 头（由 ${NUM_HEADS} 个查询头共享）`,
    body: `只有 ${NUM_KV_HEADS} 个 key/value 头，由 ${NUM_HEADS} 个查询头共享——这就是分组查询注意力，让 KV 缓存缩小 ${NUM_HEADS / NUM_KV_HEADS} 倍。`,
    related: [{ chapterId: 'multi-head-gqa', label: '多头注意力与 GQA' }],
  },
  'gqa-qnorm': {
    title: 'Q 归一化（每头）',
    facts: `${NUM_HEADS} 个头`,
    body: '在 RoPE 之前对 query 做每头 RMSNorm——稳定注意力 logits。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'gqa-knorm': {
    title: 'K 归一化（每头）',
    facts: `${NUM_KV_HEADS} 个头`,
    body: '对 key 做每头 RMSNorm，是 query 归一化在 K 侧的镜像。',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'gqa-rope': {
    title: '部分 RoPE',
    facts: `${HEAD_DIM} 维中的 ${ROPE_DIMS} 维 · θ = 1e7`,
    body: `旋转位置编码（RoPE）只作用于每个头 ${HEAD_DIM} 维中的前 ${ROPE_DIMS} 维；其余维度不携带位置信息。很大的基数 θ = 1e7 把旋转拉伸到极长的上下文上。`,
    related: [{ chapterId: 'rope', label: '位置编码（RoPE）' }],
  },
  'gqa-sdpa': {
    title: 'SDPA + 输出门控',
    facts: `缩放点积注意力 · o_proj → ${HIDDEN_DIM}`,
    body: `对所有位置做缩放点积注意力，乘上 sigmoid 门控，再由 o_proj 回到 ${HIDDEN_DIM} 维残差流。它使用一个随序列长度增长的 KV 缓存。`,
    related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
  },

  // --- vision + text inputs ---
  'vision-input': {
    title: '图像 / 视频输入',
    facts: 'patch 16 · tubelet 2',
    body: '可选。图像和视频在进入视觉编码器之前被切成 16 像素的图块（视频的时间维 tubelet 为 2）。这个纯文本浏览器演示不使用它。',
  },
  'vision-encoder': {
    title: '视觉编码器 ×12',
    facts: 'hidden 768 · 12 个头',
    body: `一个面向图块的 12 层 transformer 编码器（"视觉 transformer"，即 ViT）。隐藏维度 768，自带 12 个注意力头——与解码器的 ${NUM_HEADS} 个查询头相互独立——它把图像/视频图块变成视觉 token。`,
  },
  'vision-project': {
    title: '视觉 → hidden 1024',
    body: `一个投影把编码器的输出映射进 ${HIDDEN_DIM} 维的文本嵌入空间，让视觉 token 与文本共处于同一条残差流。`,
  },
  'text-input': {
    title: '文本 token',
    facts: `词表 ${VOCAB_LABEL}`,
    body: '文本变成整数 token id，并被嵌入到残差流中。这是浏览器演示运行的唯一一条输入通路。',
    related: [{ chapterId: 'tokenization', label: '分词（Tokenization）' }],
  },

  // --- auxiliary MTP draft head (not used by this demo) ---
  'mtp-head': {
    title: 'MTP head——辅助草稿头',
    facts: '额外 1 层 · 多 token 预测 · 投机解码',
    body: `checkpoint 携带的一个小型多 token 预测头（mtp_num_hidden_layers 1），用于额外预测一个未来 token。训练时它充当辅助目标，支持它的运行时还能在推理中用它做投机解码——先草拟一个候选的下一个 token，再由主模型验证。这里把它画成灰色，因为这个浏览器演示（以及普通的非投机生成）不使用它：token 的 logits 只来自权重共享的 LM head。和视觉编码器一样，它属于完整 checkpoint，但不在本演示中 token 实际走过的路径上。`,
  },
};

/** Live detail for the central `sequence` block, depending on the mixer toggle. */
function sequenceDetailFor(mixer: Mixer): Detail {
  if (mixer === 'gdn') {
    return {
      title: '序列块——Gated DeltaNet（线性）',
      facts: '线性注意力 · conv kernel 4 · 固定大小循环状态',
      body: `循环式 O(n) 线性注意力，${NUM_LAYERS} 层中有 ${NUM_LINEAR} 层使用。投影把输入拆成 q/k/v/z 和单独的一对 b/a（beta、衰减）；先过逐通道 Conv1D（kernel 4，SiLU），再由门控 delta 循环把它们折叠进一个固定大小的状态，b 和 a 控制保留还是覆写。最后一个 z 门控 RMSNorm 加输出投影收尾——见右侧展开的面板。`,
      related: [{ chapterId: 'kv-cache', label: 'KV 缓存与混合注意力' }],
    };
  }
  return {
    title: '序列块——门控 GQA（全量）',
    facts: `${NUM_HEADS} Q / ${NUM_KV_HEADS} KV 头 · head-dim ${HEAD_DIM} · RoPE ${ROPE_DIMS}/${HEAD_DIM}`,
    body: `分组查询全量注意力，${NUM_LAYERS} 层中有 ${NUM_FULL} 层使用（第 4、8、12、16、20、24 层）。${NUM_HEADS} 个查询头共享 ${NUM_KV_HEADS} 个 KV 头；每头做 Q/K RMSNorm；部分 RoPE 只旋转每头 ${HEAD_DIM} 维中的 ${ROPE_DIMS} 维（θ = 1e7）；然后是缩放点积注意力、一个 sigmoid 输出门控和 o 投影——见右侧展开的面板。`,
    related: [
      { chapterId: 'multi-head-gqa', label: '多头注意力与 GQA' },
      { chapterId: 'rope', label: '位置编码（RoPE）' },
    ],
  };
}

// ==========================================================================
// Interactive poster.
// ==========================================================================

type Mixer = 'gdn' | 'gqa';

export function ArchitecturePoster() {
  const [selectedId, setSelectedId] = React.useState<string>('start-here');
  const [mixer, setMixer] = React.useState<Mixer>('gdn');

  const detail = selectedId === 'sequence' ? sequenceDetailFor(mixer) : (DETAILS[selectedId] ?? DETAILS.decoder!);

  const selectMixer = React.useCallback((m: Mixer) => {
    setMixer(m);
    setSelectedId('sequence');
  }, []);

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        这是 Qwen3.5-0.8B 架构的一张静态地图——不做任何模型推理。悬停、点击或用 Tab
        聚焦任意方块，即可在下方面板阅读它的说明。切换按钮高亮代表层正在使用的序列混合器。
      </div>

      <MixerToggle mixer={mixer} onChange={selectMixer} />

      <div className="overflow-x-auto rounded-md border border-border bg-background p-3">
        <PosterSvg selectedId={selectedId} onSelect={setSelectedId} mixer={mixer} onMixerSelect={selectMixer} />
      </div>

      <DetailPanel detail={detail} />
    </div>
  );
}

// --------------------------------------------------------------------------
// Mixer toggle — real HTML buttons (keyboard-focusable), not SVG controls.
// --------------------------------------------------------------------------
function MixerToggle({ mixer, onChange }: { mixer: Mixer; onChange: (m: Mixer) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs uppercase tracking-wider text-muted-foreground">代表层的序列混合器</span>
      <div className="inline-flex overflow-hidden rounded-md border border-border" role="group" aria-label="序列混合器">
        <button
          type="button"
          aria-pressed={mixer === 'gdn'}
          onClick={() => onChange('gdn')}
          className={[
            'px-3 py-1 text-xs font-medium transition-colors',
            mixer === 'gdn'
              ? 'bg-teal-400/15 text-teal-200'
              : 'bg-transparent text-muted-foreground hover:text-foreground',
          ].join(' ')}
        >
          线性（GDN）
        </button>
        <button
          type="button"
          aria-pressed={mixer === 'gqa'}
          onClick={() => onChange('gqa')}
          className={[
            'border-l border-border px-3 py-1 text-xs font-medium transition-colors',
            mixer === 'gqa'
              ? 'bg-violet-400/15 text-violet-200'
              : 'bg-transparent text-muted-foreground hover:text-foreground',
          ].join(' ')}
        >
          全量（GQA）
        </button>
      </div>
      <span className="text-[11px] text-muted-foreground">
        {NUM_LINEAR} 层线性、{NUM_FULL} 层全量——切换可高亮右侧对应的面板。
      </span>
    </div>
  );
}

// ==========================================================================
// SVG poster. ViewBox ported 1:1 from the landscape artifact (1600×1180) and
// recoloured for the dark theme. Accent colours are inline oklch (independent
// of the Tailwind palette); neutrals use currentColor.
// ==========================================================================

const VB_W = 1600;
const VB_H = 1180;

type Variant = 'norm' | 'seq' | 'state' | 'ffn' | 'rope' | 'io' | 'header';

// Dark-theme accent colours (oklch). Mapping mirrors the artifact's legend:
// normalization=blue, sequence=purple, linear/state=teal, FFN=pink, RoPE=orange.
const ACCENT: Record<'norm' | 'seq' | 'state' | 'ffn' | 'rope', string> = {
  norm: 'oklch(0.72 0.13 235)', // blue
  seq: 'oklch(0.70 0.16 300)', // purple
  state: 'oklch(0.75 0.12 185)', // teal
  ffn: 'oklch(0.72 0.18 350)', // pink
  rope: 'oklch(0.80 0.13 80)', // orange
};

function accentFor(v: Variant): string | null {
  if (v === 'norm' || v === 'seq' || v === 'state' || v === 'ffn' || v === 'rope') return ACCENT[v];
  return null;
}

function PosterSvg({
  selectedId,
  onSelect,
  mixer,
  onMixerSelect,
}: {
  selectedId: string;
  onSelect: (id: string) => void;
  mixer: Mixer;
  onMixerSelect: (m: Mixer) => void;
}) {
  const ib = (props: Omit<IBoxProps, 'selectedId' | 'onSelect'>, key?: React.Key) => (
    <IBox key={key} {...props} selectedId={selectedId} onSelect={onSelect} />
  );

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="block h-auto"
      style={{ width: '100%', minWidth: 1280, maxWidth: VB_W }}
      role="group"
      aria-label="Qwen3.5-0.8B 交互式架构海报：中央是解码器主干，左侧展开 SwiGLU 前馈面板，右侧是 Gated DeltaNet 与门控 GQA 两个序列混合器面板，外加一条可选的视觉通路。"
    >
      <Markers />

      {/* ---- background dashed detail panels ---- */}
      <DashedPanel x={80} y={420} w={335} h={430} color={ACCENT.ffn} />
      <DashedPanel x={1160} y={190} w={350} h={496} color={ACCENT.seq} active={mixer === 'gdn'} />
      <DashedPanel x={1160} y={710} w={350} h={430} color={ACCENT.seq} active={mixer === 'gqa'} />
      <DashedPanel x={80} y={910} w={430} h={210} color={ACCENT.state} />

      {/* ---- soft container panels (model facts, decoder, text, legend) ---- */}
      <SoftPanel x={80} y={130} w={330} h={242} />
      <BigPanel x={500} y={320} w={600} h={640} />
      <SoftPanel x={520} y={1010} w={110} h={90} />
      <SoftPanel x={1160} y={110} w={350} h={52} />

      {/* ---- title ---- */}
      <text x={800} y={52} fontSize={30} fontWeight={700} textAnchor="middle" fill="currentColor" fillOpacity={0.95}>
        Qwen3.5-0.8B 混合多模态架构
      </text>
      <text x={800} y={82} fontSize={15} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        24 个稠密解码器层：{NUM_LINEAR} 层 Gated DeltaNet 线性注意力 + {NUM_FULL} 层门控 GQA 全量注意力
      </text>

      {/* ---- static text: model facts ---- */}
      <text x={105} y={164} fontSize={16} fontWeight={700} fill="currentColor" fillOpacity={0.9}>
        模型速览
      </text>
      {[
        '0.8B 稠密 checkpoint，无 MoE 路由',
        '隐藏维度 1024，24 个解码器层',
        '词表 248,320，LM head 权重共享',
        '上下文 262,144 个 token',
        'SwiGLU FFN 中间维度 3584',
        '视觉通路：12 层编码器，输出 1024',
      ].map((line, i) => (
        <text key={line} x={105} y={196 + i * 26} fontSize={12.5} fill="currentColor" fillOpacity={0.75}>
          {line}
        </text>
      ))}
      <text
        x={105}
        y={354}
        fontSize={11}
        fill="currentColor"
        fillOpacity={0.55}
        style={{ fontFamily: 'var(--font-mono, monospace)' }}
      >
        Text config: Qwen3_5ForConditionalGeneration
      </text>

      {/* ---- static text: legend ---- */}
      <text x={1182} y={132} fontSize={12.5} fill="currentColor" fillOpacity={0.72}>
        颜色图例：归一化为蓝色，序列混合为紫色，
      </text>
      <text x={1182} y={152} fontSize={12.5} fill="currentColor" fillOpacity={0.72}>
        线性/状态为青色，FFN 为粉色，RoPE 为橙色。
      </text>

      {/* ---- connector curves (drawn under boxes) ---- */}
      {/* FFN panel → in-stack Dense SwiGLU FFN */}
      <FlowCurve d="M 415 575 C 500 610, 560 595, 660 595" color={ACCENT.ffn} marker="ffn" />
      {/* sequence block → GDN panel / GQA panel (active one emphasised) */}
      <FlowCurve
        d="M 925 781 C 1040 760, 1080 540, 1160 515"
        color={ACCENT.seq}
        marker="seq"
        active={mixer === 'gdn'}
      />
      <FlowCurve
        d="M 925 781 C 1040 840, 1080 885, 1160 885"
        color={ACCENT.seq}
        marker="seq"
        active={mixer === 'gqa'}
      />
      {/* vision project → token embeddings */}
      <FlowCurve d="M 450 1070 C 550 1070, 600 1032, 658 1032" color={ACCENT.state} marker="state" />

      {/* ---- central spine arrows (data flows bottom → top) ---- */}
      <FlowLine x1={800} y1={176} x2={800} y2={142} />
      <FlowLine x1={800} y1={252} x2={800} y2={231} />
      <FlowLine x1={800} y1={326} x2={800} y2={300} />
      <FlowLine x1={800} y1={1005} x2={800} y2={961} />
      <FlowLine x1={800} y1={1108} x2={800} y2={1056} />
      <FlowLine x1={631} y1={1055} x2={658} y2={1035} />

      {/* ---- representative-layer arrows + residual skips ---- */}
      <FlowLine x1={800} y1={910} x2={800} y2={888} />
      <FlowLine x1={800} y1={838} x2={800} y2={810} />
      <FlowLine x1={800} y1={753} x2={800} y2={740} />
      <FlowLine x1={800} y1={696} x2={800} y2={684} />
      <FlowLine x1={800} y1={634} x2={800} y2={624} />
      <FlowLine x1={800} y1={567} x2={800} y2={562} />
      {/* attention residual skip — single L-path, one arrowhead into the ⊕ */}
      <SkipPath d="M 612 864 V 718 H 778" />
      {/* mlp residual skip — single L-path, one arrowhead into the ⊕ */}
      <SkipPath d="M 590 718 V 540 H 778" />

      {/* ---- FFN internal arrows ---- */}
      <ThinLine x1={248} y1={775} x2={174} y2={746} />
      <ThinLine x1={248} y1={775} x2={322} y2={746} />
      <ThinLine x1={174} y1={700} x2={215} y2={674} />
      <ThinLine x1={322} y1={700} x2={282} y2={674} />
      <FlowLine x1={248} y1={628} x2={248} y2={596} />
      <FlowLine x1={248} y1={550} x2={248} y2={520} />

      {/* ---- GDN internal arrows ---- */}
      <ThinLine x1={1335} y1={311} x2={1335} y2={336} />
      <ThinLine x1={1215} y1={383} x2={1265} y2={412} />
      <ThinLine x1={1275} y1={383} x2={1335} y2={412} />
      <ThinLine x1={1335} y1={383} x2={1400} y2={412} />
      <ThinLine x1={1395} y1={383} x2={1395} y2={412} />
      <FlowLine x1={1335} y1={465} x2={1335} y2={490} />
      <ThinLine x1={1270} y1={548} x2={1254} y2={564} />
      <ThinLine x1={1400} y1={548} x2={1416} y2={564} />
      <FlowLine x1={1335} y1={548} x2={1335} y2={628} />

      {/* ---- GQA internal arrows ---- */}
      <ThinLine x1={1300} y1={831} x2={1259} y2={854} />
      <ThinLine x1={1370} y1={831} x2={1426} y2={854} />
      <ThinLine x1={1259} y1={911} x2={1257} y2={940} />
      <ThinLine x1={1426} y1={911} x2={1429} y2={940} />
      <ThinLine x1={1257} y1={985} x2={1300} y2={1006} />
      <ThinLine x1={1429} y1={985} x2={1380} y2={1006} />
      <FlowLine x1={1339} y1={1051} x2={1339} y2={1072} />

      {/* ---- vision internal arrows ---- */}
      <FlowLine x1={234} y1={1000} x2={270} y2={1000} />
      <FlowLine x1={357} y1={1025} x2={357} y2={1050} />

      {/* ---- captions ---- */}
      <text x={248} y={452} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.ffn}>
        稠密 SwiGLU FFN
      </text>
      <text x={248} y={473} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        每一层都有，不是混合专家（MoE）
      </text>
      <text x={248} y={508} fontSize={13} fontWeight={600} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
        FFN 输出
      </text>
      <text x={1335} y={226} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.seq}>
        Gated DeltaNet 块
      </text>
      <text x={1335} y={247} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        线性注意力，循环式 O(n)
      </text>
      <text x={1335} y={746} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.seq}>
        门控 GQA 块
      </text>
      <text x={1335} y={767} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        全量注意力，预填充开销二次方
      </text>
      <text x={1339} y={1132} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
        KV 缓存：默认扁平布局；分块分页是仅原生服务端支持的可选路径，本演示未使用
      </text>
      <text x={295} y={943} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.state}>
        可选视觉通路
      </text>
      <text x={800} y={932} fontSize={12.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        代表性解码器层内部
      </text>

      {/* ---- layer schedule (inside decoder panel) ---- */}
      <ScheduleStrip mixer={mixer} onMixerSelect={onMixerSelect} selectedId={selectedId} />

      {/* ---- residual ⊕ nodes ---- */}
      <ResidualNode id="residual-1" cx={800} cy={718} selectedId={selectedId} onSelect={onSelect} />
      <ResidualNode id="residual-2" cx={800} cy={540} selectedId={selectedId} onSelect={onSelect} />

      {/* ================= interactive boxes ================= */}

      {/* central spine */}
      {ib({
        id: 'output',
        x: 700,
        y: 100,
        w: 200,
        h: 40,
        variant: 'io',
        label: '输出 logits',
        sub: `词表 ${VOCAB_LABEL}`,
      })}
      {ib({
        id: 'lm-head',
        x: 690,
        y: 178,
        w: 220,
        h: 52,
        variant: 'ffn',
        label: '权重共享 LM head',
        sub: '复用嵌入矩阵权重',
      })}
      {ib({ id: 'final-norm', x: 705, y: 255, w: 190, h: 44, variant: 'norm', label: '最终 RMSNorm' })}
      {ib({
        id: 'decoder',
        x: 540,
        y: 336,
        w: 520,
        h: 48,
        variant: 'header',
        label: `解码器堆叠 ×${NUM_LAYERS}`,
        sub: 'pre-norm 残差块 · 每层都有稠密 FFN',
      })}
      {ib({ id: 'pre-norm', x: 690, y: 840, w: 220, h: 46, variant: 'norm', label: 'RMSNorm', sub: '注意力前' })}
      <SequenceBox mixer={mixer} selectedId={selectedId} onSelect={onSelect} />
      {ib({ id: 'post-norm', x: 690, y: 636, w: 220, h: 46, variant: 'norm', label: 'RMSNorm', sub: 'MLP 前' })}
      {ib({
        id: 'mlp',
        x: 665,
        y: 568,
        w: 270,
        h: 54,
        variant: 'ffn',
        label: '稠密 SwiGLU FFN',
        sub: '1024 → 3584 → 1024，无 MoE',
      })}
      {ib({ id: 'embeddings', x: 660, y: 1006, w: 280, h: 48, variant: 'io', label: 'token + 视觉嵌入' })}
      {ib({
        id: 'input',
        x: 650,
        y: 1110,
        w: 300,
        h: 44,
        variant: 'io',
        label: '输入序列',
        sub: '文本 · 可选图像 · 可选视频',
      })}

      {/* left: FFN expanded */}
      {ib({ id: 'ffn-input', x: 143, y: 775, w: 210, h: 40, variant: 'io', label: '输入 hidden 1024' })}
      {ib({ id: 'ffn-gate', x: 114, y: 700, w: 120, h: 44, variant: 'ffn', label: 'gate Linear', sub: '1024 → 3584' })}
      {ib({ id: 'ffn-up', x: 262, y: 700, w: 120, h: 44, variant: 'ffn', label: 'up Linear', sub: '1024 → 3584' })}
      {ib({
        id: 'ffn-act',
        x: 156,
        y: 628,
        w: 184,
        h: 44,
        variant: 'rope',
        label: 'SiLU(gate) · up',
        sub: 'SwiGLU 激活',
      })}
      {ib({ id: 'ffn-down', x: 158, y: 550, w: 180, h: 44, variant: 'ffn', label: 'down Linear', sub: '3584 → 1024' })}

      {/* right top: GDN expanded */}
      {ib({ id: 'gdn-input', x: 1234, y: 275, w: 202, h: 36, variant: 'io', label: 'hidden 输入' })}
      {[
        { x: 1188, l: 'q,k', s: 'proj' },
        { x: 1248, l: 'v,z', s: 'proj' },
        { x: 1308, l: 'b,a', s: 'proj' },
        { x: 1368, l: 'conv', s: 'state' },
        { x: 1428, l: 'rec', s: 'state' },
      ].map((c, i) =>
        ib(
          {
            id: 'gdn-proj',
            x: c.x,
            y: 338,
            w: 54,
            h: 44,
            variant: 'state',
            label: c.l,
            sub: c.s,
            tiny: true,
          },
          `gdn-proj-${i}`,
        ),
      )}
      {ib({
        id: 'gdn-conv',
        x: 1200,
        y: 414,
        w: 270,
        h: 50,
        variant: 'norm',
        label: '逐通道 Conv1D，kernel 4',
        sub: 'q/k/v 通道，SiLU',
      })}
      {ib({
        id: 'gdn-recurrence',
        x: 1206,
        y: 492,
        w: 258,
        h: 56,
        variant: 'state',
        label: '门控 delta 循环',
        sub: 'β 与 g 更新循环状态',
      })}
      {[
        { x: 1190, l: '卷积缓存' },
        { x: 1352, l: '循环状态缓存' },
      ].map((c, i) =>
        ib(
          {
            id: 'gdn-cache',
            x: c.x,
            y: 566,
            w: 128,
            h: 36,
            variant: 'seq',
            label: c.l,
            tiny: true,
          },
          `gdn-cache-${i}`,
        ),
      )}
      {ib({ id: 'gdn-out', x: 1244, y: 630, w: 182, h: 38, variant: 'io', label: 'RMSNormGated + out proj' })}

      {/* right bottom: GQA expanded */}
      {ib({ id: 'gqa-input', x: 1234, y: 794, w: 202, h: 36, variant: 'io', label: 'hidden 输入' })}
      {ib({ id: 'gqa-qproj', x: 1188, y: 856, w: 142, h: 54, variant: 'ffn', label: 'q_proj', sub: 'Q + 门控' })}
      {ib({ id: 'gqa-kvproj', x: 1372, y: 856, w: 108, h: 54, variant: 'state', label: 'k/v_proj', sub: '2 个 KV 头' })}
      {ib({ id: 'gqa-qnorm', x: 1198, y: 942, w: 118, h: 42, variant: 'norm', label: 'Q 归一化', sub: '8 个头' })}
      {ib({ id: 'gqa-knorm', x: 1370, y: 942, w: 118, h: 42, variant: 'norm', label: 'K 归一化', sub: '2 个头' })}
      {ib({
        id: 'gqa-rope',
        x: 1220,
        y: 1008,
        w: 238,
        h: 42,
        variant: 'rope',
        label: '部分 RoPE',
        sub: '256 维中的 64 维，θ 1e7',
      })}
      {ib({
        id: 'gqa-sdpa',
        x: 1210,
        y: 1074,
        w: 258,
        h: 42,
        variant: 'seq',
        label: 'SDPA + sigmoid 门控',
        sub: 'o_proj → hidden 1024',
      })}

      {/* vision + text inputs */}
      {ib({
        id: 'vision-input',
        x: 118,
        y: 976,
        w: 115,
        h: 48,
        variant: 'state',
        label: '图像/视频',
        sub: 'patch 16，tubelet 2',
      })}
      {ib({
        id: 'vision-encoder',
        x: 272,
        y: 976,
        w: 170,
        h: 48,
        variant: 'norm',
        label: '视觉编码器 ×12',
        sub: 'hidden 768，12 个头',
      })}
      {ib({ id: 'vision-project', x: 266, y: 1052, w: 184, h: 38, variant: 'state', label: '投影到 hidden 1024' })}
      {ib({ id: 'text-input', x: 524, y: 1018, w: 102, h: 74, variant: 'io', label: '文本', sub: 'token id · 248k' })}

      {/* ---- auxiliary MTP draft head (greyed; not used by this demo) ---- */}
      {/* Greyed dashed panel + faint branch off the LM-head region, mirroring how
          the optional vision path is set apart. The box itself is interactive. */}
      <DashedPanel x={950} y={150} w={196} h={86} color="currentColor" />
      <path
        d="M 910 204 C 932 204, 938 193, 950 193"
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.25}
        strokeWidth={1.5}
        strokeDasharray="6 5"
      />
      <text x={1048} y={172} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.45}>
        本演示未使用
      </text>
      <g opacity={0.55}>
        {ib({
          id: 'mtp-head',
          x: 966,
          y: 184,
          w: 164,
          h: 44,
          variant: 'io',
          label: 'MTP head',
          sub: '投机解码 · 本演示未用',
        })}
      </g>
    </svg>
  );
}

// --------------------------------------------------------------------------
// Markers (arrow heads) — one neutral + one per accent used on curves.
// --------------------------------------------------------------------------
function Markers() {
  const head = (id: string, fill: string, opacity = 1) => (
    <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill={fill} opacity={opacity} />
    </marker>
  );
  return (
    <defs>
      {head('arch-arrow', 'currentColor', 0.5)}
      {head('arch-arrow-seq', ACCENT.seq)}
      {head('arch-arrow-state', ACCENT.state)}
      {head('arch-arrow-ffn', ACCENT.ffn)}
    </defs>
  );
}

// --------------------------------------------------------------------------
// Decoration: panels & flow lines.
// --------------------------------------------------------------------------
function SoftPanel({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={12}
      fill="currentColor"
      fillOpacity={0.02}
      stroke="currentColor"
      strokeOpacity={0.18}
      strokeWidth={1.2}
    />
  );
}
function BigPanel({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={12}
      fill="currentColor"
      fillOpacity={0.035}
      stroke="currentColor"
      strokeOpacity={0.25}
      strokeWidth={1.4}
    />
  );
}
function DashedPanel({
  x,
  y,
  w,
  h,
  color,
  active,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  active?: boolean;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={12}
        fill={color}
        fillOpacity={0.03}
        stroke={color}
        strokeOpacity={0.45}
        strokeWidth={1.6}
        strokeDasharray="8 6"
      />
      {active ? (
        <rect
          x={x - 4}
          y={y - 4}
          width={w + 8}
          height={h + 8}
          rx={14}
          fill="none"
          className="stroke-primary"
          strokeOpacity={0.85}
          strokeWidth={2}
        />
      ) : null}
    </g>
  );
}
function FlowLine({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="currentColor"
      strokeOpacity={0.4}
      strokeWidth={2}
      markerEnd="url(#arch-arrow)"
    />
  );
}
function ThinLine({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="currentColor"
      strokeOpacity={0.3}
      strokeWidth={1.5}
      markerEnd="url(#arch-arrow)"
    />
  );
}
// Right-angle residual bypass: one continuous path (riser → corner → into the
// ⊕) carrying a single arrowhead at the destination. Drawing it as one path
// avoids the spurious corner arrowhead produced when two separately-markered
// segments meet at the bend.
function SkipPath({ d }: { d: string }) {
  return (
    <path d={d} fill="none" stroke="currentColor" strokeOpacity={0.3} strokeWidth={1.5} markerEnd="url(#arch-arrow)" />
  );
}
function FlowCurve({
  d,
  color,
  marker,
  active,
}: {
  d: string;
  color: string;
  marker: 'seq' | 'state' | 'ffn';
  active?: boolean;
}) {
  // `active` undefined → always-on curve (FFN, vision). Defined → emphasis toggle.
  const on = active === undefined ? true : active;
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeOpacity={on ? 0.9 : 0.25}
      strokeWidth={1.8}
      strokeDasharray="8 6"
      markerEnd={`url(#arch-arrow-${marker})`}
    />
  );
}

// --------------------------------------------------------------------------
// Interactive box.
// --------------------------------------------------------------------------
type IBoxProps = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  variant: Variant;
  label: string;
  sub?: string;
  /** Render compact (two tiny stacked lines) for the small projection cells. */
  tiny?: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
};

function IBox({ id, x, y, w, h, variant, label, sub, tiny, selectedId, onSelect }: IBoxProps) {
  const selected = selectedId === id;
  const accent = accentFor(variant);
  const isHeader = variant === 'header';
  const select = () => onSelect(id);

  const cx = x + w / 2;
  const labelSize = isHeader ? 18 : tiny ? 11 : 13;
  const subSize = isHeader ? 12.5 : tiny ? 9.5 : 10.5;
  const labelY = isHeader ? y + 22 : sub ? y + h / 2 - 2 : y + h / 2 + 4.5;
  const subY = isHeader ? y + 40 : y + h / 2 + (tiny ? 11 : 13);

  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`${label}。查看详情。`}
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? (
        <rect
          x={x - 3}
          y={y - 3}
          width={w + 6}
          height={h + 6}
          rx={9}
          className="fill-primary/10 stroke-primary"
          strokeWidth={1.8}
        />
      ) : null}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={isHeader ? 8 : 6}
        fill={accent ?? (isHeader ? 'transparent' : 'currentColor')}
        fillOpacity={accent ? 0.16 : isHeader ? 0 : 0.06}
        stroke={accent ?? (isHeader ? 'transparent' : 'currentColor')}
        strokeOpacity={accent ? 0.7 : isHeader ? 0 : 0.35}
        strokeWidth={1.3}
      />
      <text
        x={cx}
        y={labelY}
        fontSize={labelSize}
        textAnchor="middle"
        fill={accent ?? 'currentColor'}
        fillOpacity={accent ? 1 : 0.9}
        style={{ fontWeight: isHeader ? 700 : 600 }}
      >
        {label}
      </text>
      {sub ? (
        <text x={cx} y={subY} fontSize={subSize} textAnchor="middle" fill="currentColor" fillOpacity={0.58}>
          {sub}
        </text>
      ) : null}
    </g>
  );
}

// The central sequence block — label/colour follow the mixer toggle.
function SequenceBox({
  mixer,
  selectedId,
  onSelect,
}: {
  mixer: Mixer;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const selected = selectedId === 'sequence';
  const color = mixer === 'gdn' ? ACCENT.state : ACCENT.seq;
  const x = 675;
  const y = 754;
  const w = 250;
  const h = 54;
  const cx = x + w / 2;
  const select = () => onSelect('sequence');
  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`序列块，当前为${mixer === 'gdn' ? 'Gated DeltaNet 线性注意力' : '门控 GQA 全量注意力'}。查看详情。`}
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? (
        <rect
          x={x - 3}
          y={y - 3}
          width={w + 6}
          height={h + 6}
          rx={9}
          className="fill-primary/10 stroke-primary"
          strokeWidth={1.8}
        />
      ) : null}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={color}
        fillOpacity={0.16}
        stroke={color}
        strokeOpacity={0.7}
        strokeWidth={1.3}
      />
      <text x={cx} y={y + 24} fontSize={14} textAnchor="middle" fill={color} style={{ fontWeight: 600 }}>
        序列块
      </text>
      <text x={cx} y={y + 42} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.62}>
        {mixer === 'gdn' ? 'Gated DeltaNet（线性）→' : '门控 GQA（全量）→'}
      </text>
    </g>
  );
}

// Residual ⊕ node.
function ResidualNode({
  id,
  cx,
  cy,
  selectedId,
  onSelect,
}: {
  id: string;
  cx: number;
  cy: number;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const selected = selectedId === id;
  const select = () => onSelect(id);
  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label="残差相加。查看详情。"
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? <circle cx={cx} cy={cy} r={23} className="fill-primary/10 stroke-primary" strokeWidth={1.8} /> : null}
      <circle
        cx={cx}
        cy={cy}
        r={20}
        fill="currentColor"
        fillOpacity={0.06}
        stroke="currentColor"
        strokeOpacity={0.5}
        strokeWidth={1.6}
      />
      <line x1={cx - 11} y1={cy} x2={cx + 11} y2={cy} stroke="currentColor" strokeOpacity={0.7} strokeWidth={2} />
      <line x1={cx} y1={cy - 11} x2={cx} y2={cy + 11} stroke="currentColor" strokeOpacity={0.7} strokeWidth={2} />
    </g>
  );
}

// --------------------------------------------------------------------------
// Layer schedule — faithful [GDN, GDN, GDN, Gated GQA] ×6 pattern. The three
// GDN cells select the linear mixer; the Gated GQA cell selects full.
// --------------------------------------------------------------------------
function ScheduleStrip({
  mixer,
  onMixerSelect,
  selectedId,
}: {
  mixer: Mixer;
  onMixerSelect: (m: Mixer) => void;
  selectedId: string;
}) {
  const cells: { x: number; w: number; label: string; m: Mixer }[] = [
    { x: 590, w: 78, label: 'GDN', m: 'gdn' },
    { x: 684, w: 78, label: 'GDN', m: 'gdn' },
    { x: 778, w: 78, label: 'GDN', m: 'gdn' },
    { x: 872, w: 92, label: 'Gated GQA', m: 'gqa' },
  ];
  // Schedule cells are click-only (focus must NOT flip the mixer), so they need
  // their own visible focus ring — tabbing to an inactive cell would otherwise
  // give keyboard users no indication of where focus landed.
  const [focusedCell, setFocusedCell] = React.useState<number | null>(null);
  return (
    <g>
      {/* dashed schedule container */}
      <rect
        x={540}
        y={390}
        width={520}
        height={116}
        rx={8}
        fill={ACCENT.seq}
        fillOpacity={0.03}
        stroke={ACCENT.seq}
        strokeOpacity={0.45}
        strokeWidth={1.6}
        strokeDasharray="8 6"
      />
      <text x={800} y={414} fontSize={13.5} fontWeight={700} textAnchor="middle" fill="currentColor" fillOpacity={0.7}>
        层类型排布，重复 6 次
      </text>
      {cells.map((c, i) => {
        const color = c.m === 'gdn' ? ACCENT.state : ACCENT.seq;
        const active = mixer === c.m;
        const sel = selectedId === 'sequence' && active;
        const focused = focusedCell === i;
        const select = () => onMixerSelect(c.m);
        return (
          <g
            key={`sched-${c.x}`}
            role="button"
            tabIndex={0}
            aria-pressed={active}
            aria-label={`${c.label} 层。高亮${c.m === 'gdn' ? '线性' : '全量'}序列混合器。`}
            onClick={select}
            onFocus={() => setFocusedCell(i)}
            onBlur={() => setFocusedCell((prev) => (prev === i ? null : prev))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                select();
              }
            }}
            style={{ cursor: 'pointer' }}
          >
            {sel ? (
              <rect
                x={c.x - 3}
                y={430}
                width={c.w + 6}
                height={50}
                rx={8}
                className="fill-primary/10 stroke-primary"
                strokeWidth={1.6}
              />
            ) : null}
            {focused && !sel ? (
              <rect
                x={c.x - 3}
                y={430}
                width={c.w + 6}
                height={50}
                rx={8}
                fill="none"
                className="stroke-primary"
                strokeWidth={1.6}
                strokeDasharray="4 3"
              />
            ) : null}
            <rect
              x={c.x}
              y={433}
              width={c.w}
              height={44}
              rx={5}
              fill={color}
              fillOpacity={active ? 0.22 : 0.1}
              stroke={color}
              strokeOpacity={active ? 0.9 : 0.45}
              strokeWidth={active ? 1.5 : 1}
            />
            <text
              x={c.x + c.w / 2}
              y={460}
              fontSize={c.label.length > 4 ? 12 : 14}
              textAnchor="middle"
              fill={color}
              style={{ fontWeight: 600 }}
            >
              {c.label}
            </text>
          </g>
        );
      })}
      <text x={1005} y={461} fontSize={14} fontWeight={700} fill="currentColor" fillOpacity={0.7}>
        ×6
      </text>
      <text x={800} y={496} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        全量注意力层：4、8、12、16、20、24。其余各层均为线性注意力。
      </text>
    </g>
  );
}

// --------------------------------------------------------------------------
// Detail panel below the poster. Related links are small secondary <Link>s.
// --------------------------------------------------------------------------
function DetailPanel({ detail }: { detail: Detail }) {
  return (
    <div className="rounded-md border border-border bg-card/40 p-4" aria-live="polite">
      <div className="text-sm font-semibold text-foreground">{detail.title}</div>
      {detail.facts ? (
        <div className="mt-1 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{detail.facts}</div>
      ) : null}
      <p className="mt-2 text-[13px] leading-relaxed text-foreground/85">{detail.body}</p>
      {detail.related && detail.related.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
          {detail.related.map((r) => (
            <ChapterLink
              key={r.chapterId}
              chapterId={r.chapterId}
              className="text-[11px] text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
            >
              相关章节：{r.label} →
            </ChapterLink>
          ))}
        </div>
      ) : null}
    </div>
  );
}

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { GenerationLoopTrace } from '../../widgets/GenerationLoopTrace';
import { KvGrowthCurve } from '../../widgets/KvGrowthCurve';
import { PrefillVsDecodeChart } from '../../widgets/PrefillVsDecodeChart';
import { RecallFailure } from '../../widgets/RecallFailure';
import { RunningStateVsCache } from '../../widgets/RunningStateVsCache';

/**
 * 第 12 章（KV 缓存与混合注意力）的简体中文译文 — 仅包含章节正文与
 * learning 数据。Try-It 演示（KvCacheDemo）保持英文，仍从英文模块渲染。
 * 所有数字均派生自 Qwen3.5-0.8B 的 config，与英文版逐一对应。
 */

const NUM_LAYERS = 24;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const FULL_ATTENTION_INTERVAL = 4;
const BYTES_PER_FLOAT = 2; // bf16

// 使用全量 softmax 注意力的层索引。与模型 config.json 中的 layer_types
// 数组一致：每第 4 层（按 1 起数）为全量注意力。
const FULL_LAYER_INDICES: number[] = Array.from({ length: NUM_LAYERS }, (_, i) => i).filter(
  (i) => (i + 1) % FULL_ATTENTION_INTERVAL === 0,
);
const NUM_FULL_LAYERS = FULL_LAYER_INDICES.length;
const NUM_LINEAR_LAYERS = NUM_LAYERS - NUM_FULL_LAYERS;

// GatedDeltaNet 每层的循环状态：固定大小的张量
// [linear_num_value_heads, value_head_dim, key_head_dim] = [16, 128, 128]，
// 把全部历史压缩在内 — 与序列长度无关。这些是 Qwen3.5 config 中线性注意力
// 的维度（与全量注意力的 8 头 × 256 head-dim 是两套维度）。
const LINEAR_NUM_HEADS = 16;
const LINEAR_HEAD_DIM = 128;
const LINEAR_STATE_SIZE_BYTES = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * BYTES_PER_FLOAT;

function fullLayerCacheBytes(seqLen: number): number {
  return 2 * NUM_KV_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT;
}
function gqaAllLayersBytes(seqLen: number): number {
  return NUM_LAYERS * fullLayerCacheBytes(seqLen);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

/**
 * 第 12 章的脚手架元数据 — 驱动 `<ChapterFrame>` 渲染的章节头、术语表、
 * 要点、练习和随堂测验。`chapterId` 必须与 `learn/chapters.ts` 中的
 * 'kv-cache' 条目一致。
 */
export const learning: ChapterLearningData = {
  chapterId: 'kv-cache',
  objective: '解释 KV 缓存为什么存在、它如何随上下文增长，以及 Qwen3.5 的混合注意力在内存上换来了什么。',
  problem:
    '如果不缓存 K 和 V，每解码一个 token 都要从头重做全部注意力计算——使生成的代价随上下文长度呈平方增长。',
  minutes: 10,
  glossary: [
    {
      term: 'KV cache',
      definition:
        '每一层为每个提示词 token 和已生成 token 存储的 K、V 张量，在各个 decode 步骤间复用，使注意力计算保持大致线性。',
    },
    {
      term: 'prefill',
      definition: '推理的第一阶段：整个提示词在一趟以矩阵乘法为主的传播中处理完毕，并为每个提示词 token 写入 K 和 V。',
    },
    {
      term: 'decode',
      definition: '逐 token 生成阶段：只计算新 token 的 Q/K/V，Q 在缓存的 K/V 历史上做注意力。',
    },
    {
      term: 'GatedDeltaNet',
      definition: 'Qwen3.5 采用的线性注意力变体。每层存储一个固定大小的循环状态，而不是逐 token 的 KV 缓存。',
    },
    {
      term: 'hybrid attention',
      definition: '在同一个层堆叠里混用全量 softmax 注意力层和线性注意力层。Qwen3.5 用 6 个全量层 + 18 个线性层。',
    },
    {
      term: 'linear state',
      definition:
        '每层一个形状为 [16, 128, 128] 的张量（线性 value 头数 × value 维度 × key 维度），把全部历史压缩在内；其大小与序列长度无关。',
    },
    {
      term: 'recurrent state (RNN-style)',
      definition:
        '一块固定大小的记忆，模型在每一步覆盖并向前滚动它——就像累计总和或滑动平均——而不是保留每一条历史记录。线性注意力层用它来保证内存占用不随上下文变长而增长。',
    },
  ],
  takeaways: [
    'KV 缓存的内存随上下文线性增长，长上下文时它会远超模型权重本身。',
    'Qwen3.5 的 6 全量 + 18 线性布局压平了缓存曲线，因为线性层只贡献一个常数，而不是随 token 数增长的块。',
    '全量注意力层负责"精确回忆某个特定 token"的任务；线性层在很紧的内存预算下承担语言流转的主体工作。',
  ],
  exercise: {
    prompt:
      '把上下文长度滑块从 1k 拖到 131k，同时观察三条 KV 缓存曲线。在 131k token 处，混合布局的“实际”曲线大约比 MHA 假想曲线小多少倍？在短上下文时，哪条曲线与混合曲线重合？',
    answer:
      '在 131k token 处，混合曲线大约比 MHA 低一个数量级（"Savings" 瓦片显示的是当前上下文下 GQA 相对混合布局的比值）。在短上下文时，线性层的常数状态主导了总量，所以混合曲线呈平台状，而 GQA 和 MHA 继续向下逼近它——只有当 seq_len 增长到足以主导每层的算式时，它们才会分开。',
  },
  quiz: [
    {
      id: 'q1-why-cache',
      prompt: '如果模型在 decode 阶段不维护 KV 缓存，会发生什么？',
      options: [
        { id: 'a', label: 'decode 仍然能工作，速度也不变。' },
        {
          id: 'b',
          label:
            '每个新 token 都要为整段历史重新计算 K 和 V，于是每一步 decode 的开销都和 prefill 一样贵——整个生成过程对上下文长度呈平方级。',
        },
        {
          id: 'c',
          label: '模型会失去 RoPE 位置信息。',
        },
      ],
      correctId: 'b',
      explanation:
        '旧 token 的 K 和 V 不依赖于新 token，所以缓存它们能把 decode 的代价从对序列长度的平方级降为线性级。',
    },
    {
      id: 'q2-linear-state-size',
      prompt: 'GatedDeltaNet 线性注意力层的内存开销如何随序列长度变化？',
      options: [
        { id: 'a', label: '像普通 KV 缓存一样随 seq_len 线性增长。' },
        {
          id: 'b',
          label:
            '不随之变化——循环状态是一个固定大小的张量 [16, 128, 128]（线性头数 × value 维度 × key 维度），无论流过多少 token 都不变。',
        },
        {
          id: 'c',
          label: '随 seq_len 平方增长。',
        },
      ],
      correctId: 'b',
      explanation: '这正是循环压缩的全部意义：状态大小不随序列增长而变化，代价是失去逐 token 的精确回忆能力。',
    },
    {
      id: 'q3-why-hybrid',
      prompt: '为什么 Qwen3.5 仍保留 6 个全量注意力层，而不是全部改用线性注意力？',
      options: [
        {
          id: 'a',
          label:
            '线性注意力压缩历史、丢失精确回忆；全量层保留了在模型需要时回看某个特定早期 token 的能力。',
        },
        {
          id: 'b',
          label: '全量层是 RoPE 正常工作的前提。',
        },
        {
          id: 'c',
          label: '长上下文下全量层比线性层跑得更快。',
        },
      ],
      correctId: 'a',
      explanation:
        '线性层廉价地处理模式延续；全量层处理"回看那个特定 token"的任务。混合布局以一小部分缓存代价换回了绝大部分质量。',
    },
  ],
};

export function KvCacheChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>KV 缓存与混合注意力</h1>
        <p>
          到目前为止，每一章描述的都是<em>一次前向传播</em>内部发生的事。但生成是一连串前向传播——每输出一个 token
          跑一次——朴素的实现里，每一次传播都要为历史中的每个 token 重做全部注意力计算。这会让
          decode（解码）的代价随上下文长度呈平方增长，一旦提示词里有一份 32k token
          的文档，显然就毫无希望了。解决办法是一项工程手段，而不是新架构：把步骤之间不变的部分缓存起来。这块缓存正是推理内存的大头，也左右着你在现代
          LLM 里看到的大多数架构决策。
        </p>

        <h2>Prefill 与 decode</h2>
        <p>
          推理分两个阶段。<strong>Prefill</strong>（预填充）一次性接收整个提示词，在一趟以矩阵乘法为主的大计算中算出每个
          token 的 K、V 和隐藏状态。<strong>Decode</strong>（解码）则逐 token 生成：把上一个输出送回模型，只为
          <em>这一个</em>新 token 计算 K/V/Q，所有更早 token 的 K/V 直接从缓存里读出，而不是重新计算。没有缓存，每一步
          decode 都会和 prefill 一样昂贵——有了它，decode 的代价大致与缓存大小成线性关系。
        </p>

        <h2>为什么缓存 K 和 V 行得通</h2>
        <p>
          在 decode 第 <code>t</code> 步，每一层都需要新 token 的注意力输出：{' '}
          <code>softmax(q_t · K^T / √d) · V</code>——和第 4 章一模一样的注意力公式。Q
          是全新的（它来自刚算出的隐藏状态）。新 token 的 K 和 V 也是新的。但 token <code>0..t-1</code> 的 K 和 V
          在更早的步骤里就已经算过了——{' '}
          <em>
            而且它们不依赖于 token <code>t</code>
          </em>
          。所以我们只存一次，之后一直读回来用。
        </p>
        <p>
          从形状上看，缓存为每一层存两个形状为 <code>[num_kv_heads, seq_len, head_dim]</code> 的张量。随着{' '}
          <code>seq_len</code> 增长，缓存线性增长。问题就出在这里：缓存内存与上下文长度成正比，上下文一长，它就会远超权重。
        </p>

        <GenerationLoopTrace />

        <h2>用 Qwen3.5-0.8B 算一笔账</h2>
        <p>
          每个全量注意力层、每个 token 的缓存开销是{' '}
          <code>
            2 · num_kv_heads · head_dim · 2 bytes = 2 · {NUM_KV_HEADS} · {HEAD_DIM} · {BYTES_PER_FLOAT} ={' '}
            {fullLayerCacheBytes(1)} bytes
          </code>
          。在 32k token 的上下文下，一个全量注意力层占{' '}
          <code>{formatBytes(fullLayerCacheBytes(32_768))}</code>。如果全部 {NUM_LAYERS}{' '}
          层都是全量注意力，整个模型的缓存将达到 <code>{formatBytes(gqaAllLayersBytes(32_768))}</code>——而这还只是
          <em>最小</em>的 Qwen3.5 型号。更大的 Qwen3.5 型号在此基础上继续放大。
        </p>
        <p>
          这就是为什么现代推理服务的头条数字是 KV
          缓存内存，而不是权重。也正因如此，一整代架构决策——多查询注意力、分组查询注意力、滑动窗口，以及各类线性注意力变体——全都围绕着“在不损失太多质量的前提下削减缓存”展开。
        </p>

        <h2>Qwen3.5 的答案：混合注意力</h2>
        <p>
          Qwen3.5 交错使用两种不同的注意力层。{NUM_LAYERS} 层中的<strong>六</strong>层（每四层一个，位于索引{' '}
          {FULL_LAYER_INDICES.join(', ')}）是常规的全量注意力层——就是第 4 章那种，其 KV 缓存随
          <code>seq_len</code> 线性增长。其余 {NUM_LINEAR_LAYERS} 层使用一种
          <strong>叫 GatedDeltaNet 的线性注意力变体</strong>——之所以叫“线性”，是因为它的开销与序列长度成正比，而不是像全量
          softmax 注意力那样与其平方成正比。GatedDeltaNet 层不逐 token 缓存 K 和 V，而是把全部历史压缩进一个形状为{' '}
          <code>
            [{LINEAR_NUM_HEADS}, {LINEAR_HEAD_DIM}, {LINEAR_HEAD_DIM}]
          </code>{' '}
          的<em>固定大小循环状态</em>（线性 value 头数 × value 维度 × key 维度——与全量注意力那 {NUM_HEADS} 个 head-dim 为{' '}
          {HEAD_DIM} 的头是两套不同的维度）。这个状态在每一步向前滚动，就像早期的循环神经网络（RNN）维护一份“迄今所见一切”的滚动摘要——与流过多少
          token 无关。
        </p>
        <p>
          在 {LINEAR_NUM_HEADS} 个线性头、<code>head_dim = {LINEAR_HEAD_DIM}</code> 的配置下，每个线性层的状态恒为{' '}
          <code>{formatBytes(LINEAR_STATE_SIZE_BYTES)}</code>，与上下文无关。所以 Qwen3.5-0.8B 的总缓存是{' '}
          <code>6 × full + 18 × constant</code>——右侧的图表把它和一个每层都用全量注意力的假想 Qwen 画在了一起。
        </p>
        <p>
          这也正是模型那个完整的{' '}
          <ChapterLink chapterId="architecture">262,144 token 窗口</ChapterLink>之所以负担得起的原因。设想把一部
          200 页的小说——就算它 262,144 个 token——整段粘进提示词。如果全部 {NUM_LAYERS}{' '}
          层都像那 {NUM_FULL_LAYERS} 个注意力层一样逐 token 保留完整的 KV
          缓存，缓存就会贴着图中那条琥珀色的"GQA 全层"曲线——在这个长度下大约是混合方案那几 GB 占用的 4
          倍（无论哪种方式，缓存都随上下文线性增长，混合架构只是把斜率压了下来）。可实际上，那 {NUM_LINEAR_LAYERS}{' '}
          个线性层无论上下文多长都只占恒定的 <code>{formatBytes(LINEAR_STATE_SIZE_BYTES)}</code>，于是窗口能在原来 32,768
          这道线之上再放大约 8 倍，而缓存不会爆。只有那 {NUM_FULL_LAYERS} 个全量注意力层仍要按 token 付出代价。
        </p>
        <p>
          下面把这种对比落到 token 级别：把同一条 token 流分别喂给两种记忆，看一边每来一个 token
          就追加一个格子，另一边只在原地更新同一个槽位。
        </p>

        <RunningStateVsCache />

        <h2>取舍</h2>
        <p>
          线性注意力是近似的。它无法像全量 softmax 注意力那样，精确回忆历史深处某个任意的单个 token——它的压缩状态把所有东西混在了一起。它擅长的是语言处理的主体工作：模式延续、句法走向、局部风格。与六个全量注意力层交错，就把精确回忆的能力补回到需要它的地方。具体来说：线性层负责长上下文的语言流转，全量层负责"回看
          <em>那个</em>特定 token"的任务。
        </p>
        <p>
          这件事的“人类一侧”你大概体会过：在一段很长的对话里，模型把线索跟丢了——你早早提过一个名字，四十轮之后它就溜走了。造成这种情况的原因有两种。第一，你可能已经滑过了
          262,144 token 这道上限，那么那一早期的轮次已经整段掉出窗口，干脆就没了。第二——也是更有意思的一种——那一轮其实还在窗口内，只是那个细节自始至终只活在那
          {NUM_LINEAR_LAYERS} 个线性层模糊的循环状态里，正是下面面板让你看到的那种被抹开的分布，而那 {NUM_FULL_LAYERS}{' '}
          个全量注意力层只是偶尔出手、还能把它钉住的安全网。两种失败都不是绝对的：一个名字重复得足够多就能熬过这种模糊，而即便滑过了上限，模型先前写下的一段摘要也能把这个事实一路带下去。
        </p>

        <RecallFailure />

        <h2>为什么这是未来的方向</h2>
        <p>
          当上下文长度突破 1M token，平方级内存的注意力将难以为继——缓存和计算都是如此。混合注意力是业界正在探索的技术之一。像
          Mamba 这样的纯状态空间模型——用上面 GDN
          那样的滚动状态彻底取代注意力——走得更远。趋势很清楚：缓存就是瓶颈，架构会继续围绕它被重塑。
        </p>

        <div className="not-prose my-4 rounded-md border border-amber-500/50 bg-amber-500/5 p-4">
          <div className="text-xs uppercase tracking-wider text-amber-700 dark:text-amber-300">常见误区</div>
          <div className="mt-1 text-sm font-semibold text-foreground">KV 缓存不是一个 Python dict。</div>
          <p className="mt-2 text-[12px] leading-relaxed text-foreground/85">
            人们有时把它想象成一个从 token 文本或位置映射到 (k, v) 元组的哈希表。不是的。缓存是一个预先分配好的张量，形状为{' '}
            <code className="rounded bg-background px-1 py-0.5 font-mono text-[11px]">
              [layers, 2, batch, kv_heads, max_seq, head_dim]
            </code>{' '}
            （其中 <code>2</code> 是 key/value 这一对——一块给 K，一块给
            V），每个注意力层向它写入、从中读取，并由一个位置计数器记录张量里有多少是有效数据。所谓的“键”是位置索引，不是哈希。它是一块缓冲区，不是哈希表。
          </p>
        </div>

        <h2>Prefill 与 decode：时间维度</h2>
        <p>
          内存是一个维度，墙钟时间是另一个。下图对比了 prefill（对整个提示词做一次并行矩阵乘法）和
          decode（许多次小而串行的矩阵乘法，每生成一个 token 一次）。
        </p>
        <p>
          为什么每 token 的差距如此巨大——大约 0.7 ms 对 200 ms，约 293×？一步 decode
          必须把模型的每个权重都从内存里读出来，而这趟搬运只换来<em>一个</em>{' '}
          token：它是内存受限的，没有任何可并行的对象。Prefill 同样的权重只读一次，却同时作用在全部 1,024 个提示词 token
          上，搬运成本被摊成 1,024 份。
        </p>
        <p>
          混合布局在这里还多出一道褶皱。线性层的循环状态必须严格地一次更新一个 token、按顺序来——没有第{' '}
          <code>t-1</code> 步的状态就算不出第 <code>t</code>{' '}
          步——而全量注意力层没有这条链，本可以一次性给所有位置打分。所以这份恒定内存，部分是用让出 attention
          在 decode 时的一些并行性换来的。（这只是 decode 阶段的问题：训练时线性注意力有 parallel-scan
          形式，能把丢掉的并行性找回来。）
        </p>

        <PrefillVsDecodeChart />

        <h2>缓存如何随上下文增长</h2>
        <p>
          这里单独画出了每种布局的"缓存 vs 上下文"曲线。公式在下方逐项拆开：
          <code>2 · L · kv_heads · d · seq · bf16</code> 里的每一个因子，眼下都有人在想办法削减。
        </p>

        <KvGrowthCurve />

        <p className="mt-6 text-muted-foreground">
          右侧的小组件画出了 MHA、纯 GQA 和 Qwen3.5
          混合布局的缓存曲线；拖动上下文滑块即可读出任意长度下的内存占用。它下方的层条展示精确的交错方式——点击任意一层，可查看该层在当前上下文长度下的缓存形态。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { FfnNeurons } from '../../widgets/FfnNeurons';
import { MlpNeuronPanel } from '../../widgets/MlpNeuronPanel';
import { SiluVsRelu } from '../../widgets/SiluVsRelu';
import { SwiGluVsPlain } from '../../widgets/SwiGluVsPlain';

/**
 * Scaffolding metadata for chapter 7 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[6].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'mlp',
  objective: '描述 SwiGLU 门控 MLP 如何变换单个 token 的隐藏状态，以及它与残差流之间的关系。',
  problem: '注意力在 token 之间混合信息，却做不了逐 token 的特征计算；这部分工作就发生在 MLP 里。',
  minutes: 8,
  glossary: [
    {
      term: 'MLP block',
      definition: 'Transformer 层内逐 token 的前馈子模块。在每个位置上运行同一个小网络。',
    },
    {
      term: 'SwiGLU',
      definition: '门控 MLP 变体：silu(gate_proj(x)) ⊙ up_proj(x)，再投影回去。⊙ 表示逐元素相乘。',
    },
    {
      term: 'SiLU',
      definition: '激活函数 z * sigmoid(z)。平滑，z 为较大正数时接近线性，并抑制负值——起到软门控的作用。',
    },
    {
      term: 'intermediate_dim',
      definition:
        'MLP 内部加宽的草稿空间。通常约为 hidden_dim 的 3-4 倍。Qwen3.5-0.8B 为 3584（hidden 为 1024）。',
    },
    {
      term: 'residual stream',
      definition: '贯穿模型全部深度、未归一化的隐藏状态；每个模块从中读取，再把结果加回去。',
    },
    {
      term: 'residual connection',
      definition:
        '即 x_out = x_in + sub_block(norm(x_in)) 中的 x_in + 部分。让梯度可以跳过子模块，保持健康。',
    },
  ],
  takeaways: [
    'MLP 对每个 token 各自独立处理——MLP 模块内部不存在跨 token 的信息流动。',
    'gate_proj/up_proj/down_proj 是模型中最大的参数块之一——约占 Qwen3.5-0.8B 的三分之一（≈264M），规模与嵌入表大致相当。',
    '每个 MLP 只向残差流写入一个小的修正量；预测来自许多次小写入的总和，而不是最后一个模块的输出。',
  ],
  exercise: {
    prompt:
      '自动运行结束后，查看 "MLP contribution across all layers" 长条图。点击长条最大的那一层，再点击非空长条中最小的某一层。两者的 "MLP / layer-output ratio" 百分比相比如何？',
    answer:
      '靠前或中间的层往往贡献最大，最深的几层贡献较小，但没有任何一层接近 100%。即使最高的长条，通常也只占该层完整输出 L2 的个位数到百分之十几——其余幅值在这一层开始之前就已经在残差流里了。',
  },
  quiz: [
    {
      id: 'q1-elementwise',
      prompt: '在 SwiGLU 中，silu(gate_proj(x)) 与 up_proj(x) 之间的 ⊙ 是哪种乘法？',
      options: [
        { id: 'a', label: '矩阵乘法。' },
        {
          id: 'b',
          label: '逐元素相乘：一个向量中 intermediate_dim 个特征里的每一个，都与另一个向量中对应位置的特征相乘。',
        },
        { id: 'c', label: '点积，结果是一个标量。' },
      ],
      correctId: 'b',
      explanation:
        '门控投影给出每个特征通过多少；up 投影给出通过的是什么。逐元素相乘把两者纠缠在一起，之后 down_proj 才把结果压回 hidden_dim。',
    },
    {
      id: 'q2-mlp-magnitude',
      prompt: '单个 MLP 模块的输出幅值，与它写入的残差流相比通常如何？',
      options: [
        {
          id: 'a',
          label: '小得多——它是叠加上去的修正量，不是替换。',
        },
        {
          id: 'b',
          label: '大致相等——每个 MLP 都会覆写残差流的幅值。',
        },
        {
          id: 'c',
          label: '大得多——残差相加之后 MLP 占主导。',
        },
      ],
      correctId: 'a',
      explanation:
        '这就是高速公路直觉：残差流的幅值已经累积了许多层；一个 MLP 只在上面加一个小增量。这也是演示中残差占比很小的原因。',
    },
    {
      id: 'q3-where-params-live',
      prompt: '为什么 MLP 是模型中最大的参数块之一？',
      options: [
        {
          id: 'a',
          label: 'MLP 权重以比注意力权重更高的精度存储。',
        },
        {
          id: 'b',
          label:
            'gate_proj、up_proj 和 down_proj 都很大（hidden_dim × intermediate_dim，且 intermediate_dim ≈ 3-4 × hidden_dim），所有层加起来约占模型的三分之一——是模型中最大的参数块之一。',
        },
        {
          id: 'c',
          label: 'MLP 模块在每层中重复的次数比注意力多。',
        },
      ],
      correctId: 'b',
      explanation:
        '这三个矩阵每个都是 hidden_dim × intermediate_dim。乘上全部层数后，它们约占模型参数的三分之一——是最大的参数块之一，规模与嵌入表大致相当。',
    },
  ],
};

export function MlpChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>MLP 模块：逐 token 的前馈计算</h1>
        <p>
          先说名字。<strong>MLP</strong> 是<em>多层感知机</em>（multi-layer
          perceptron）的缩写——最朴素的神经网络：一次矩阵乘法，接一个压缩函数（即&ldquo;激活函数&rdquo;），再接一次矩阵乘法。&ldquo;前馈&rdquo;（feed-forward）是同一个东西的另一个名字——数字一路直行，没有回环。
        </p>
        <p>
          注意力是 Transformer 中在 <em>token 之间</em>混合信息的部分：通过 softmax(QKᵀ)·V，每个位置都能读取其他所有位置。MLP
          模块恰恰相反——它对每个 token <em>各自独立</em>
          处理，在每个位置的隐藏向量上运行同一个小神经网络，并把结果写回该位置。跨 token 的混合发生在注意力里；逐 token
          的计算发生在 MLP 里。一个 Transformer 层就在两者之间交替。
        </p>

        <h2>先 pre-MLP 归一化，再门控 MLP，最后残差相加</h2>
        <p>与它的注意力同胞一样，Qwen3.5 的 MLP 子模块也包在 pre-norm + 残差的模式里：</p>
        <MathDisplay
          latex={String.raw`x_\text{out} = x_\text{in} + \text{mlp}\bigl(\text{rmsnorm}(x_\text{in})\bigr)`}
        />
        <p>
          <strong>残差连接</strong>就是其中的 <code>x_in +</code>{' '}
          部分。它的存在有两个理由。其一是梯度：反向传播可以沿恒等路径直接穿过，即使 24
          层的堆叠也能保持可训练。其二是语义：把残差流想象成一条贯穿模型全部深度的<em>高速公路</em>
          （下一章——完整的 Transformer
          块——会画出残差流）。每个模块从公路上读取，计算一个小的贡献，再加回去。模型的预测是每个模块贡献的累积——而不是最后一个模块单独的输出。
        </p>

        <p>
          看公式之前还有一件事：为什么需要压缩函数？因为把矩阵乘法直接叠起来、中间什么都不放，等于白叠——
          <code>A·(B·x)</code> 就是 <code>(A·B)·x</code>
          ，两个线性映射坍缩成一个。夹在中间的压缩函数正是阻止坍缩、让模块能算出真正新东西的关键。
        </p>

        <h2>&ldquo;门控 MLP&rdquo;到底指什么</h2>
        <p>
          Qwen3.5（与 Llama、Mistral 以及大多数现代开源 LLM 一样）使用
          <strong> SwiGLU 风格的门控 MLP</strong>。每层有三个线性投影：
        </p>
        <ul>
          <li>
            <code>gate_proj</code>：hidden_dim → intermediate_dim。其输出会经过 <strong>SiLU</strong> 激活函数（也叫
            Swish）：
            <code> silu(z) = z · sigmoid(z)</code>。SiLU 平滑，<code>z</code>{' '}
            为较大正数时近乎线性，但会抑制负值——一道软门。
          </li>
          <li>
            <code>up_proj</code>：hidden_dim → intermediate_dim。这是&ldquo;取值&rdquo;路径——真正被传递的特征。
          </li>
          <li>
            <code>down_proj</code>：intermediate_dim → hidden_dim。把逐元素乘积投影回残差流的宽度。
          </li>
        </ul>
        <MathDisplay
          latex={String.raw`\text{MLP}(x) = \text{down\_proj}\!\bigl(\,\text{silu}(\text{gate\_proj}(x)) \,\odot\, \text{up\_proj}(x)\bigr)`}
        />
        <p>
          仔细读这个表达式。<code>⊙</code> 是<em>逐元素</em>相乘，不是矩阵乘法（<code>⊙</code>{' '}
          表示逐元素相乘——对应位置的元素两两相乘）：<code>silu(gate_proj(x))</code> 中 intermediate_dim
          个特征里的每一个，都与 <code>up_proj(x)</code> 中对应的特征相乘。门控投影决定每个特征<em>通过多少</em>；up
          投影提供取值。两者按特征纠缠在一起后，再由 <code>down_proj</code> 压回 hidden_dim。
        </p>
        <p>
          亲手走一个特征。假设门控路径产生 <code>z = 2</code>，up 路径产生 <code>0.5</code>。那么{' '}
          <code>silu(2) = 2 / (1 + e⁻²) ≈ 1.76</code>，于是通过的特征是{' '}
          <code>1.76 × 0.5 ≈ 0.88</code>。如果门控是很大的负数，同样的取值就会被压向零。
        </p>
        <p>
          中间维度就是模型的&ldquo;草稿空间&rdquo;——这里约为隐藏维度的 3.5 倍（Qwen3.5-0.8B 在 1024 维隐藏状态上使用
          3584；3–4 倍在各模型间都很常见）。这三个矩阵是模型中最大的参数块之一：3 个矩阵 × 每个 1024 × 3584 个数 × 24 层 ={' '}
          <code>264,241,152 ≈ 264M</code>——约占 ~0.8B 总参数的三分之一，规模与嵌入表大致相当。
        </p>

        <FfnNeurons />

        <SiluVsRelu />

        <h2>为什么要门控——以及为什么它并没有变大</h2>
        <p>
          你可能会问为什么要三个矩阵。经典的前馈模块——原版 Transformer
          里的那个——只用两个：一个升维投影，一个固定的非线性（ReLU 或 GELU），再一个降维投影。SwiGLU
          保留了升维和降维矩阵，但增加了第三个——<code>gate</code>
          ——并把固定的非线性换成一个可学习的、乘性的非线性。门控不再对每个特征施加同一个阈值，而是让网络自己决定——按
          token、按特征——每个升维后的取值能存活<em>多少</em>
          。这种依赖输入的门控严格地比固定激活函数更具表达力，实践中也确实能训到更低的损失。
        </p>
        <p>
          自然的反驳是：第三个矩阵不是要多花 50% 的参数吗？在经典的 4× 中间维度下确实如此——三个 4× 宽的矩阵合计{' '}
          <code>12 h²</code>，而朴素模块是 <code>8 h²</code>。常规的解决办法（Llama、Mistral）是把中间维度缩到原来的约 ⅔（
          <code>≈ 8⁄3 · hidden</code>），于是三个更窄的矩阵正好落回 <code>8 h²</code>
          ——也就是下图所示的收支平衡点。切换开关，看长条在总长度不变的情况下重新分配比例。（Qwen3.5
          没有缩到那么窄——见图下方的说明。）
        </p>

        <SwiGluVsPlain />

        <p>管线讲完了——下面看看这些中间特征训练完成后究竟在<em>做</em>什么：</p>

        <MlpNeuronPanel />

        <h2>MLP 贡献的是一个小修正</h2>
        <p>
          令人意外的部分来了：MLP 的原始输出比它写入的残差流<em>小得多</em>。由于每个模块都是把输出<em>加</em>
          进残差流——从不覆写——这些贡献逐层累积，所以即使每次写入都很小，残差流的逐 token L2 仍会增长（L2
          范数就是向量的长度——各元素平方和的平方根）。每个 MLP
          的输出只是叠在上面的一个小增量——是修正，不是替换。看下面的图表：<strong>MLP 输出</strong>的逐 token L2
          只是该层完整输出 L2 的一小部分。残差流中的大部分幅值来自输入，而不是这一层的 MLP。
        </p>
        <p>
          这就是被量化了的&ldquo;高速公路&rdquo;直觉。全层长条图显示，各层 MLP
          贡献的大小随深度变化：有的层贡献多些，有的少些，但没有任何一层占据主导。预测从许多次小写入的总和中涌现——这恰恰是让深层
          Transformer 可学习的设计选择。
        </p>
        <p>
          关于 Qwen3.5 混合堆叠的一个统一要点：
          <strong>每一层——无论全量注意力还是线性注意力——都带有同一个 MLP 模块。</strong>两种层只在 token
          混合的那一半有差别；其后的 SwiGLU MLP 处处相同。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

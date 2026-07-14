// 中文（简体）版第 11 章正文 — 采样。
// Demo 组件保持英文，仍从英文模块 ../09-sampling.tsx 渲染；本文件只翻译
// 正文 Body 与 learning 数据。术语表的 term 字段保持英文原样。

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { NucleusSlider } from '../../widgets/NucleusSlider';
import { SamplingFailureModes } from '../../widgets/SamplingFailureModes';
import { SoftmaxTemperatureSteps } from '../../widgets/SoftmaxTemperatureSteps';
import { TemperatureStories } from '../../widgets/TemperatureStories';

const MAX_NEW_TOKENS = 6;
const TOP_K = 16;

/**
 * Scaffolding metadata for chapter 11 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match the 'sampling' entry in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'sampling',
  objective:
    '拿到一个 logits 向量，应用 temperature 和 top-p，并能说清每个旋钮为什么会让最终分布发生那样的变化。',
  problem: '每次前向传播都以 logits 结束；模型仍然需要一条规则，把这个向量变成一个具体的下一个 token。',
  minutes: 7,
  glossary: [
    {
      term: 'logit',
      definition: '词表中每个条目对应的一个无界实数分数。模型在任何归一化之前的原始输出。',
    },
    {
      term: 'softmax',
      definition: 'exp(l_i / T) / sum_j exp(l_j / T)——把 logits 变成一个总和为 1 的概率分布。',
    },
    {
      term: 'temperature (T)',
      definition: '施加在 softmax 之前的锐度旋钮。T<1 让分布更尖（更接近贪心）；T>1 让分布更平（更多样）。',
    },
    {
      term: 'greedy decoding',
      definition: '每一步都取 argmax(logits)。确定性、可复现；但容易陷入重复循环。',
    },
    {
      term: 'top-p (nucleus)',
      definition: '保留概率累加到 >= p 的最小 token 集合，重新归一化后从中采样。把长尾砍掉。',
    },
    {
      term: 'top-K sampling',
      definition: '一种采样规则：只保留概率最高的 K 个 token，重新归一化后从中采样。是 top-p 的固定大小近亲。',
    },
    {
      term: 'top-16 capture (this demo)',
      definition:
        '这是展示层面的截断，不是采样：inspector 每一步记录 16 个最高的 logits，组件才有东西可画。模型本身仍然是在整个词表上采样的。',
    },
  ],
  takeaways: [
    'logits 只有经过 softmax 才变成概率。在同一步之内，重要的是它们的相对顺序——最大的 logit 就是贪心选择——但绝对值在不同运行或不同模型之间不可比。',
    'temperature 重塑分布；top-p 截断长尾。常见默认值是 T=0.7 搭配 top_p=0.9。',
    '被高亮的柱子永远是模型在这一步的贪心（argmax）选择——捕获这次运行时用的是 temperature=0，之后无论怎样重新缩放或截断都不可能改变哪根柱子最高。滑块真正改变的是它和第二名柱子之间的置信度差距：差距大，说明真正的采样器几乎总会选出和贪心一样的结果；差距小，说明它常常会选到别的 token。',
  ],
  exercise: {
    prompt:
      '自动运行结束后，保持 temperature 为 1.0，慢慢把 top-p 从 1.0 降到 0.3，同时观察第 1 步的柱状图。降到哪个值时图表明显坍缩成一两根柱子？重新归一化后的柱高告诉你什么？',
    answer:
      '大约在 top_p ≈ 0.5-0.7 时长尾消失，最上面的两三根柱子猛涨，因为幸存下来的概率质量被重新归一化到总和为 1。到 top_p = 0.3 时通常只剩一根接近 100% 的柱子——top-p 实际上已把这一步变成了贪心解码。坍缩点取决于这一步的置信程度：置信的一步即使在非常小的 p 下也能存活，不确定的一步则需要更大的 p。',
  },
  quiz: [
    {
      id: 'q1-greedy-issue',
      prompt: '为什么生产环境的 LLM 服务不总是用贪心解码？',
      options: [
        { id: 'a', label: '贪心在 GPU 上太慢。' },
        {
          id: 'b',
          label: '贪心是确定性、可复现的，但容易重复——模型一旦找到一个高置信的循环，就会不断重新进入它。',
        },
        {
          id: 'c',
          label: '贪心与字节对编码（BPE）不兼容。',
        },
      ],
      correctId: 'b',
      explanation:
        '贪心是一个完全合法的解码器，只是太脆。一点小小的随机化（T 或 top-p）正是让模型跳出局部最优循环的手段。',
    },
    {
      id: 'q2-temperature-effect',
      prompt: '把 temperature 设为 T = 0.2 会对概率分布产生什么影响？',
      options: [
        {
          id: 'a',
          label: '让它更尖——高 logit 的 token 比 T=1 时更占主导，输出表现更接近贪心。',
        },
        {
          id: 'b',
          label: '让它更平——logits 之间的细小差异被抹平。',
        },
        {
          id: 'c',
          label: '把长尾完全去掉（等价于 top-p）。',
        },
      ],
      correctId: 'a',
      explanation: 'T < 1 时，logits 在取指数之前被一个小于 1 的数除，差异被放大。分布因此更尖。',
    },
    {
      id: 'q3-top-p-mechanism',
      prompt: 'p = 0.9 的 top-p 采样意味着：',
      options: [
        {
          id: 'a',
          label: '把每个概率都乘以 0.9。',
        },
        {
          id: 'b',
          label: '保留概率累加到 >= 0.9 的最小 token 集合，把这个集合重新归一化到总和为 1，再从中采样。',
        },
        {
          id: 'c',
          label: '丢掉第 90 百分位的 token。',
        },
      ],
      correctId: 'b',
      explanation:
        'top-p 是动态的：在置信的一步，核可能只有 2-3 个 token；在不确定的一步，可能有几十个。这正是它比固定的 top-K 适应性更好的原因。',
    },
  ],
};

export function SamplingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>采样：把 logits 变成下一个 token</h1>
        <p>
          Qwen3.5 的每次前向传播都以同样的方式结束：一个 <strong>logits</strong> 向量——模型词表（约 248k）中每个
          token 一个实数。logit 是一个无界的分数：模型<em>有多强烈地</em>推荐这个 token
          作为下一个。采样就是把这个向量变成一个具体选择的那一步。
        </p>

        <h2>为什么不直接选最大的？</h2>
        <p>
          最简单的规则是<strong>贪心解码</strong>（greedy decoding）：每一步都取 <code>argmax(logits)</code>——
          <code>argmax</code>{' '}
          的意思就是“选数值最大的那个位置”，也就是得分最高的单个
          token。它确定且可复现，但有一个著名的失败模式——重复。一旦模型找到一个它对续写很有信心的短语，就会不断重新进入同一个循环，因为那个循环永远是局部最优的选择。贪心解码还丢掉了大量信息：如果两个
          token 的 logits 几乎相等，选其一而完全无视另一个是一种脆弱的平局裁决。
        </p>

        <h2>Softmax：logits → 概率</h2>
        <p>
          要采样，我们先用 <strong>softmax</strong> 函数把 logits 转换成一个真正的概率分布：
        </p>
        <MathDisplay latex={String.raw`p_i = \frac{\exp(l_i / T)}{\sum_j \exp(l_j / T)}`} />
        <p>
          这里发生了两件事。指数让每个值都变成正数；除以总和让它们加起来等于 1。每个 logit 在取指数之前还会先除以一个{' '}
          <strong>temperature</strong>（温度）<code>T</code>——同一个 <code>T</code>{' '}
          出现在分子和分母的每一项里。temperature 起到锐度旋钮的作用：
        </p>
        <ul>
          <li>
            <code>T &lt; 1</code> 让分布更尖——高 logit 的 token 更加占主导，输出看起来更像贪心。
          </li>
          <li>
            <code>T &gt; 1</code> 让分布更平——logits 之间的细小差异被抹平，输出更多样但连贯性更差。
          </li>
          <li>
            <code>T = 0</code> 退化为贪心（argmax）。组件会把它钳制到一个极小的正数以避免除以零，数值上等价。
          </li>
        </ul>

        <h2>为什么偏偏是这个函数？</h2>
        <p>
          softmax 其实是 <code>argmax</code> 的一个平滑、可微的替身。<code>argmax</code> 会硬生生地跳到最大的那个
          logit 上，softmax 则在候选之间平滑地滑动。喂给它 <code>[5.0, 4.9, 1.0]</code>，你会得到大约{' '}
          <code>[0.52, 0.47, 0.01]</code>：两个相近的领先者几乎均分了概率质量，而第三个仍然小到可以忽略。把差距拉大到{' '}
          <code>[8.0, 4.9, 1.0]</code>，输出就锐化到大约 <code>[0.96, 0.04, 0.00]</code>——几乎是一个硬{' '}
          <code>argmax</code>，而这正是贪心解码（<code>T = 0</code>）所趋近的极限。
        </p>
        <p>
          这种平滑性也正是 softmax 远不止用于采样的原因：因为它处处都有干净的梯度，它既是模型训练时所对照的那个函数，也是
          <strong>注意力</strong>内部用来给各个 key 称重的同一个归一化。
        </p>

        <p>
          要看清分布<em>为什么</em>会长成那个样子，下面把这次计算拆成四个机械的阶段——原始 logits、除以{' '}
          <code>T</code>、取指数、归一化——一步一步循环播放。用 T 按钮对同样的八个 logits 分别跑 0.2、0.7 和
          2.0：归一化阶段的柱子在 0.2 时尖锐地集中到一个 token 上，在 2.0 时摊平到全部八个上。
        </p>

        <SoftmaxTemperatureSteps />

        <h2>同一个种子，三种 temperature</h2>
        <p>
          上面的柱子是因，这里是果。我们把同一段种子文本交给真实的 Qwen3.5-0.8B 检查点——正是这个 playground
          运行的那一个——跑了三次，只改 temperature。在 <code>T = 0</code> 时采样器是贪心的：每一步都取概率最高的单个
          token，所以每次重跑的输出完全相同，并且倾向于最保险、最平淡的措辞——注意它多快就开始绕着同样几个词打转。在{' '}
          <code>T = 0.7</code> 时分布被锐化但没有坍缩，采样出的故事——至少这一次运行——读起来很自然；重跑会得到不同的文本。
        </p>
        <p>
          <code>T = 1.5</code> 的运行展示了高 temperature 为什么劣化得这么快：生成是一个循环，每个采样出的 token
          都会被追加到上下文里，去产生<em>下一步</em>
          的分布。一次低概率的选择本来还能挽回——但在高 temperature
          下，低概率的选择会接连发生，每一次都把上下文拖得离模型见过的任何东西更远——在这次运行里，错误不断累积，直到输出变成词语沙拉。
        </p>

        <TemperatureStories />

        <h2>Top-p（核）采样</h2>
        <p>
          即便 temperature
          取得合理，词表的长尾仍然带着微小但非零的概率质量——偶尔采样器就会落在那里。这些尾部 token
          大多在上下文里是胡话。<strong>Top-p 采样</strong>（也叫<strong>核采样</strong>，nucleus
          sampling）负责修剪尾巴：
        </p>
        <ul>
          <li>把 token 按概率从高到低排序。</li>
          <li>
            沿排序后的列表累加概率，直到累计和达到 <code>p</code>。
          </li>
          <li>之后的全部丢弃。</li>
          <li>把幸存者重新归一化，使它们的总和重新等于 1。</li>
          <li>
            从这个“核”中采样：在 0 到 1 之间抽一个随机数 <code>r</code>
            ，沿列表向下累加概率，停在累计值第一个超过 <code>r</code> 的 token 上——切片越大被命中的次数越多。
          </li>
        </ul>
        <p>
          常见默认值是 <code>T = 0.7</code> 搭配 <code>top_p = 0.9</code>：temperature
          给模型留出一些发挥创造力的空间，top-p
          则保证我们永远不会从荒谬的尾部采样。这里的组件可以让你在一次已捕获的运行上扫动这两个旋钮，看看“本来会发生什么”。
        </p>

        <p>
          拖动下面的截断滑块，观察“核”如何形成：token 自上而下被保留，直到累计概率达到 <code>p</code>
          ，尾部被丢弃，幸存者被重新归一化到总和为 1。
        </p>

        <NucleusSlider />

        <h2>这个组件如何工作</h2>
        <p>
          按下 <em>Run</em> 会生成 <code>{MAX_NEW_TOKENS}</code> 个 token，inspector 在每一步捕获 top-
          <code>{TOP_K}</code> 个原始 logits——捕获本身永远是在 temperature=0（也就是贪心）下运行的。temperature 和
          top-p 滑块随后在缓存的 logits 上重新应用 softmax + 截断——不会重新运行模型。因为把每个 logit 都除以同一个{' '}
          <code>T</code> 不会改变它们的相对顺序，top-p 也永远不会丢掉排名第一的 token，所以被高亮的柱子在任何滑块设置下都
          <em>必然</em>是最高的那根——它永远是模型在这一步的贪心选择。随着你拖动滑块，真正变化的是它与第二名柱子之间的
          <strong>置信度差距</strong>：差距大，说明真正的采样器几乎总会认同这个贪心选择；差距小，说明它常常会不认同，落到另一个
          token 上。
        </p>
        <p className="text-muted-foreground">
          关于柱子高度的一点说明：这些柱子只在捕获的 top-<code>{TOP_K}</code> 个 logits 上做 softmax/top-p
          重新归一化，而不是在完整的 248,320 个 token
          的词表上。被省略的尾部仍然携带真实的概率质量，所以每根柱子读起来都比它真实的全词表概率略高——在高
          temperature 下最明显，因为那时模型把更多质量摊进了被丢弃的尾部。
        </p>

        <h2>当采样出问题时</h2>
        <p>
          有两种失败模式值得并排观看：一个低 temperature 的贪心运行陷入循环，一个高 temperature
          的运行变成胡言乱语。中间的面板展示的是一个合理的生产环境设置。
        </p>

        <SamplingFailureModes />

        <p className="mt-6 text-muted-foreground">
          提示：在同一个置信的步骤上分别试试低 temperature（0.2）和高
          temperature（1.5）。柱状图肉眼可见的“形状”变化，就是采样参数对 LLM 至关重要的全部原因。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

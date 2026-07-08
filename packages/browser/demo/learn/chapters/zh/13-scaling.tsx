// 第 14 章「扩展与正则化」的简体中文译文。与英文版 ../13-scaling.tsx 平行维护：
// 同名导出 learning 与 ScalingChapterBody。本章为纯阅读内容，无演示组件。

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { GradClipDemo } from '../../widgets/GradClipDemo';
import { LrScheduleViz } from '../../widgets/LrScheduleViz';
import { ScaleLadder } from '../../widgets/ScaleLadder';
import { WarmupLossCurve } from '../../widgets/WarmupLossCurve';

export const learning: ChapterLearningData = {
  chapterId: 'scaling',
  objective: '读懂一份有代表性的 LLM 训练配置（学习率、预热、裁剪、权重衰减），并解释每个旋钮在防止什么出错。',
  problem: '在深层 Transformer 上，交叉熵 + AdamW 还不够——没有特定的工程技巧，损失会发散、梯度会爆炸、模型会过拟合。',
  minutes: 7,
  glossary: [
    {
      term: 'AdamW',
      definition: 'LLM 预训练的主流优化器。Adam（逐参数自适应步长）加上解耦的权重衰减（即那个 W）。',
    },
    {
      term: 'learning rate warmup',
      definition:
        '在训练最初约 1-10% 的步数里，从 0 线性爬升到 lr_max。让优化器的滑动平均先稳定下来，再迈大步。',
    },
    {
      term: 'cosine decay',
      definition: '预热之后，学习率沿半个余弦波从 lr_max 降到 lr_min，覆盖剩余的全部步数。LLM 预训练的标准做法。',
    },
    {
      term: 'gradient clipping',
      definition: '若 ||g|| > c（通常为 1.0），就把 g 缩放为 c/||g||。当坏 batch 产生巨大梯度时给步长封顶。',
    },
    {
      term: 'weight decay',
      definition:
        '与 ||θ||^2 成正比的惩罚项，加进损失（或者像 AdamW 那样，直接从权重里减去）。把权重拉向零，起正则化作用。',
    },
    {
      term: 'dropout',
      definition:
        '训练时随机把一部分激活置零，防止模型过度依赖任何单一特征。旧模型里常见；现代 LLM 预训练往往不用——数据集的多样性完成了正则化。',
    },
  ],
  takeaways: [
    '深层 Transformer 不是用固定学习率训练的——预热接余弦是标准调度，也是训练初期损失曲线保持正常的原因。',
    '梯度裁剪是一行代码的护栏，把“模型在第 3,247 个 batch 上发散了”变成损失曲线上的一个小鼓包。',
    'dropout 基本退出了现代 LLM 预训练；权重衰减 + 数据规模 + 早停纪律取代了它。',
  ],
  exercise: {
    prompt:
      '在学习率小部件里，把预热设为 0、峰值学习率设为约 1e-3，然后看曲线的开头。对一个全新初始化的模型，用这个学习率直接开训会发生什么？余弦的那一半为什么存在？',
    answer:
      '没有预热、学习率 1e-3 时，第一步就会在几乎完全随机的权重上做一次巨大更新。AdamW 的滑动平均还没积累起来，每个参数的步长约为 lr * grad——很可能一步跨进优化器无法恢复的区域。余弦衰减解决的是结尾处相反的问题：模型逼近一个好的极小值时，小步长让它安顿下来而不是被弹出去。调度呈山丘形（低、峰、低），正是这两个原因的合并。',
  },
  quiz: [
    {
      id: 'q1-warmup',
      prompt: '为什么 LLM 训练要从学习率预热开始？',
      options: [
        { id: 'a', label: '让 GPU 在正式训练前先热热身。' },
        {
          id: 'b',
          label: '优化器对梯度和梯度平方的滑动平均还没有填满——早期迈大步会导致发散。',
        },
        { id: 'c', label: '预热是基础模型训练的监管要求。' },
      ],
      correctId: 'b',
      explanation:
        'AdamW 跟踪 g 和 g² 的滑动平均。第 1 步时两者都为零，逐参数步长基本就是原始梯度。预热让步长保持小，直到平均值稳定下来。',
    },
    {
      id: 'q2-clip',
      prompt: '“把梯度范数裁剪到 1.0”做的是什么？',
      options: [
        {
          id: 'a',
          label: '若 ||g|| > 1.0，把每个分量都按 1.0 / ||g|| 缩放。方向不变，幅度封顶。',
        },
        {
          id: 'b',
          label: '把超出 ±1.0 范围的单个梯度分量截断为 ±1.0。',
        },
        {
          id: 'c',
          label: '直接丢弃幅度最大的那个梯度分量。',
        },
      ],
      correctId: 'a',
      explanation:
        '按范数裁剪保留梯度的方向——只对全局幅度封顶。逐分量裁剪（选项 b）是另一种更少见的配方。',
    },
    {
      id: 'q3-dropout-modern',
      prompt: '为什么现代 LLM 预训练配置不再像 2017 年的模型那样使用 dropout？',
      options: [
        {
          id: 'a',
          label: 'dropout 在数学上等价于 RMSNorm；现代模型二选一。',
        },
        {
          id: 'b',
          label:
            '在数万亿个不重复 token 上预训练，数据丰富到模型没有机会过拟合任何单个样例；dropout 变得多余，还拖慢训练。',
        },
        { id: 'c', label: 'dropout 会破坏残差流。' },
      ],
      correctId: 'b',
      explanation:
        'dropout 是数据受限时代的正则化手段。LLM 预训练对每个 token 大约只看一次，dropout 要防的那种过拟合失效模式根本不会出现。',
    },
  ],
};

export function ScalingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>扩展与正则化：让训练循环真正收敛</h1>
        <p>
          第 13 章把训练化简成一行交叉熵。这个描述正确，但作为配方<em>惊人地不够用</em>。一个用朴素 SGD
          在数千亿 token 上训练的 24 层 Transformer 会：在前 100 步内发散；因为单个坏 batch 让损失飙到{' '}
          <code>NaN</code>（Not-a-Number——算术爆炸到浮点数装不下）；落进一个<em>泛化</em>
          很差（训练文本上表现好、没见过的文本上表现差）的局部极小值。解法是一小撮工程技巧——没有一个属于模型架构，但每一次现代训练都全部用上。
        </p>

        <p>
          先问一句：这里说的“扩展”到底是什么？标题里这个词指的是参数量。一条直线 <code>y = a·x + b</code>{' '}
          有两个参数。这个模型有将近十亿个。最大的研究模型则有几千亿个。下面这个阶梯把它们全都放到同一条对数轴上，好让这道鸿沟看得清楚——也好对“这个模型究竟处在什么位置”保持诚实。
        </p>

        <ScaleLadder
          renderLink={(slug, text) => <ChapterLink chapterId={slug}>{text}</ChapterLink>}
        />

        <p>
          在讲技巧之前，先说整个扩展故事赖以成立的一个前提：Transformer 之所以<em>值得</em>
          扩展，是因为它的训练计算几乎全是对所有位置同时进行的矩阵乘法。RNN 必须算完第 <em>i</em> 个 token
          才能碰第 <em>i+1</em> 个；Transformer 一次并行前向就处理整个序列——
          <ChapterLink chapterId="attention">因果掩码</ChapterLink>
          正是让这次并行保持诚实的那道闸门——而巨大的批量矩阵乘法恰好是 GPU 为之而生的工作负载。（我们这个模型的
          GatedDeltaNet 层在解码时确实是逐 token
          循环运行的；并行优势说的是训练和预填充。）这才是这个架构真正买到的东西：不是某种单一的聪明行为，而是便宜到可以一路扩展、直到聪明行为自己涌现的计算。
        </p>

        <h2>优化器：AdamW，而不是 SGD</h2>
        <p>
          朴素的随机梯度下降（<code>θ := θ - η·g</code>——把每个权重 <code>θ</code> 沿其梯度 <code>g</code>{' '}
          的反方向微调，按学习率 <code>η</code> 缩放）在 Transformer
          上效果不佳。不同参数看到的梯度量级天差地别——单一的全局步长，要么对响亮的参数太大，要么对安静的参数太小。
          <strong>Adam</strong> 为每个参数维护 <code>g</code> 和 <code>g²</code> 的滑动平均，再按 <code>√g²</code>{' '}
          归一化迈步，于是每个参数的步长由它<em>自己的</em>
          梯度历史重新标定，而不是共用一个速率。“滑动平均”（论文称之为<em>矩，moments</em>）一点也不玄：{' '}
          <code>avg ← 0.9·avg + 0.1·g</code>——保留 90% 的昨日估计，混入 10% 的新梯度。<strong>AdamW</strong> 加上了
          <em>解耦权重衰减</em>：不是通过损失去惩罚 <code>||θ||²</code>，而是优化器在每一步直接从 <code>θ</code>{' '}
          里减去它的一小部分。为什么要把权重往零拉？大权重让单个特征压过一切；保持权重小，迫使模型把证据分散到许多特征上。这是每个现代
          LLM 预训练的标准配方。
        </p>

        <h2>学习率调度：预热 + 余弦</h2>
        <p>
          这是 LLM 训练里最普及的技巧。学习率不是常数——它走一条山丘形的曲线：
        </p>
        <ul>
          <li>
            <strong>线性预热</strong>：训练最初约 1-10% 的步数里，从 0 升到 <code>lr_max</code>。
          </li>
          <li>
            <strong>余弦衰减</strong>：剩余步数里，从 <code>lr_max</code> 降到 <code>lr_min</code>（≈{' '}
            <code>1e-5</code>）。
          </li>
        </ul>
        <p>
          预热存在，是因为 AdamW 的滑动平均需要几百步的梯度历史才有意义；第 1
          步就迈满步长，基本等于朝随机方向开枪。余弦衰减存在，是因为训练后期受益于越来越小的步长——模型离极小值更近，大步会把它弹出去。
        </p>

        <LrScheduleViz />

        <WarmupLossCurve />

        <h2>梯度裁剪：一行代码的护栏</h2>
        <p>
          单个坏 batch——比如一段全是空白的文本，或一个分词器的边角案例——能产生巨大的梯度。没有保护时，AdamW
          会尽职尽责地朝那个方向迈出巨大一步，损失从 2.5 跳到
          8.0，模型要花几百步才能恢复（如果还能恢复的话）。
        </p>
        <p>
          按范数裁剪（clip-by-norm）是通用答案。计算模型全部梯度的全局 L2 范数，若超过裁剪阈值 <code>c</code>
          （几乎总是 <code>1.0</code>），就把每个分量按 <code>c / ||g||</code>{' '}
          缩放。方向不变、幅度封顶，训练平稳继续。下面的小部件让你把两个旋钮都拧拧看。
        </p>

        <GradClipDemo />

        <h2>正则化：dropout 的缓慢退场</h2>
        <p>
          原始 Transformer 论文激进地使用 dropout——每个注意力层、每个 MLP、每条残差连接。现代 LLM 预训练配置通常把
          dropout 设为 0。两个原因：
        </p>
        <ul>
          <li>
            <strong>预训练数据极其充裕。</strong>在数万亿 token 的语料上，模型对每个 token
            大约只看一次。不存在可供 dropout 预防的“背下训练集”失效模式。
          </li>
          <li>
            <strong>权重衰减覆盖了大部分相同的地盘。</strong>AdamW
            的衰减项把权重拉向零，防止任何单一特征独大。
          </li>
        </ul>
        <p>
          微调是另一回事——小而精的数据集<em>可能</em>被<em>过拟合</em>
          （模型背下那些具体样例，而不是学到可迁移的模式），dropout 在微调配方里常以非零值（通常
          0.05-0.1）重新出现。
        </p>

        <h2>读一份有代表性的配置</h2>
        <p>
          上面这些技巧的组合，把一次训练从“立刻发散”变成“终于收敛”。一份有代表性的预训练配置——泛指，不是任何特定模型公布的设置——大致长这样：
        </p>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-[12px]">
          {`optimizer:   AdamW(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
lr:          3e-4 peak, 2000 warmup steps, cosine decay to 1e-5
grad_clip:   1.0  (clip by global norm)
dropout:     0.0  (pretraining)
batch_size:  4M tokens (gradient accumulation across many devices)
seq_len:     8192
total_steps: 500,000`}
        </pre>
        <p>
          其中两个数字值得乘出来。“4M token 的 batch”是每个优化器步{' '}
          <code>512 sequences × 8,192 tokens = 4,194,304 ≈ 4M</code> 个 token。而整次训练是{' '}
          <code>4M × 500,000 steps ≈ 2 trillion tokens</code>——这就是训练一章反复念叨的那个“数万亿”。
        </p>
        <p>这块配置里不那么显然的旋钮，用大白话说：</p>
        <ul>
          <li>
            <code>beta1 = 0.9</code>、<code>beta2 = 0.95</code>——AdamW 两个滑动平均的遗忘速度。beta1 平滑梯度{' '}
            <code>g</code>；beta2 平滑梯度的平方 <code>g²</code>。值越高 = 记忆越长。
          </li>
          <li>
            <code>eps = 1e-8</code>——加在分母里的微小常数，保证当某个参数的 <code>g²</code>{' '}
            平均接近零时，步长不会除以零。
          </li>
          <li>
            <code>weight_decay = 0.1</code>——上文那股拉向零的力的强度；0.1 是典型的预训练取值。
          </li>
          <li>
            <strong>梯度累积（gradient accumulation）</strong>——先把几个小 batch
            的梯度加起来，再做一次优化器步，让少量 GPU 也能模拟出一个它们根本装不进内存的 4M token 巨型 batch。
          </li>
        </ul>
        <p>
          这里的每一行，都是针对过去十年 LLM
          训练中用血泪换来的某个具体失效模式的护栏。架构是模型本身；而这份配方，才让架构得以训成。
        </p>

        <h2>想学更多</h2>
        <p>
          本章是整门课中刻意最轻的一章——不内化这里的每个细节也能训练
          LLM，但不认识这些旋钮就读不懂研究论文。这套代码库里可训练的那一侧，见 <code>@mlx-node/trl</code>（GRPO 与
          SFT）和 <code>crates/mlx-tui</code>（<code>mlx-train</code> TUI 程序）——它们在 Apple Silicon
          上实现的是同一份配方。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

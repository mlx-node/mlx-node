// 第 13 章「训练与 teacher forcing」的简体中文译文。与英文版 ../12-training.tsx
// 平行维护：同名导出 learning 与 TrainingChapterBody；TrainingDemo（实时训练
// 练习场）保留在英文模块中渲染，因此这里不重复导出，也不引入仅供它使用的依赖。

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { BackpropFlowDiagram } from '../../widgets/BackpropFlowDiagram';
import { ExposureBias } from '../../widgets/ExposureBias';
import { GradientDescent } from '../../widgets/GradientDescent';
import { TeacherForcingAnimation } from '../../widgets/TeacherForcingAnimation';

export const learning: ChapterLearningData = {
  chapterId: 'training',
  objective: '解释 LLM 实际被训练去做什么：在因果掩码下并行地预测每个位置的下一个 token，并对照真实标签计分。',
  problem: '我们已经看过一个训练好的模型如何运行。那些权重从哪来？模型当初被优化去做什么？',
  minutes: 13,
  glossary: [
    {
      term: 'next-token prediction',
      definition: '训练任务：给定 token [t1..ti]，预测 t_{i+1}。在每条训练序列的每个位置上重复。',
    },
    {
      term: 'gradient descent',
      definition:
        '训练背后的更新规则：梯度指向上坡——损失增长最快的方向——所以每个权重朝相反方向（下坡）微调：w ← w − lr·gradient。学习率（lr）是步长：太小爬得慢，太大会越过头并发散。AdamW 对全部约 800M 个权重同时逐参数执行这条规则。',
    },
    {
      term: 'batch',
      definition:
        '一步中一起处理的若干到许多条训练序列（真实预训练的 batch 通常是数百条序列、数百万 token，靠梯度累积拼出来——见第 14 章）。损失在它们之间取平均，于是一次权重更新反映许多样例而不是一个——梯度更稳，硬件利用率更高。',
    },
    {
      term: 'cross-entropy loss',
      definition: '单次预测的损失：−log p(target)。跨位置和 batch 元素求和或取平均，得到总损失。',
    },
    {
      term: 'teacher forcing',
      definition:
        '训练时，位置 i+1 的输入是位置 i 的真实 token（而不是模型预测出的 token）。这样每个位置都有合法的真实目标，模型从正确的上下文学习，而不是从自己（最初很糟糕的）猜测学习。',
    },
    {
      term: 'one training step',
      definition:
        '对一个 batch 的序列做前向传播 → 计算损失 → 反向传播 → 优化器更新权重。在一次预训练中重复几十万到几百万次（每一步都要吃掉数百万 token，所以 token 总数会累积到数以万亿计，尽管步数并不是）。',
    },
    {
      term: 'parallel positions',
      definition:
        '多亏因果掩码，一条序列的所有位置可以在一次前向传播中同时训练——模型永远看不到未来，预测因此保持有效。',
    },
    {
      term: 'pretraining vs fine-tuning',
      definition:
        '预训练：在数万亿 token 的通用文本上做下一个 token 预测。微调：同样的目标，更小的精选数据集，通常在预训练之后进行。',
    },
    {
      term: 'AdamW',
      definition:
        'LLM 训练几乎清一色使用的优化器。它为每个参数维护梯度及其平方的滑动平均（即“矩”），让每个权重获得自适应步长；权重衰减与梯度分开施加。',
    },
    {
      term: 'backpropagation (autograd)',
      definition:
        '通过把链式法则沿前向传播倒着应用，计算标量损失对每个权重的梯度的算法。下方的练习场在 WebGPU 上运行真实的自动微分——与生产训练器用的是同一个 value_and_grad 原语。',
    },
  ],
  takeaways: [
    '整个训练目标就一行：在每条序列的每个位置上最小化 -log p(next_token)。',
    '因果掩码让训练得以并行——每个位置在一次前向传播里算出自己的损失而不偷看未来；teacher forcing 在每个位置提供真实目标，使这些并行预测都从正确的上下文中学习。',
    '不存在单独的“训练模式”架构——推理时跑的那个前向传播，训练时跑的也是它，只是改成对照真实答案计分。',
  ],
  exercise: {
    prompt:
      '在动画里，位置 5（最后一个输入 " mat"）显示 “no target”。为什么最后一个位置从不被训练？如果硬要训练它，会发生什么？',
    answer:
      '最后一个位置没有可预测的 t_{i+1}——序列已经结束了。硬要训练的话，你就是在没有真实答案的情况下猜 “mat” 后面是什么，而这恰恰是推理的设定。实践中，数据加载器把输入/目标数组错开一位，这种不对称是自动的：输入是位置 0..N-1，目标是位置 1..N，最后一个输入位置没有目标。',
  },
  quiz: [
    {
      id: 'q1-objective',
      prompt: '像 Qwen3.5 这样的 decoder-only LLM 的训练目标是什么？',
      options: [
        { id: 'a', label: '重建随机隐藏在序列各处的被掩 token（MLM）。' },
        { id: 'b', label: '在每个位置预测下一个 token，用交叉熵计分。' },
        { id: 'c', label: '把输入从英语翻译成某个固定的目标语言。' },
      ],
      correctId: 'b',
      explanation:
        'decoder-only LLM 用下一个 token 预测来训练。encoder-only 模型（BERT 类）做掩码语言建模；decoder LLM 做因果/自回归语言建模。',
    },
    {
      id: 'q2-teacher-forcing',
      prompt: '“teacher forcing” 是什么意思？',
      options: [
        {
          id: 'a',
          label: '在每个训练位置，模型的输入是真实的前一个 token——而不是模型自己本会预测出的 token。',
        },
        {
          id: 'b',
          label: '模型在一个由高质量“教师”输出组成的精选数据集上训练。',
        },
        {
          id: 'c',
          label: '一种调度技巧：学习率由一个教师网络来设定。',
        },
      ],
      correctId: 'a',
      explanation:
        'teacher forcing 让各个训练位置解耦：每个位置都从真实上下文出发，预测彼此独立。没有它，训练将是串行的、非常慢。',
    },
    {
      id: 'q3-shift-by-one',
      prompt: '对长度为 N 的训练序列，标准的输入/目标设置是什么？',
      options: [
        { id: 'a', label: '输入和目标都是完整序列；损失忽略对角线。' },
        {
          id: 'b',
          label: '输入是位置 [0..N-1]，目标是位置 [1..N]。模型的位置 i 预测 target[i]，也就是 input[i+1]。',
        },
        { id: 'c', label: '输入是前一半，目标是后一半。' },
      ],
      correctId: 'b',
      explanation:
        '简单地错开一位：每个位置预测自己的后继。最后一个输入位置没有对应的目标（超出序列末尾），通常被从损失中掩掉。',
    },
  ],
};

export function TrainingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>训练：LLM 究竟在优化什么</h1>
        <p>
          到目前为止，这门课讲的都是<em>推理</em>：一个训练好的模型接收提示词，一次吐出一个
          token。但权重从哪来？那场优化到底求解了什么？答案出奇地简单，而且对每个现代 decoder LLM
          都是同一个答案——GPT、Llama、Qwen、Gemma、Mistral，全都用同一个目标训练。
        </p>

        <h2>“训练”到底是什么意思</h2>
        <p>
          先看图景，再看公式。把模型想成一排可调的旋钮。墙上的恒温器只有一个旋钮：往上拧，房间变暖；往下拧，房间变凉。你靠微调来找到合适的设定——有点太冷，往上拨一点；拨过头了，再往回收——直到房间感觉刚好。训练就是同一个循环，只有一处不同：读房间的不是人，而是一道自动流程，它读取模型的预测错得有多离谱，然后把每个旋钮朝让预测不那么错的方向拨动一丝。没有谁手工挑选这些数值，它们是靠不断重复这次拨动而被<em>找到</em>的。
        </p>
        <p>
          难点在于旋钮的数量。给一堆散点拟合一条直线，你调的是两个旋钮——一个斜率、一个截距。沿着这架梯子往上爬，数量便爆炸：在你浏览器里运行的 Qwen3.5
          模型有 <strong>852,985,920</strong> 个旋钮，每一个都是 backpropagation 将要拨动的一个数。（要感受这架梯子能走多远：GPT-3 带着 1750
          亿个旋钮——大约多 200 倍；每个旋钮的拨动方式完全相同，只是数量多得多，这正是{' '}
          <ChapterLink chapterId="scaling">scaling</ChapterLink>
          的全部故事。）这些拨动正在压低的那个误差——整道流程读来决定每个旋钮往哪个方向拧的那一个数——正是下面的{' '}
          <code>−log p</code>。
        </p>

        <h2>一行写完的目标</h2>
        <MathDisplay
          latex={String.raw`\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(t_{i+1} \mid t_1, \dots, t_i)`}
        />
        <p>
          就这么多。<strong>对每条序列的每个位置，预测下一个 token；最小化真实下一个 token 的负对数概率。</strong>
          在一条序列内部按位置取平均（如上式所示），再在 batch
          内的各条序列上取一次平均（真实预训练的 batch 通常是数百条序列、数百万
          token——见第 14 章），这就是优化器往下压的东西。没有奖励模型，（在预训练阶段）没有人在回路里——只有海量文本和一个错开一位的目标。
        </p>
        <p>
          逐位置的损失 <code>-log p(target)</code> 叫<strong>交叉熵</strong>。当 <code>p(target)</code> 接近 1
          时损失接近 0；当 <code>p(target)</code> 很小时损失飙升。为什么偏偏取<em>对数</em>？两个理由：它把“把整条序列里每个
          token 的概率乘起来”变成“把逐 token 的代价加起来”（几千个小于 1 的数相乘会下溢成零；对数之和则表现良好）；并且由于{' '}
          <code>-log p → ∞</code>（当 <code>p → 0</code>
          ），一次信心十足的错误预测会受到任意重的惩罚——模型恰恰在又错又笃定的时候被罚得最狠。优化器的工作就是把模型的概率质量推到真正的下一个
          token 上。
        </p>
        <p>
          那损失从哪里<em>起步</em>？初始化时权重是随机的，248,320 个词表 token 几乎人人均分概率——约{' '}
          <code>1/248,320</code>。代进去：<code>−ln(1/248,320) = ln(248,320) ≈ 12.4</code>
          。这就是模型学到任何东西之前的损失，此后每一点下降都是学习。本页的实时训练练习场以同样的方式开局，只是词表是微型字符级的：默认语料只有
          3 个不同字符，曲线从约 <code>ln(3) ≈ 1.1</code> 附近出发（两个 11 字符的语料从约 <code>ln(11) ≈ 2.4</code>{' '}
          附近出发）。
        </p>

        <h2>Teacher forcing——让训练并行的技巧</h2>
        <p>
          对“预测下一个 token”的朴素理解可能是逐 token 跑模型：生成预测 1，喂回去，生成预测
          2，依此类推——正是推理的工作方式。训练时这样做慢得要命，而且更糟：模型将从自己（最初很糟糕的）预测中学习。
        </p>
        <p>
          解法是 <strong>teacher forcing（教师强制）</strong>。不把模型自己的预测喂回去当下一个输入，而是喂<em>真实的</em>{' '}
          token。teacher forcing 做的就这一件事：在每个位置提供一个合法的真实目标，让模型从正确的上下文学习，而不是从自己最初糟糕的猜测学习。
          <em>并行性</em>来自另一个机制——因果掩码（第 4 章）。由于每个位置只能向后注意，所有位置的损失可以在
          <em>单次</em>前向传播中算完，谁也偷看不到自己的目标。{' '}
          <em>真实目标（teacher forcing）+ 单遍并行（因果掩码），合起来让全部 N 个位置同时训练。</em>
        </p>

        <TeacherForcingAnimation />

        <ExposureBias />

        <h2>推理和训练是同一个前向传播</h2>
        <p>
          一个值得停下来体会的推论：不存在单独的“训练时”架构。同一个前向传播——嵌入查表、24 个 Transformer
          层（每层先混合 token：6 层靠全量注意力、18 层靠 GatedDeltaNet，再用 MLP 做变换）、顶端的 RMSNorm、LM head
          矩阵乘法——既在训练时跑，也在推理时跑。训练只是一次对整条序列做这件事，并把输出对照真实答案打分；推理则一次一个
          token 地做，并从输出中采样。
        </p>
        <p>
          这种对称性是本课程的推理套路能推广到训练的部分原因。KV 缓存（第 12
          章）是仅推理的优化，但被缓存的运算与训练时运行的运算一模一样，只是训练时没有缓存。
        </p>

        <h2>预训练 vs 微调 vs 指令/RLHF</h2>
        <p>
          上面的目标描述的是<strong>预训练</strong>：数万亿 token
          的通用网页文本、书籍和代码，模型学习在任意上下文中预测下一个 token。Qwen3.5 约 0.8B
          参数的绝大部分知识来自这里——LLM 研发的几乎全部 GPU 时也耗在这里。
        </p>
        <p>
          <strong>微调</strong>
          用同一个损失、同一个前向传播——只是换成一个更小的、精心挑选的数据集（例如指令跟随对话）。模型继续预测下一个
          token；变的是数据集。
        </p>
        <p>
          <strong>RLHF / DPO / 直接偏好</strong>
          类方法偏离了简单的交叉熵故事——它们用人类偏好或奖励模型给模型输出打分，损失不再是纯粹的下一个 token
          交叉熵。这些超出本章范围；重要的是，每一个这类后训练阶段的<em>基础行为</em>，仍然建立在这里打下的下一个 token
          交叉熵地基之上。
        </p>

        <h2>优化器实际在做什么</h2>
        <p>
          损失是一个标量。<strong>反向传播</strong>——把链式法则沿前向传播倒着算——计算它对模型每个权重的梯度（Qwen3.5-0.8B
          约有 800M 个）。下面是不需要微积分的完整算法——损失沿着堆叠一块一块往下回流：
        </p>

        <BackpropFlowDiagram />

        <p>
          这一次反向传播是<em>对架构盲视的</em>。链式法则并不在意 Qwen3.5 的 24 层里有 6 层跑 softmax attention，另外 18 层是线性递归的{' '}
          <ChapterLink chapterId="architecture">GatedDeltaNet</ChapterLink>
          ——两类层里的每个权重都从同一次下坡行走中拿到自己的梯度，这正是为什么一个 AdamW 循环就能把这两种层并排训练，而从不需要知道哪个是哪个。
        </p>

        <p>
          每个块都拿到自己的梯度后，优化器（几乎总是 <strong>AdamW</strong>，第 14
          章拆解）朝降低损失的方向迈一小步。简化到单个权重，这条规则一目了然：梯度指向上坡，所以朝反方向迈步——而这一步的大小（学习率）存在一个甜点区。
        </p>

        <GradientDescent />

        <p>
          把这个逐权重的步骤对全部约 800M
          个权重同时重复，跑上几百万到几万亿个 token。下一章讲让这个循环在这么深的堆叠上保持数值稳定的工程技巧——预热、余弦衰减、梯度裁剪、dropout。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

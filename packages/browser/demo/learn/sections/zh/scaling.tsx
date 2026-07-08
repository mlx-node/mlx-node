// 子章节 14.1（scaling laws 与 compute-optimal 训练）正文的简体中文译文，与英文版
// ../scaling.tsx 平行维护：同名导出 ScalingLawsSection。与英文版一样是纯静态阅读内容
// （SSR 安全，可被预渲染器直接渲染为静态 HTML）；翻译不改动任何数字与结论，论文
// 作者名、模型名（Kaplan、Chinchilla、Gopher、Llama 3、Qwen3.5-0.8B 等）、arXiv 编号
// 保留英文，只有周围的说明文字翻译成中文。
//
// 本章正文（chapters/13-scaling.tsx）已经有一个按参数量比较的 ScaleLadder widget，
// 以及 AdamW / 学习率调度 / 梯度裁剪的工程内容。本子章节补上缺的那根轴——同样大小的
// 模型该用多少 token 训练——不重复以上任何内容。诚实性口径与英文版完全一致：本课程
// 模型（Qwen3.5-0.8B）自己的预训练 token 数没有任何一手来源披露过。

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { TokensPerParamScatter } from '../../widgets/TokensPerParamScatter';

/** 子章节 14.1——神经网络 scaling laws 与 compute-optimal 训练。 */
export function ScalingLawsSection() {
  return (
    <Prose>
      <h1>Scaling laws：多少参数该配多少 token？</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 14 章 Scaling &amp; regularization——本章的阶梯数的是参数量；这里补上 compute
        预算的另一半：这些参数到底该看多少数据。
      </p>

      <h2>缺的那根轴</h2>
      <p>
        本章前面的参数阶梯把这个模型、GPT-2 XL、GPT-3 放在了同一条对数轴上比较<strong>参数量</strong>。这回答了「模型有多大」，但一个孤立的模型算不上一份配方——训练它还要决定<strong>用多少
        token</strong>去跑它。这第二个数字要是选错了，同样的 compute 预算下，一个更大的模型反而可能比一个训练得当的小模型<em>更差</em>。两个研究团队，相隔两年，对「给定固定
        compute 预算,该用多少 token」给出了截然相反的答案——而它们之间的这次修正，是 LLM 训练史上影响最深远的结果之一。
      </p>

      <h2>Kaplan 2020：把模型做大，别把数据做大</h2>
      <p>
        2020 年，OpenAI 的 <strong>Kaplan 等人</strong>（《Scaling Laws for Neural Language Models》，arXiv:2001.08361）跑了一大批模型的扫描实验，发现
        loss 分别对三个量各自呈现干净的幂律衰减——参数量 <code>N</code>、训练 token 数 <code>D</code>、以及 compute{' '}
        <code>C</code>：
      </p>
      <MathDisplay
        latex={String.raw`L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\quad \alpha_N \approx 0.076 \qquad\qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D},\quad \alpha_D \approx 0.095`}
      />
      <p>
        指数越大，说明 loss 对那根轴越敏感，乍一看数据（<code>0.095</code>）似乎是比参数（<code>0.076</code>）更好用的杠杆。但
        Kaplan 等人对自己的结果不是这么解读的。他们又拟合了第三条曲线——loss 对 <em>compute</em> 的关系，<code>C</code>——并追问在固定
        compute 预算下,该如何在「更大的模型」和「更多的数据」之间分配,结果发现最优分配是明显偏向一边的：
      </p>
      <MathDisplay latex={String.raw`N_{\text{optimal}} \propto C^{0.73}`} />
      <p>
        compute 每增加 <MathDisplay inline latex={String.raw`10\times`} />，模型规模就该增长{' '}
        <MathDisplay inline latex={String.raw`10^{0.73} \approx 5.4\times`} />，而数据量（以及训练步数）的增长要温和得多——剩下的指数只有
        <code>0.27</code>。他们自己对这个结论的表述是：<strong>训练非常大的模型,只用相对不多的数据,并在远未收敛前就停止训练</strong>。给定一个固定的
        compute 预算，宁可把一个大模型训练不足，也不要把一个小模型训练充分。这在近两年里都是这个领域的默认工作假设——它在很大程度上解释了为什么像 GPT-3（1750 亿参数）这样的模型,现在回头看,训练语料相对其规模来说都偏小。
      </p>

      <h2>Chinchilla 2022：让两者等比例增长</h2>
      <p>
        2022 年，DeepMind 的 <strong>Hoffmann 等人</strong>（《Training Compute-Optimal Large Language Models》，arXiv:2203.15556——正是这篇论文提出了
        <strong>Chinchilla</strong>）在远大得多的规模上重跑了这个实验：400 多个模型,参数从 7000 万到 160 亿,token 从 50 亿到
        5000 亿不等。他们对 Kaplan 结论的修正，直接引用自论文摘要：
      </p>
      <blockquote>
        「for compute-optimal training, the model size and the number of training tokens should be scaled equally:
        for every doubling of model size the number of training tokens should also be doubled.」（对
        compute-optimal 的训练而言，模型规模和训练 token 数应当等比例扩大：模型规模每翻一倍，训练 token 数也应当翻一倍。）
      </blockquote>
      <p>
        用符号写就是，两者随 compute 以<em>相同的</em>速率增长——<code>N ∝ C^0.5</code> 且 <code>D ∝ C^0.5</code>——而不是
        Kaplan 那种偏向模型的 <code>C^0.73</code> / <code>C^0.27</code> 分法。为了证明这一点，他们造出了{' '}
        <strong>Chinchilla</strong>：700 亿参数，训练了 1.4 万亿 token，用的 compute 和 DeepMind 自己更早的{' '}
        <strong>Gopher</strong> 模型完全一样——Gopher 是 2800 亿参数，却只用了 3000 亿 token。同样的 compute
        账单，分配方式却天差地别。Chinchilla 赢得干脆利落——比如 MMLU 平均分拿到 67.5%，高于参数量是它四倍的 Gopher。大不代表好；喂得饱才代表好。
      </p>
      <p>用 Chinchilla 自己的头条数字做一下乘除,一个著名的比例就出来了：</p>
      <MathDisplay latex={String.raw`\frac{1.4 \times 10^{12} \text{ tokens}}{70 \times 10^{9} \text{ params}} = 20 \text{ tokens per parameter}`} />
      <p>
        这不是论文里直接引用的一句话——它是从 Chinchilla 自己公布的 70B/1.4T 数字里推出来的比例，而这也是这个结果在业界近乎通用的引用方式：<strong>每个参数大约配
        20 个 token</strong>。作为对照，Gopher 只有 <code>300e9 / 280e9 ≈ 1.07</code> tokens/param——相对它的规模严重「吃不饱」，正是
        Kaplan 的配方所建议的做法,也正是 Chinchilla 证明是个错误的做法。
      </p>

      <h2>给本课程的模型算一算</h2>
      <p>
        这个模型有 <strong>{(852_985_920).toLocaleString('en-US')}</strong> 个参数。如果按 Chinchilla-optimal 的
        ~20:1 比例训练，token 预算会是：
      </p>
      <MathDisplay
        latex={String.raw`852{,}985{,}920 \times 20 \approx 17.06 \times 10^{9} \text{ tokens} \approx 17.06\text{B tokens}`}
      />
      <p className="text-muted-foreground">
        请仔细读这个数字：它是<strong>本节自己做的乘法</strong>，不是这个 checkpoint 已披露的事实。没有任何一手来源——不是 Qwen
        的博客,不是技术报告,也不是这个仓库自己的 config——说明 Qwen3.5-0.8B 实际预训练用了多少 token。整个 Qwen 家族里唯一能验证到的具体
        token 数字,属于<em>另一个、更早</em>的模型世代：Qwen3 技术报告（arXiv:2505.09388）说明（体量大得多、非混合架构的）Qwen3
        家族预训练大约用了 <strong>36 万亿 token</strong>。这个数字属于另一个模型,不能假设它能套用到这个 0.8B 的混合架构
        checkpoint 上——放在这里只是让你知道哪些是公开的、哪些不是。
      </p>

      <h2>真实模型故意训练过量——以 Llama 3 为例</h2>
      <p>
        Chinchilla 回答的是「在给定目标 loss 下,如何让训练 compute 最小化」。它没有回答「模型部署之后每天回答几百万次查询,如何让这笔开销最小化」。Meta 的 Llama
        3（《The Llama 3 Herd of Models》，arXiv:2407.21783）把第二个目标讲得很明白，原话是：
      </p>
      <blockquote>
        「While our scaling laws suggest our flagship model is an approximately compute-optimal size for our
        training budget, we also train our smaller models for much longer than is compute-optimal. The resulting
        models perform better than compute-optimal models at the same inference budget.」（虽然我们的 scaling
        law 显示旗舰模型在我们的训练预算下已接近 compute-optimal 的规模，但我们把更小的模型训练得远比
        compute-optimal 更久。这样得到的模型，在同样的推理预算下表现更好。）
      </blockquote>
      <p>
        对 Llama 3 8B 做同样的 tokens-per-parameter 算术——按 Meta 自己发布时的说法，它预训练用了「over 15T tokens」：
      </p>
      <MathDisplay
        latex={String.raw`\frac{15 \times 10^{12} \text{ tokens}}{8 \times 10^{9} \text{ params}} = 1{,}875 \text{ tokens per parameter}`}
      />
      <p>
        一个 8B 模型的 Chinchilla-optimal 应该是 <code>8e9 × 20 ≈ 160B tokens</code>。Llama 3 8B 用的接近它的一百倍：<strong>1,875
        / 20 ≈ 94×</strong> Chinchilla-optimal 的比例——刻意训练得远超「让训练 compute 最小」的那个点，因为一个每
        FLOP 回答质量稍好一点的小模型，在大规模<em>服务</em>时便宜太多了，比一个更大、更「compute-optimal」的模型划算得多。对比一下旗舰模型：<strong>405B</strong>
        模型用 3.8×10²⁵ FLOPs 预训练了 15.6 万亿 token，而论文<em>自己的</em> scaling-law 拟合对这个 compute 预算给出的最优点是
        402B 参数 / 16.55 万亿 token（那个规模下约 41 tokens/param）——旗舰模型几乎正好落在自己算出来的最优点上，而更小的几个兄弟模型则被故意推得远远超过了它们各自的最优点。
      </p>

      <TokensPerParamScatter />

      <h2>你浏览器里的 Qwen 遵循哪一种配方？</h2>
      <p>
        不知道——而这就是诚实的答案，不是在打太极。上面的理论是真实的，并且都从一手来源独立验证过：Kaplan 的幂律与偏向模型的分法、Chinchilla
        的等比例修正及其 ~20:1 比例、Llama 3 为了更便宜的推理而刻意训练过量。<em>无法</em>验证的是，这几种配方——或者完全是别的什么配方——究竟哪一种真正产出了这个具体的
        852,985,920 参数 checkpoint。<ChapterLink chapterId="training">第 13 章</ChapterLink>
        已经说过，这个模型在预训练时见过「trillions of tokens of generic web text, books, and code」——故意没给出精确数字，因为根本没有公开的数字。本子章节没法在
        Qwen 没公布数字的地方,不负责任地填上一个真实数字。上面那个 170.6 亿 token 的数字是一个假设值，在图表里也明确标注为假设值——它有用的地方在于让你理解「在这个规模下
        Chinchilla-optimal 意味着什么」，而不是这个模型真实训练过程的事实。
      </p>
    </Prose>
  );
}

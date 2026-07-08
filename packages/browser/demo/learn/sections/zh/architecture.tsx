// 第 16 章（完整模型，端到端）子章节正文的简体中文译文，与英文版
// ../architecture.tsx 平行维护：同名导出 ArchitectureLocalEdgeSection 与
// ArchitectureEvaluationSection。与英文版一样是纯静态阅读内容（SSR
// 安全，可被预渲染器直接渲染为静态 HTML）；翻译不改动任何数字与结论，术语
// （token、benchmark→基准、perplexity→困惑度、checkpoint、thinking/non-thinking
// 等）按本课程 zh 惯例处理。ArchitectureLocalEdgeSection 的视角是
// cloud-vs-edge 的部署权衡；ArchitectureEvaluationSection 覆盖困惑度、
// MMLU/HumanEval/GSM8K、LLM-as-judge 与污染检测，并把 Qwen3.5-0.8B 官方
// model card 里真实存在的「未公布 GSM8K/HumanEval/经典 MMLU」缺口原样呈现。

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { LocalEdgeTradeoffs } from '../../widgets/LocalEdgeTradeoffs';
import { PerplexityScale } from '../../widgets/PerplexityScale';
import { QwenBenchmarkHonesty } from '../../widgets/QwenBenchmarkHonesty';

/** 子章节 16.1——为什么要在自己的设备上跑一个小模型：cloud-vs-edge 的权衡。 */
export function ArchitectureLocalEdgeSection() {
  return (
    <Prose>
      <h1>local &amp; edge inference</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 16 章 完整模型——到底为什么要在自己的设备上跑一个小模型，以及你为此换走了什么。
      </p>
      <p>
        本章刚刚把这个模型放在了 size ladder 的接近底部：{(852_985_920).toLocaleString('en-US')}{' '}
        个参数，远低于多数人说「large language model」时所指的那些 frontier 系统。这就引出一个合理的问题——既然一个托管 API
        能伺候比它大 100× 的模型，<em>为什么还要在自己的设备上跑一个小的？</em>答案是：「小」和「在你自己的设备上」能换来数据中心给不了的东西，而这个选择是一笔两边都有赢面的实打实的
        trade。
      </p>

      <h2>把这笔 trade 讲具体</h2>
      <p>
        在本地运行——也就是 <strong>在 edge 上</strong>，在用户手里已经拿着的那台设备上——在书里点名的四件事上取胜：
        <strong>latency</strong>（没有网络往返，省下几十到几百毫秒）、<strong>independence</strong>（无 server、无宕机，可离线）、
        <strong>privacy</strong>（数据永不离开设备）以及 <strong>cost</strong>（对开发者免费——是用户自己的硬件在干活）。它为这些付出的代价是能力与速度：edge
        硬件只有 datacenter GPU 的一小部分，会因发热降频，hardware×software 的组合很碎片化，还会耗电。没有哪一侧赢下每一行——这正是
        trade：
      </p>

      <LocalEdgeTradeoffs />

      <h2>哪个大小能跑在哪里</h2>
      <p>
        模型大小决定了下限。手机超过一两 billion 参数就吃力——而这正是这个模型舒舒服服待着的地方。一台笔电能以{' '}
        <code>bf16</code> 跑一个 8B；要把 100B+ 的模型搬上高端桌面，就得用之前某个<a href="/zh/chapters/kv-cache/quantization">子章节</a>
        的那根杠杆——<strong>quantization</strong> 正是把大模型塞进小内存的桥梁（通过 Ollama / llama.cpp），而 Mixture-of-Experts
        也有帮助：每个 token 只激活其中一小片参数。再往上，一个几千亿到万亿参数的 frontier 模型就只能住在 cloud 上了；低端笔电和
        Chromebook 根本跑不动任何有意义的本地模型。
      </p>
      <p>
        而硬件本身也编码着一笔 trade。一块游戏 GPU 和一台工作站可以价位相近，却为相反的目标优化——<strong>capacity 对 speed</strong>
        ——这就是上面 widget 里的第二张表：一块 32&nbsp;GB、bandwidth 巨大的卡，对上一台 512&nbsp;GB unified-memory
        的机器——后者能装下大得多的模型，却把它们搬得更慢。
      </p>
      <p className="text-muted-foreground">
        你在这里。这个浏览器标签页<em>就是</em> edge inference：Qwen3.5-0.8B、<code>bf16</code>、batch 1，跑在 WebGPU
        上——没有 server，没有 API key，不上传任何东西。它为你换来 privacy、zero network latency 和 $0
        成本，代价是能力（一个 0.8B 模型，而非 frontier）与速度（你的 GPU，而非 H100）。inference 的未来不是 local
        <em>或</em> cloud，而是 <strong>两者皆是</strong>——小模型与快查询跑在设备上，更重的工作负载交给数据中心——而这整门课，正是那个小模型、在设备上的那一半，活生生地跑在你眼前。
      </p>
    </Prose>
  );
}

/** 子章节 16.2——困惑度、基准、LLM-as-judge 与污染检测：模型跑完了，怎么给它的好坏打一个数，以及该多信任这个数。 */
export function ArchitectureEvaluationSection() {
  return (
    <Prose>
      <h1>评测：困惑度、基准，以及该多大程度上信任它们</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 16 章 完整模型——模型已经跑完了，怎么给它的好坏打一个数？
      </p>
      <p>
        <ChapterLink chapterId="training">第 13 章</ChapterLink>从 <code>-log p(target)</code>{' '}
        构建出了训练目标：模型的损失，就是它赋给真实下一个 token 的负对数概率，在所有位置上取平均。这个数字从没离开过训练场——它，不取指数的话，也是评测<em>唯一</em>需要的原料。<strong>困惑度（perplexity）</strong>
        就是这同一个损失，取个指数：
      </p>
      <MathDisplay
        latex={String.raw`\text{PPL}(X) = \exp\!\left(-\frac{1}{t}\sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})\right) = \exp(\text{cross-entropy loss})`}
      />
      <p>
        Hugging Face Transformers 的官方文档把它讲得很直白：这就是「数据与模型预测之间交叉熵的指数」。它有一种干净的读法：困惑度就是——如果模型是在这么多个等可能选项里<em>均匀</em>乱猜，它平均会有多困惑，那个「这么多」就是困惑度这个数。两个选项里满怀信心地猜对，困惑度是
        1；在整个词表上彻底迷失，困惑度就逼近词表本身的大小。
      </p>

      <h2>这个具体模型自己的数字，取个指数</h2>
      <p>
        第 13 章已经算过一次：初始化时，这个模型 248,320 个词表 token 里的每一个都大致拿到相同的概率，所以损失从{' '}
        <code>−ln(1/248,320) = ln(248,320) ≈ 12.4</code> nats 起步。把它取个指数，正好还原出{' '}
        <strong>248,320</strong>——也就是词表大小本身。这不是巧合：对 <code>V</code> 个结果做完全均匀的猜测，困惑度恒等于{' '}
        <code>V</code>，因为均匀分布上的交叉熵是 <code>ln V</code>，而 <code>exp(ln V) = V</code>。所以在迈出第一步梯度之前，这个具体模型在自己词表上的困惑度就是{' '}
        <strong>≈ 248,320</strong>——训练中损失每降一个 nat，这个数字就跟着缩水。两个真实、已发表的锚点，标出了一个训练充分的模型能落到哪里：GPT-3
        175B 在 Penn Treebank 上把困惑度做到了 <strong>20.50</strong>（此前 GPT-2 时代的最好成绩是 35.8），在 LAMBADA
        下一词预测上，few-shot 提示下做到了 <strong>1.92</strong>（此前最好成绩是 8.63）——两个数字都来自 GPT-3 原始论文。GPT-2
        第二大的模型（762M 参数——1.5B 才是最大的那个），在 WikiText-2 上报告的困惑度是 <strong>19.93</strong>。下面这整条阶梯用的都是同一个公式——从这个模型初始化时的
        ≈248,320，到 GPT-3 的 1.92，跨越了五个数量级：
      </p>

      <PerplexityScale />

      <h2>标准基准：固定的问题，固定的答案</h2>
      <p>
        困惑度算起来便宜，也不需要设计任何专门的留出任务，但它只衡量模型对<em>通用</em>文本的预测有多准——不能告诉你模型能不能答对一道知识题、写出能跑的代码、或者解出一道数学题。为此，这个领域靠的是<strong>基准（benchmark）</strong>：一套固定的问题，配一套固定、可核对的答案。
      </p>
      <p>
        <strong>MMLU</strong>（Massive Multitask Language Understanding，Hendrycks 等人，2020）是 57 个中学到专业水平的学科——历史、法律、医学、物理等等——打包成
        15,908 道四选一的选择题。评分就是普通的选择题准确率，对比 25% 的随机猜测下限；GPT-3 175B 在 few-shot 下拿到
        43.9%，而更小的 GPT-3 尺寸几乎只是压线超过随机水平（24.9%–26.0%）。
      </p>
      <p>
        <strong>HumanEval</strong>（Chen 等人，2021，也就是提出 OpenAI Codex 的那篇论文）是 164 道手写的 Python
        题目：给一个函数签名和一段 docstring，补全函数体，再跑真实的单元测试（平均每题 7.7 个测试）。因为采样本身是随机的，模型会拿到
        <code>k</code> 次尝试，指标就是 <strong>pass@k</strong>——<code>k</code> 个样本里<em>至少有一次</em>命中的问题占比，用一个无偏估计量算出来，这样就不会单纯因为多采样几次而占便宜：
      </p>
      <MathDisplay
        latex={String.raw`\text{pass@}k := \mathbb{E}_{\text{Problems}}\left[\,1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right]`}
      />
      <p>
        其中每道题采样 <code>n</code> 次，<code>c</code> 次通过全部测试。Codex 做到了 pass@1 = 28.8%、pass@100 =
        70.2%——而没有代码微调的普通 GPT-3，得分是 0%。
      </p>
      <p>
        <strong>GSM8K</strong>（Cobbe 等人，2021）是 8.5K 道小学水平的应用题（7.5K 用于训练，留出 1K
        用于测试）——「Natalia 在四月卖出玩偶给 48 个朋友，五月卖出的数量是四月的一半……」——评分靠<strong>精确匹配</strong>
        最终的数字答案，不是 pass@k：没有代码要跑，只有一个数字要核对。论文的核心结果讲的是验证器，而不是单纯拼规模：一个微调来解题的
        6B 模型，配上另一个训练来给 100 个采样候选解重新排序打分的 6B 验证器模型，其表现略微超过了单样本、没有验证器的微调
        175B 模型——论文把这个提升描述为「大致相当于把模型规模扩大 30 倍」。
      </p>

      <h2>LLM-as-judge：当没有唯一正确答案时</h2>
      <p>
        上面这些方法都搞不定开放式输出——「帮我写一句生日祝福」根本没有一个固定的正确字符串可以对比。Zheng
        等人（2023）提出了如今业界普遍采用的解法：找一个强 LLM（他们用的是 GPT-4）去读两份候选回复，像人类评审一样<strong>判断</strong>哪个更好。他们提出的
        MT-Bench（一套固定的多轮问题集）和 Chatbot Arena（众包的正面对决）正是建立在这个想法上，而 GPT-4 作为裁判达到了{' '}
        <strong>&gt;80%</strong> 与人类偏好的一致率——和两个人类彼此之间的一致程度相当。
      </p>
      <p>
        这篇论文同样毫不含糊地指出了 LLM 裁判会在哪里出错，点名了三种具体的 bias：<strong>position bias</strong>
        （偏向无论谁先展示的那个回复）、<strong>verbosity bias</strong>（偏向更长的回答，不管它是不是真的更好），以及{' '}
        <strong>self-enhancement bias</strong>（裁判偏向那些读起来像自己「家族」风格的回答）。LLM-as-judge
        之所以强大，正是因为它能覆盖固定答案基准碰不到的开放式文本——但作为换来这份灵活性的代价，它也继承了那个裁判模型自己的盲点。
      </p>

      <h2>污染：为什么这些数字都不能照单全收</h2>
      <p>
        上面每一个基准都是互联网上的公开文本，而每一个现代模型都是在互联网的海量抓取上预训练出来的。如果一个基准的题目——更糟的话，连答案——泄漏进了预训练数据，模型完全可以靠<em>背下了这套题</em>拿高分，而不是真的推理出了答案。这就是<strong>污染（contamination）</strong>，而抓污染的手段一直在变严格。
      </p>
      <p>
        GPT-3 自己的论文用的是相对宽松的检测：评测样本和训练抓取数据之间的 <strong>13-gram 重叠</strong>。用这个方法，它把
        QuAC、SQuADv2、DROP 里超过 90% 的样本都标记为受污染——这个数字相当惊人，说明只要网页抓取规模够大，公开基准的文本会渗透进去多深。GPT-4
        的技术报告把门槛收紧成了 <strong>50 字符子串检测</strong>：先去掉评测文本和训练文本里的空格与符号，再看评测样本里随机抽的三段
        50 字符片段，有没有原样出现在训练数据里。这个更严的检测确实找到了真正的泄漏——OpenAI 把受污染题目去掉后重新给考试打分，还整个丢弃了一项基准
        BIG-bench，因为在其中确认发现了污染。更新的一种方法干脆绕开训练语料本身：<strong>Min-K% Prob</strong>（Shi
        等人，2023）只看模型自己——如果一段序列里那些「最不该被猜中」的
        token，模型给出的概率却高得可疑，就标记它为很可能是被记住的：一段真正没见过的文本里，理应有几个真正让模型意外的
        token，而被记住的文本通常没有。
      </p>
      <p className="text-muted-foreground">
        所以读这个子章节里的每一个基准数字，以及你在任何地方看到的每一个基准数字，都请带上这个星号：它是一个真实的、可核查的测试分数——同时也是一个可能在某种程度上是在衡量记忆、而不是能力的数字，具体到什么程度，训练这个模型的实验室之外没人能完整审计。
      </p>

      <h2>你浏览器里的 Qwen 有基准数字吗？</h2>
      <p>
        部分有。Qwen3.5-0.8B 官方的 Hugging Face model card 公布了一整族知识与推理基准上的真实、带日期的分数——其中包括
        MMLU-Pro、MMLU-Redux、C-Eval、SuperGPQA、GPQA 和 IFEval——大多数都报了两遍（一遍是模型默认的{' '}
        <strong>non-thinking</strong> 模式，一遍是 <strong>thinking</strong> 模式，也就是训后章节里那个{' '}
        <ChapterLink chapterId="post-training">&lt;think&gt; 块</ChapterLink>），唯一的例外是 GPQA，卡片只报告了
        thinking 模式下的分数。它<em>没有</em>公布的——对这个具体的 0.8B
        checkpoint 来说——是一个 GSM8K 分数、一个 HumanEval 分数、或者一个经典（非 Pro、非 Redux）的 MMLU 分数——直接搜索这三项，也都没能找到任何独立的官方数字。这是卡片自己给出的表格，缺口原样保留：
      </p>

      <QwenBenchmarkHonesty />

      <p className="text-muted-foreground">
        这个缺口就是诚实的答案，本课程不打算用一个编出来的数字把它抹平。不是每个 checkpoint 都会跑遍每一个基准——同一个系列里更小、发布更快的模型，经常只在旗舰版跑过的一个较窄切片上被评测过——报告「没有测过」，比报告一个没人核实过的数字，更值得信任。
      </p>
    </Prose>
  );
}

// 子章节 11.1（多 token 预测与自推测式解码）正文的简体中文译文，与英文版
// ../sampling.tsx 平行维护：同名导出 SamplingMtpSection。
// 与英文版一样是纯静态阅读内容（SSR 安全，可被预渲染器直接渲染为静态 HTML）；
// 每条架构断言都以真实的 Qwen3.5 MTP 头为依据，翻译不改动任何数字与结论。

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { ConstrainedDecodeMask } from '../../widgets/ConstrainedDecodeMask';
import { MtpLineageDiagram } from '../../widgets/MtpLineageDiagram';
import { MtpModuleDiagram } from '../../widgets/MtpModuleDiagram';
import { SelfConsistencyVoting } from '../../widgets/SelfConsistencyVoting';
import { SpecDecodeFlamechart } from '../../widgets/SpecDecodeFlamechart';
import { SpecDecodeVariantsDiagram } from '../../widgets/SpecDecodeVariantsDiagram';
import { SpeculativeDecodeDiagram } from '../../widgets/SpeculativeDecodeDiagram';
import { SpeculativeVerifyLive } from '../../widgets/SpeculativeVerifyLive';

/** 子章节 11.1——多 token 预测，以及由它驱动的推测式解码。 */
export function SamplingMtpSection() {
  return (
    <Prose>
      <h1>多 token 预测</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 11 章 Sampling——模型如何在<em>一次</em>前向中掏出<em>不止一个 token</em>。
      </p>
      <p>
        上一章讲的 sampling，把一行 logits 变成一个 token。要产出下一个 token，你得把它接回提示词，再把
        <em>整个模型重新跑一遍</em>——全部 24 层、每一个权重。一段 200 token 的回复就是 200 次完整前向，严格一个接一个，因为每个
        token 都依赖前一个。这就是<strong>自回归税</strong>，也是生成感觉慢的最大单一原因。
      </p>
      <p>
        让它真正难受的，是每一次前向<em>把时间花在哪</em>。产出一个 token 会把模型里每个权重恰好碰一次，所以一个解码步是
        <strong>访存受限</strong>的：GPU 这一步的大部分时间只是在把那十亿来个参数从内存里读出来，而那一个 token
        实际要做的数学，相比之下只是个零头。芯片大多在闲着、等内存。于是有一个诱人的问题：既然我们反正要付出把这些权重全都搬进来的代价，能不能为这同一笔钱拿到
        <em>不止一个 token</em>？
      </p>

      <h2>这个赌注：先 draft，再 verify</h2>
      <p>
        <strong>推测式解码（speculative decoding）</strong>就是那个说「能」的把戏。思路是先廉价地<em>猜</em>出接下来几个
        token，再让真正的模型在一次前向里<em>把所有猜测一次性查完</em>。查 <code>k</code> 个猜测，开销和正常产出一个 token
        差不多——都是过一遍权重——所以每个活下来的猜测，都是你白赚的一个 token。猜得好，一次前向就吐出好几个 token；猜得差，你也只是浪费了一点点廉价的草稿，仅此而已。
      </p>
      <p>
        传统上你需要<em>第二个、更小的</em>「draft 模型」来做猜测——一个额外的模型，要训练、要加载、还要保持同步。
        <strong>多 token 预测</strong>（multi-token prediction，MTP）去掉了这笔成本：模型为<em>它自己</em>打草稿。Qwen3.5
        随身带着一个小小的额外模块——MTP 头——它唯一的活，就是在主模型还在收尾当前 token 时，先提议<em>下一个</em> token。
      </p>

      <h2>Qwen3.5 随身带的 MTP 头</h2>
      <p>
        它刻意做得很小，因为几乎所有东西都从主模型那里借。它接收<em>两个</em>输入：主模型在当前位置的<strong>最后一个
        hidden state</strong>，以及<strong>刚刚吐出的那个 token 的 embedding</strong>——关键在于，这个 embedding
        来自主模型用的<em>同一个</em> embedding 矩阵（<code>mtp_use_dedicated_embeddings: false</code>）。它把两者都归一化，用一次投影把它们融合，过<em>一层</em>
        transformer，再通过主模型用的<em>同一个 tied LM head</em> 读出一个 token。整个头就这么多：
      </p>

      <MtpModuleDiagram />

      <p>
        有两个细节对后面很重要。第一，这个头的那一层是<strong>全量注意力</strong>层——不是 18 个 GatedDeltaNet
        层里的某一层——而且它自带一个小小的注意力 cache；它从不碰主模型的循环 state。第二，因为 embedding 和输出头是
        <em>共享</em>的，唯一的新权重就是三个 RMSNorm、一次投影，以及那单独一层。在配置里这就是一行——
        <code>mtp_num_hidden_layers: 1</code>——一个预测<em>下一个之外再多一个</em> token 的单一模块。
      </p>

      <h2>这个循环：draft <MathDisplay inline latex={String.raw`D`} /> 个、verify 一次、接受匹配的</h2>
      <p>
        现在让这个头干活。一个解码步变成三个动作。<strong>Draft（打草稿）：</strong>廉价的 MTP 头一个接一个地提议
        <code>D</code> 个候选 token（Qwen3.5 的默认深度是 <code>D = 1</code>，不过引擎可以把它调高）。<strong>Verify（核验）：</strong>完整模型把窗口
        <code>[最后提交的 token, d₁, …, d_D]</code> 在<em>一次</em>批处理前向里跑完，一口气在每个位置都给出它自己的意见。
        <strong>Accept（接受）：</strong>从左到右走过草稿，保留每一个与完整模型本会选出的相符的猜测；在第一个不匹配处，改吐出一个修正
        token，然后停下——在 temperature 0 时，这个修正就是完整模型的 top token；而带采样时，它是从一个由两个模型的概率共同构建的
        调整后分布中抽取的，而不是简单地取 target 自己选定的那个 token。一步步看：
      </p>

      <SpeculativeDecodeDiagram />

      <p>
        算一算收益。如果 <code>D</code> 个草稿里有 <code>K</code> 个被接受，这一步就吐出 <code>K + 1</code> 个 token——
        <code>K</code> 个好猜测，加上 verify 这趟白赚的一个 token（第一处失误处的修正，或者当<em>所有</em>草稿都对时、末尾再多送的一个 token）。所以过一遍权重，能产出
        1 个 token（全被拒——和朴素解码跑同样一趟完整模型前向，只是多花了草稿那一点点开销）一直到 <code>D + 1</code> 个之间的任意数量。
      </p>

      <h2>为什么它恰好是无损的</h2>
      <p>
        下面这部分，正是让推测式解码诚实、而非以质量换速度的关键：在 temperature 0 时，一个草稿<em>只有</em>在它与完整模型自己在那个位置本会选出的顶选相符时，才会被保留。带采样时，接受判据换了一套，但同样严格——它比较
        target 和 draft 对那个具体草稿 token 各自给出的概率，并以概率{' '}
        <MathDisplay inline latex={String.raw`\min(1, p_{\text{target}}/p_{\text{draft}})`} /> 接受，被拒时则从一个修正性的残差分布里重新抽样。不管哪种情况，verify
        这趟用的都是真模型的真分布；MTP
        头永远说不了最后一句话。配上恰当的接受规则——Leviathan 等人提出、Chen 等人独立提出的推测式采样/解码算法，两者都发表于 2023 年——你吐出的 token 流
        <strong>在分布上与朴素的逐个解码完全一致</strong>——每一步的概率、以及 temperature 行为都相同（在 temperature 0 时就是完全相同的文本；带采样时则是完全相同的概率），只是用更少的前向产出。MTP
        头永远改不了答案——最坏只是付一点点草稿税，最好则能省下好几趟前向。
      </p>
      <p className="text-muted-foreground">
        这也是为什么猜错很便宜：一个被拒的草稿，只花掉产出它的那一点点 MTP 计算。昂贵的 verify
        那趟无论如何都要发生——它就是朴素解码本会跑来产出那一个 token 的同一次前向。推测式解码，最坏不过是带一点点草稿税的朴素解码；最好则是好几倍的速度。
      </p>

      <h2>这个想法从哪来</h2>
      <p>
        MTP 不是一出场就是成品。我们走到这里用了三步，Qwen3.5 所采用的是第三步的设计：
      </p>

      <MtpLineageDiagram />

      <p>
        Meta 2024 年的版本在一个共享主干上栓了<em>四个并行的头</em>，每个同时预测一个不同的未来 token——作为训练信号很棒，但这些头彼此独立，所以没法互相作为条件。DeepSeek-V3
        把形状改成了<em>单个串行模块</em>，它预测多出来的一个 token，同时<em>保住因果链</em>——预测 <em>t+2</em>
        这个 token 的草稿，能看到已经选定的 <em>t+1</em>。Qwen3.5（建立在 Qwen3-Next 架构之上）采用的正是这种串行、权重共享的设计：一个模块，深度为一。
      </p>

      <h2>它把自己的饭钱挣了两次</h2>
      <p>
        同一个头在两个截然不同的时刻都有回报。在<strong>训练</strong>时，要求每个位置去预测接下来<em>两个</em> token
        而不是一个，是一个更稠密的学习信号——每一个 token 的数据里都有更多可学的东西。Meta 报告说他们的多 token 模型在 13B
        规模下多解出了 <strong>12%</strong> 的 HumanEval 和 <strong>17%</strong> 的 MBPP 题；DeepSeek-V3 把一个 MTP
        损失折进预训练，其权重从 <MathDisplay inline latex={String.raw`\lambda = 0.3`} /> 起步，在最后一段 token 上降到
        <MathDisplay inline latex={String.raw`\lambda = 0.1`} />：
      </p>
      <MathDisplay latex={String.raw`\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}`} />
      <p>
        Qwen 把自己的 MTP 描述为<em>同时</em>提升预训练效率和推理速度。在<strong>推理</strong>时，同样的这些权重就变成了上面那个循环里白送的自我 draft 者。一个模块，两份工作：一个训练得更好的模型，外加一个更快的模型。
      </p>

      <h2>时间都花在了哪里</h2>
      <p>
        前面我们说过，解码步是访存受限的——芯片大多在闲着、等权重。只有亲眼<em>看到</em>这一点，speculative
        decoding 才真正讲得通。下面就是 GPU profiler 会画出来的那幅图：同一条时钟上的两条时间线，上面是朴素解码，下面是
        draft → verify → accept。每次前向画成 ≈20 ms 宽，因为这大致就是这个标签页自己的朴素 decode
        实测出来的数字（M 系列 Mac 上约 50 tok/s）——而 verify 前向和朴素前向<em>一样宽</em>，因为不管哪种，都只是把权重过一遍。把
        draft 深度调高、把文本换得更难猜，看看 trace 会发生什么：
      </p>

      <SpecDecodeFlamechart />

      <p>
        有两件事值得注意。第一，<code>D</code> 的收益是递减的：第四个 draft 只有在前三个全部存活时才算数，所以每多猜一个，价值都比上一个低。第二，可预测性卡住一切——在难猜的文本上，speculative
        那条会塌回朴素解码外加 ≈1 ms 的草稿税，draft 得再深也没用。再看看红色小格对时间线做了什么：什么也没做。被拒的位置从不会把一次前向拉宽——浪费搭的是一趟你反正要付钱的访存。这也正是为什么这个把戏在
        batch = 1 时划算、在打满的服务器上会被关掉：白车只有在算力那一行还空着的时候才存在。
      </p>

      <h2>一座 drafter 动物园</h2>
      <p>
        上面所有内容都把 MTP 头当作那份廉价 draft 的来源。但 <strong>draft → verify → accept</strong>
        这副骨架并不在乎猜测<em>从哪来</em>——它只在乎猜测够廉价、以及 verify
        那趟守住无损保证。换掉 drafter，你就得到一整个 <strong>speculative decoding</strong>
        家族，它们全都共享你刚刚走过的那同一趟 verify。
      </p>
      <p>
        有 <strong>draft-target</strong> 方案（由第二个更小的模型来写 draft——最容易加装，但你现在要跑两个模型）；
        <strong>Medusa</strong>（在模型自身上长出几个额外的预测 head，不需要独立模型）；<strong>EAGLE</strong>
        （一个专门打造、读取 target hidden states 的微型模型——通用对话的首选）；以及 <strong>n-gram / lookahead</strong>
        解码（完全不用 draft model——只是查一下通常接什么，在代码和可预测语法上收益巨大）。你刚认识的 MTP 头，就是
        Qwen3.5 随身带的那一种。挑一个 drafter，比一比什么变了、什么没变：
      </p>

      <SpecDecodeVariantsDiagram />

      <p>
        有三条规则对整座动物园都成立。它们<strong>全部无损</strong>——在 temperature 0 时，verify 只有在 draft
        与真实模型自己的顶选一致时才保留；带采样时，它跑的是上面那同一套概率比接受/重采样判据。所以每一种变体只改变<em>速度，绝不改变答案</em>。它们都在
        <strong>低 batch</strong> 时收益最大，此时 GPU 有富余算力去白跑那趟 verify；在高 batch
        时芯片已经被占满（<a href="/zh/chapters/kv-cache/batching">batching</a> 子章节会讲为什么），于是服务器会把推测
        <em>关掉</em>。而 <strong>temperature</strong> 越高，下一个 token 越难预测，接受率就越低。空闲且文本可预测时，推测为你买来速度；反之，它会安静地让到一边。
      </p>

      <h2>看真实模型做 verify——就在一次前向里</h2>
      <p>
        上面的一切都是示意——预设的 token、写好的剧本。下面这个是<strong>现场版</strong>：你可以把真正的 Qwen3.5-0.8B
        加载进这个标签页，让它在<em>一次真实的前向</em>里 verify 一份廉价的 draft，每个判定都来自模型自己的
        logits。这份 draft 来自 <strong>prompt-lookup</strong>——就是上面动物园里那个 n-gram
        drafter，这里是真实现的：拿前缀末尾的几个 token
        去前缀更早的位置查一查，当时紧跟其后的内容就成了猜测——或者干脆你自己输入一段续写。它<em>不是</em>来自 Qwen 的
        MTP head：这份 checkpoint 里没有任何 <code>mtp.*</code>{' '}
        tensor，下一节会讲清楚。还有一句诚实声明：前缀是以原始文本喂给模型的，不套 chat
        template——这是一次原始语言模型续写，不是一轮对话。
      </p>

      <SpeculativeVerifyLive />

      <p>
        有三点值得注意。这里的 verify 是 greedy、temperature 0——一个 draft token 被接受，<em>当且仅当</em>
        它就是模型在那个位置自己的 argmax，没有任何放水。被拒时，接着往下写的是模型自己的选择（绿色的纠正
        chip）——这正是推测式解码无损的原因：输出就是朴素解码本来也会产出的东西。而这整个检查——每个位置、以及上面每一份可以展开的
        top-K 分布——只花了<strong>一次</strong>
        过权重的前向。这一次前向，就是这个子章节一直在讲的全部速度故事。（这个小部件是专为演示接出来的一次性 verify
        探针——浏览器实际的聊天解码循环仍是朴素自回归，下一节会说明。）
      </p>

      <h2>你浏览器里的 Qwen 用它吗？</h2>
      <p>
        没有——而且照例，诚实的答案才是有意思的那个。架构是真的：这个具体模型的配置写着
        <code>mtp_num_hidden_layers: 1</code>，所以一份完整的 Qwen3.5 checkpoint <em>确实</em>定义了你刚看到的那个头。但有两件事让它在这个演示里没上路。浏览器下载的那份
        bf16 checkpoint 在转换时没带上 MTP 权重——它的 474 个 tensor 里，<em>零个</em>是 <code>mtp.*</code>——而这里的
        WebGPU 解码循环是朴素自回归的，一次前向一个 token，完全没有 draft/verify 那套机制。（你不孤单：原版的 Hugging Face
        Transformers 也会丢掉 MTP 权重；如今主要是 vLLM、SGLang，以及本项目自己的原生 Metal 引擎这类推理引擎，才真的去跑那个循环——在那里实测大约快
        <strong>1.06–1.46×</strong>，取决于文本有多好预测。）
      </p>
      <p>
        所以多 token 预测，和视觉编码器以及完整 checkpoint 里其他「架构有定义、但本次浏览器之旅不会走到」的部分坐在一起：在 Qwen3.5 里是真的，在这里是有意关掉的。你甚至能在
        <a href="/zh/chapters/architecture">架构图</a>里看到它被置灰——那个「MTP head」节点，就在一个 token 在本演示里真正会走的路线之外。
      </p>
    </Prose>
  );
}

/** 子章节 11.2——推理、自洽性，以及测试时计算。 */
export function SamplingReasoningSection() {
  return (
    <Prose>
      <h1>推理与测试时计算</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 11 章 Sampling——<ChapterLink chapterId="post-training">第 15 章</ChapterLink>已经带你看过{' '}
        <code>{'<think>'}</code> 这颗药丸；这里要讲的是，为什么让模型在回答前「想得更久」这件事本身会奏效，以及它怎么用到了本章自己讲过的那些
        sampling 旋钮。
      </p>
      <p>
        一次性给出答案的模型，不管问题有多难，每个 token 花的计算量都是固定的。<strong>思维链（chain-of-thought，CoT）</strong>提示——Wei
        等人，2022 年——发现了一件近乎令人尴尬地简单的事：给一个权重冻结的模型看几个把中间步骤写出来的样例，它就会开始写出<em>自己的</em>
        中间步骤，在算术、逻辑和多步应用题上表现明显更好。没有改动任何权重。一条 8 个样例的 CoT
        提示，就能把一个 540B 的 PaLM 模型推上 GSM8K 数学题的新 state-of-the-art——甚至超过了外挂计算器/校验器、专门微调过的
        GPT-3。
      </p>

      <h2>一条链不够——多条链投票</h2>
      <p>
        CoT 给你的是一条推理链。<strong>自洽性（self-consistency）</strong>（Wang 等人，2022 年）问了一个更尖锐的问题：为什么只信一条？在
        temperature <MathDisplay inline latex={String.raw`T > 0`} />——就是本章前面讲过的那个 sampling
        temperature——下独立采样 <MathDisplay inline latex={String.raw`n`} /> 条链，让每条链各自得出自己的最终答案，再报出多数链都同意的那个答案：
      </p>
      <MathDisplay latex={String.raw`\hat{y} = \operatorname*{arg\,max}_{a} \sum_{i=1}^{n} \mathbb{1}[\,y_i = a\,]`} />
      <p>
        这是本章自己那套机制的一次直接、具体的运用：在 <MathDisplay inline latex={String.raw`T = 0`} />（greedy）时，每一次「重新采样」都是同一条链，没什么可投的——自洽性<em>需要</em>
        temperature sampling 提供的那份随机性，才能一开始就产生真正不同的链。Wang 等人测到了实打实的收益：GSM8K 上 +17.9
        个百分点，SVAMP +11.0，AQuA +12.2，StrategyQA +6.4，ARC-challenge +3.9，全部相对于 greedy
        思维链而言。看它如何从一批混杂的样本里把正确答案救回来——其中还有一个样本，犯了和 greedy 解码本会犯下的一模一样的错误：
      </p>

      <SelfConsistencyVoting />

      <h2>从一个 prompt 技巧，到一根被训练出来、可扩展的坐标轴</h2>
      <p>
        第 14 章的尺寸阶梯，画的是准确率相对<em>参数量</em>——模型越大、训练时算力越多，分数越高。2024 年 9 月，OpenAI 的 o1
        在这张图上加了第二根轴：<strong>测试时计算（test-time compute）</strong>。o1 用大规模强化学习训练，让它在回答前先产出一段很长的内部思维链——按输出
        token 计费，但对用户隐藏——它的准确率随着<em>训练时</em>更多的 RL、以及<em>推理时</em>更多的思考 token
        都呈对数线性增长。跳跃幅度很大：在 AIME 2024 上，GPT-4o 大约 12%，而 o1 单样本 74%、64 个样本做类似自洽性的共识投票达到
        83%、对 1,000 个样本重排序达到 93%——外加 Codeforces 89th 百分位，以及超过博士水平的 GPQA 科学题成绩。Snell
        等人（2024 年）把「为什么这会奏效」形式化了：测试时计算是一根独立的、随 prompt 自适应的扩展轴，一种<em>计算最优</em>的花法，能在小模型力所能及的问题上，打平一个大{' '}
        <MathDisplay inline latex={String.raw`14\times`} /> 的模型，比朴素的 best-of-<MathDisplay inline latex={String.raw`N`} />{' '}
        采样在计算效率上高出 4 倍以上。
      </p>

      <h2>纯 RL 炼出的推理：DeepSeek-R1</h2>
      <p>
        DeepSeek-R1（2025 年 1 月）把训练那一侧又往前推了一步：DeepSeek-R1-Zero 在 DeepSeek-V3-Base 之上，靠<strong>纯强化学习</strong>——
        <em>零</em>监督热启动，只用 GRPO（Group Relative Policy Optimization）和纯规则的正确性/格式奖励，完全没有学出来的
        reward model——学会了推理：长链、自我反思、检查自己的工作。在 AIME 2024 上，pass@1 从 15.6%（基础模型）经 RL
        之后升到 71.0%——再配上你刚才见过的那种自洽性式的、对 64 个样本的多数投票，升到 86.7%，追平 OpenAI 的
        o1-0912。正式发布的 DeepSeek-R1 在 RL 之前加了一小份冷启动监督数据集，纯粹是为了修好可读性和语言混杂问题，最终达到
        79.8%，与 o1 打平。
      </p>

      <h2>省钱的做法：<code>s1</code> 与 budget forcing</h2>
      <p>
        不需要 R1 那种规模的 RL，也能炼出一条形状相似的曲线。<code>s1</code> 论文（2025 年 1 月）只在 1,000 条精心挑选的样例（s1K）上对
        Qwen2.5-32B-Instruct 做微调，再加上<strong>budget forcing</strong>：提前截断模型的思考、强迫它给出答案；或者反过来——如果它想过早停下，就压住那个「思考结束」token，直接在后面接上字面的一个词「Wait」，逼它继续推理。仅这一根杠杆，就把同一个模型在
        AIME24 上的准确率从 50% 拉到了 57%，最终的 s1-32B 在竞赛数学上比 o1-preview 高出多达 27 分，完全不需要
        RL。注意 budget forcing 在 token 层面到底在做什么：直接编辑<ChapterLink chapterId="post-training">第 15 章</ChapterLink>
        给你看过的、逐 token 吐出来的那个同一个 <code>{'<think>...</think>'}</code>{' '}
        块的长度与内容——测试时计算被落到了字面意思上：「在强制它停下之前，允许这段思考块跑多少个 token」。
      </p>

      <h2>你浏览器里的 Qwen 也这样做吗？</h2>
      <p>
        部分是——差距值得说清楚。Qwen3.5 自己的 model card 关于推理训练只公开了一句含糊的话——
        <em>「Scalable RL Generalization: reinforcement learning scaled across million-agent environments with
        progressively complex task distributions」</em>——没有 GRPO 这个名字，没有奖励描述，也没有 verifier
        对数。相比上一代架构自己的 Qwen3-Next 博客，这是一次实打实的披露倒退——那篇博客点名了 GRPO，还给出了一个具体数字——3,995
        组 query-verifier pair——用于它的 Reasoning-RL 阶段。而这份 card 明确说了的是：旗舰型号 Qwen3.5-397B-A17B
        默认<strong>开启</strong> thinking mode，但这个浏览器里跑的这个具体模型 Qwen3.5-0.8B，默认<strong>关闭</strong>{' '}
        thinking mode——正好相反的默认值——厂商还专门加了一条警告：为这个最小的 checkpoint 强行打开 thinking
        mode，容易让它陷入无法正常终止的推理循环。所以诚实的说法是：{' '}
        <ChapterLink chapterId="post-training">第 15 章</ChapterLink>那套 <code>{'<think>'}</code>{' '}
        机制，在这份 checkpoint 里是真实存在的；思维链和自洽性是通用技巧，你可以自己对包括这个模型在内的任何模型多采样几次去用；但厂商对这个具体在这个标签页里跑的
        0.8B 模型给出的建议，是让 thinking 保持关闭。
      </p>
    </Prose>
  );
}

/** 子章节 11.3——工具调用，以及基于语法/受限的解码。 */
export function SamplingToolCallingSection() {
  return (
    <Prose>
      <h1>工具调用与受限解码</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 11 章 Sampling——<ChapterLink chapterId="post-training">第 15 章</ChapterLink>已经带你看过{' '}
        <code>tool_call</code> 格式；这里要讲的是，一个生产系统怎样只靠本章自己的 sampling 机制、外加一个额外的
        mask，就能让这个格式变得<em>不可能出错</em>。
      </p>
      <p>
        让模型调用外部函数这件事，可以追溯到 <strong>ReAct</strong>（Yao 等人，2022 年 10 月）：在同一段生成流里，把自由文本推理和离散的动作——一次查询、一次搜索——交织在一起，让模型能够计划、行动、观察、再重新计划。OpenAI
        在 2023 年 6 月把这个模式产品化，做成了 <strong>function calling</strong>（模型 <code>gpt-4-0613</code> /{' '}
        <code>gpt-3.5-turbo-0613</code>）：模型被微调成会吐出一段描述函数调用的 JSON，而不是散文。Simon Willison
        就在同一周写道，这其实正是——「an implementation of the ReAct pattern, with models that have been fine-tuned to
        execute it」（对 ReAct 模式的一次实现，只是模型被微调过去执行它）。这里的陷阱是：微调只能教会模型<em>大多数时候</em>
        吐出格式良好的 JSON，并没有任何硬性保证它每次都会这样。
      </p>

      <h2>靠 mask，而不是靠祈祷</h2>
      <p>
        这个修法根本不碰训练——它发生在本章整章都在讲的那个确切时刻：<strong>挑选下一个 token</strong>。通常你拿到原始
        logits，可能套上 temperature，可能只保留 top-k 或 top-p 那部分概率质量，再采样或
        argmax。<strong>受限（基于语法的）解码</strong>在这之前再加一步：跟踪给定 schema
        和目前为止已生成内容，哪些 token 仍然<em>在句法上合法</em>，把每一个不合法的都 mask 成{' '}
        <MathDisplay inline latex={String.raw`-\infty`} />，然后才跑通常的 sampling。Outlines 论文（Willard &amp;
        Louf，2023 年）把这形式化为：把一个正则表达式或 JSON schema/CFG 编译成一个有限状态机，再把词表按 FSM
        状态建好索引；llama.cpp 的 GBNF 语法在 C++ 里做的是同一件事。跟着
        <ChapterLink chapterId="post-training">第 15 章</ChapterLink>引入的那个确切 tag 格式——
        <code>{'<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>'}</code>
        ——一步步走一遍：
      </p>

      <ConstrainedDecodeMask />

      <p>
        注意这个 mask 的形状并不总是一样。取值取到一半时，几乎什么都合法（值本来就是自由文本）——mask
        几乎不怎么收窄候选范围。而紧接在 get_weather 唯一必需的那个参数闭合之后，恰好只有<em>一个</em> token
        合法，<code>{'</function>'}</code>——在 temperature 或 top-p 有机会说话之前，mask
        本身就已经决定了下一个 token。这是一条通用的道理：语法 mask 保证的是<em>句法</em>不可能出错；它对<em>内容</em>——那个城市名对不对、背后的推理靠不靠谱——什么都没保证。这是两个彻底独立的问题。
      </p>

      <h2>mask 买来了什么，有数字为证</h2>
      <p>
        OpenAI 自己的数字把这个道理讲得很干净。2024 年 8 月上线 <strong>Structured Outputs</strong>（
        <code>strict: true</code>）时，他们把这次改动描述成从「微调然后祈祷」走向「一种确定性的、工程化的方法，叫做
        constrained sampling」。在他们自己的复杂 JSON schema 评测上：<code>gpt-4-0613</code>（2023
        年那一代、只靠微调的 function calling）得分不到 40%；更新的 <code>gpt-4o-2024-08-06</code>
        模型只靠微调达到 93%；在此之上再加上受限解码（Structured Outputs），达到
        <strong>100%</strong>。mask 并不会让模型更聪明地知道该调用<em>什么</em>——它只是让调用的<em>形状</em>变得不可能出错。
      </p>

      <h2>你浏览器里的 Qwen 这样做吗？</h2>
      <p>
        没有。这个仓库的 sampler（<code>crates/mlx-core/src/sampling.rs</code>）实现的正好就是
        temperature、top-k、top-p、min-p——本章前面讲过的那几个旋钮——没有任何为语法或 schema 合法性 mask token
        的东西：没有 <code>logit_bias</code>，没有允许 token 的 mask，整条流水线里也没有编译过的语法（这一点通过直接读
        sampler 源码、并在这个项目的 browser/WASM 驱动代码里搜索语法相关关键词确认过——一个都没有）。工具调用反而是<em>事后</em>找回来的：模型完全自由地生成，而
        <code>crates/mlx-core/src/tools/mod.rs</code>——它自己的文档注释就直说了——「uses simple string-based parsers
        instead of regex」（用简单的基于字符串的解析器，而不是正则），在生成结束之后去扫描已经写完的文本里的{' '}
        <code>{'<tool_call>'}</code> 标签。没有硬约束这件事，最清楚的证据是：这个解析器自己的结果类型{' '}
        <code>ToolCallResult</code> 有真实的失败状态——<code>invalid_json</code>、<code>missing_name</code>、
        <code>parse_error</code>——而这些状态之所以存在，正是因为一开始就没有任何东西阻止模型生成格式不对的输出。真正的语法
        mask 会让这些状态在设计上就不可达；在这里，它们是活生生会被走到的代码路径。所以{' '}
        <ChapterLink chapterId="post-training">第 15 章</ChapterLink>给你看过的那个 tag
        格式，其可信程度正好等同于模型自己学会的、把格式写对的那个习惯——和 OpenAI 最早 2023 年的 function
        calling 属于同一类，而不是后来那种受限解码的类别。
      </p>
    </Prose>
  );
}

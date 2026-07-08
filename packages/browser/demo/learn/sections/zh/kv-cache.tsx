// 第 12 章（KV cache 与混合注意力）两个子章节正文的简体中文译文，与英文版
// ../kv-cache.tsx 平行维护：同名导出 KvCacheBatchingSection /
// KvCachePrefixCachingSection。与英文版一样是纯静态阅读内容（SSR 安全，可被
// 预渲染器直接渲染为静态 HTML）；翻译不改动任何数字与结论，术语保留英文。

import { Prose } from '../../Prose';
import { BatchingModesDiagram } from '../../widgets/BatchingModesDiagram';
import { DynamicQuantMix } from '../../widgets/DynamicQuantMix';
import { FormatZooLadder } from '../../widgets/FormatZooLadder';
import { Fp8Fork } from '../../widgets/Fp8Fork';
import { GgufQuantLadder } from '../../widgets/GgufQuantLadder';
import { GranularityZoom } from '../../widgets/GranularityZoom';
import { InferenceMetricsDiagram } from '../../widgets/InferenceMetricsDiagram';
import { MxBlockAnatomy } from '../../widgets/MxBlockAnatomy';
import { NF4BellCurve } from '../../widgets/NF4BellCurve';
import { NumberLineGrid } from '../../widgets/NumberLineGrid';
import { PrefixCacheDiagram } from '../../widgets/PrefixCacheDiagram';
import { QuantizationDiagram } from '../../widgets/QuantizationDiagram';
import { QuantizeRoundTrip } from '../../widgets/QuantizeRoundTrip';
import { QuantMethodsMap } from '../../widgets/QuantMethodsMap';
import { RangePrecisionScatter } from '../../widgets/RangePrecisionScatter';
import { SixteenBitFork } from '../../widgets/SixteenBitFork';
import { SymmetricAsymmetric } from '../../widgets/SymmetricAsymmetric';

/** 子章节 12.1——服务速度怎么衡量，以及 batching 如何把它放大。 */
export function KvCacheBatchingSection() {
  return (
    <Prose>
      <h1>latency、throughput 与 batching</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 12 章 KV cache——「快」到底怎么衡量，以及一台服务器如何把一个 memory-bound 的 decode 变成面向众人的
        throughput。
      </p>
      <p>
        KV cache 正是 decode 快的<em>原因</em>：每个新 token 复用它早已算好的 key 和 value，而不是把整段提示词重算一遍。但「快」其实是两个不同的数字，取决于你指的是哪一个；而一台真正的服务器要同时伺候的不是一个用户，而是数百个。这个子章节讲的就是服务层：先看速度怎么
        <em>衡量</em>，再看 <strong>batching</strong> 如何把你在演示里看到的那个 memory-bound decode 变成 throughput。
      </p>

      <h2>两个时钟——以及为什么平均值会骗人</h2>
      <p>
        当输出逐 token 流式吐出时，有两个时钟在走，它们量的是不同的东西。<strong>TTFT</strong>（time to first
        token）是第一个词出现前的停顿——它由<em>受算力限制</em>的 prefill 决定，prefill 要一口气消化你整条提示词。
        <strong>TPOT</strong>（time per output token，也叫 <strong>ITL</strong>，inter-token latency）是此后稳定的流式速度——由<em>受显存带宽限制</em>的
        decode 决定。两者此消彼长，你分开调它们：<code>10&nbsp;ms</code> 的 ITL，对那一个用户来说就是{' '}
        <code>100 tokens/sec</code>。（对于<em>非流式</em>调用——一个 agent 发起一次工具调用、等整段 JSON
        ——你感受到的不是这两个时钟中的任何一个，而是总响应时间。）
      </p>
      <p>
        陷阱在于：把其中任何一个当成<em>平均值</em>来报告。推理 latency 是<strong>右偏</strong>的：大多数请求聚集在典型值附近，拖着一条由倒霉请求组成的长长慢尾。mean
        被那条尾巴拉向右边、把它<em>掩盖</em>了——于是工程师改用<strong>百分位</strong>来报告。P50 是中位数（一半更慢），P90
        是十分之一，P99 是一百个里最糟的那个。可靠性工作盯的是 <strong>P90/P99</strong>，因为用户真正记住的正是那条尾巴。盯住两个时钟，也盯住平均值如何撒谎：
      </p>

      <InferenceMetricsDiagram />

      <h2>batching——一次权重读取，许多 token</h2>
      <p>
        现在是那根杠杆。回想 <a href="/zh/chapters/attention/roofline">roofline</a>：一个 decode step 把模型的{' '}
        <em>全部</em>权重从内存里恰好读一遍，而在 batch 1 时，这整趟扫描只产出一个 token——最左侧的 memory
        悬崖。解法直接得近乎令人难堪。让 <code>N</code> 个请求共用<em>同一次</em>权重读取，GPU 就用一次内存扫描吐出{' '}
        <code>N</code> 个 token。total throughput 随 batch 上升，而内存流量纹丝不动——batching 就是 memory wall
        的解药，也是 decode 负载沿 roofline 爬向 ridge 的方式。
      </p>
      <p>
        各家服务器的区别在于<em>何时</em>启动一个 batch。<strong>static</strong> batching 等到 batch 装满，于是第一个到的为最后一个埋单。
        <strong>dynamic</strong> batching 在装满<em>或</em>超时触发时启动，谁先到算谁。<strong>continuous</strong>
        （即 <em>in-flight</em>）batching——<code>vLLM</code> 和 <code>SGLang</code> 的做法——在 <em>token</em>
        级别把新请求换进空位，于是每个几乎立刻开始、queue 始终很短。切换这三种，看请求 lane 重新排布时序：
      </p>

      <BatchingModesDiagram />

      <p>
        关键在于 batching 是一个<strong>旋钮，不是白赚</strong>。把 batch 往上滑，两个读数朝<em>相反</em>方向移动：total
        throughput 先升后饱和（你离开 memory-bound 区间、撞上 compute 上限），而每个用户各自的 latency
        上升（每一步要分摊的同行更多了）。挑一个 batch size，就是在那条 latency-对-throughput
        曲线上选一个点——要么对单个用户快，要么对许多人来说每 token 更便宜。
      </p>
      <p className="text-muted-foreground">
        这也正是为什么这趟浏览器之旅和托管 API 的体感不同：你是 <strong>batch 1 的单一用户</strong>，是 memory-bound
        decode 的最坏情况，因为那一次昂贵的权重读取只驮着一个 token。共享的生产服务器把几十个陌生人编织进一个
        continuous batch，让同一次读取被所有人分摊——同一个模型，相反的经济学。
      </p>
    </Prose>
  );
}

/** 子章节 12.2——跨请求复用一个 prompt 的 KV cache，而不仅仅是在一次生成内部。 */
export function KvCachePrefixCachingSection() {
  return (
    <Prose>
      <h1>prefix caching</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 12 章 KV cache——<em>跨</em>请求复用一个 prompt 的 KV，而不仅仅是在一次生成内部。
      </p>
      <p>
        你在本章认识的 KV cache 是在一次生成<em>内部</em>复用的：单条回复里的每个 token，复用它前面那些 token 的 key 和
        value。<strong>prefix caching</strong> 把同一个想法往上抬一层——在恰好以相同 token 开头的<em>不同请求之间</em>复用那份
        KV。
      </p>
      <p>
        机制是精确的。两个共享前导 <strong>prefix</strong> 的请求，共享这些 token 的 key 和 value，于是这段 prefix 只计算
        <strong>一次</strong>，之后每个请求都在它上面<em>跳过 prefill</em>——一次 <strong>cache hit</strong>，把它的 TTFT
        一路降到第一个新 token。但复用在<strong>第一个不同的 token</strong> 处终止。模型是自回归的，所以一个不同的 token
        会改变其后<em>所有</em> token 的 hidden state——也就改变了 KV——哪怕再往下看上去完全一样的文本也是如此。过了第一处 miss，cache
        就一文不值：
      </p>

      <PrefixCacheDiagram />

      <p>
        这条「第一处差异」的规则也是一条设计经验：<strong>把新内容尽量放在靠后的位置</strong>。一段固定的长{' '}
        <strong>system prompt</strong> 后面跟着用户的问题，每次都能复用整段 system prompt；同样的内容若把每请求各异的 id
        贴在<em>最前面</em>，则什么都复用不了。最大的收益正是那些高复用形状——agent 和 chatbot 的长 system prompt、带共享检索文档的{' '}
        <strong>RAG</strong>、共享的代码上下文，以及多轮对话（它每一轮都会重发到目前为止的整段对话）。
      </p>
      <p className="text-muted-foreground">
        对本 demo 还要补一句诚实的脚注：作为 batch 1 的单一用户，你没有任何对象可以共享 prefix——每个请求都要把整段
        prompt 冷启动重新 prefill 一遍。生产环境的 API 会在数百万请求间复用同一段固定的 system prompt，这正是按 token
        计费的 API 把 <strong>cache-hit</strong> 的 token 算得更便宜、在样板内容上几乎瞬时、只在真正新的 token
        上「思考」的原因。
      </p>
    </Prose>
  );
}

/** 子章节 12.3——每个 weight 用更少的 bit：对 memory-bound decode 最直接的那根杠杆。 */
export function KvCacheQuantizationSection() {
  return (
    <Prose>
      <h1>quantization</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 12 章 KV cache——把每个 weight 存得更省 bit，同一个 memory-bound 的 decode 就跑得更轻、更快。
      </p>
      <p>
        本章的一切都指向同一个瓶颈：decode 是 <strong>memory-bound</strong> 的。每个 token 都要把整个模型从内存里重读一遍，而在
        batch 1 时，这整趟扫描只换来一个 token。<a href="/zh/chapters/attention/roofline">roofline</a>{' '}
        给它标了个数——你被钉在最左侧的 memory 悬崖上。<strong>quantization</strong> 是撬动这道悬崖最直接的一根杠杆，而且它<em>不动模型结构</em>
        ：形状不变、前向过程不变——只是<em>把每个 weight 存得更省 bit</em>，一份落在更粗格点上的近似副本。每个 weight 的字节更少，就意味着要搬运的更少，memory-bound
        的 decode 也就跑得更轻。
      </p>

      <h2>更少的 bit，更小的模型，更快的 decode</h2>
      <p>
        一个 <code>bf16</code> 的 weight——也就是本 demo 用的——占 2 字节。降到 <code>fp8</code> 或 <code>int8</code>{' '}
        就是 1 字节；<code>int4</code> 只占半字节。占用随着 bit 数直线下降，而因为 decode 是 memory-bound 的，搬运这些 weight
        的时间也随之下降。但这<strong>不是</strong>干净的每级 2×：开销——要把它反量化回一个计算格式、bandwidth
        用不满——意味着书里给的是更诚实的<strong>每降一个精度级别 +30–50%</strong>。逐个切换这些格式，看三根条——大小、速度、质量——一起移动：
      </p>

      <QuantizationDiagram />

      <h2>它怎么工作——又要付出什么</h2>
      <p>
        诀窍是一个 <strong>scale factor</strong>：一块 weight 被映射到一小撮整数档位上，再由一个共享的浮点数在计算时把它们重新缩放回去（
        <span className="font-mono">real ≈ scale × (q − zero_point)</span>——其中 <em>zero_point</em>
        只是把格点整体平移，让真实的 0 依然落在一个精确的码上；详见 <a href="/zh/chapters/kv-cache/number-formats">number formats 子章节</a>
        ）。bit 越少，格点越粗，代价就是<strong>质量</strong>。书里把谁更耐受排了个序，由弱到强：
        <strong>weights &lt; activations &lt; KV cache &lt; attention</strong>。像 <strong>GPTQ</strong> 和{' '}
        <strong>AWQ</strong> 这类 weight-only 方法只量化 weight，常常一路到 <code>int4</code>{' '}
        都近乎无损；推进到 weight <em>和</em> activation 一起（<code>W8A8</code>，例如 <strong>SmoothQuant</strong>、
        <strong>LLM.int8()</strong>）就更难，因为 activation 带着更狂野的 outlier。经验法则：<code>fp8</code> / MXFP8
        是 sweet spot——几乎察觉不到损失——而整数格式缺少动态范围，所以对质量敏感的工作你还是坚持浮点，并把低于 4-bit 当成悬崖。
      </p>

      <h2>KV cache 也能被量化</h2>
      <p>
        这不只关乎 weight。你在本章认识的 <strong>bf16 KV cache</strong> 本身也是一串字节，decode 每一步都要把它读回来——所以它同样是一个量化对象。它<em>中等</em>敏感（还是书里的那个排序）：它的误差会<strong>逐 token
        累积</strong>，因为每个缓存下来的 key/value 会被序列其后的全部内容复用。<code>fp8</code> / MXFP8
        在这里是稳妥之选；激进的整数 KV 量化才是质量开始打滑的地方。
      </p>
      <p className="text-muted-foreground">
        对本 demo 的诚实脚注：浏览器内的模型以 <strong>bf16</strong> 运行，<strong>完全没有 quantization</strong>
        ——最简单、保真度最高的选择。书中指出，整数 / 低于 4-bit 的 quant 多半是<em>本地推理</em>的做法（你为 Ollama 或
        llama.cpp 下载的那些 GGUF 文件），与数据中心的服务有别。<em>同一个模型</em>的量化副本会小 ≈2×（<code>int8</code>）到
        ≈4×（<code>int4</code>），decode 也会实打实地快上一截——这正是整章一直在指向的那根杠杆。
      </p>
    </Prose>
  );
}

/** 子章节 12.4——number formats 与 precision：上面 quantization 入门篇的深入版，与英文版 ../kv-cache.tsx 同名导出。 */
export function KvCacheNumberFormatsSection() {
  return (
    <Prose>
      <h1>number formats 与 precision</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 12 章 KV cache——把 quantization 的盖子掀开：一个 weight 到底怎么以 bit 存储、各种格式能表示与不能表示什么，以及一个真实数值是怎样被吸附到一张粗糙的格点上的。
      </p>
      <p>
        <a href="/zh/chapters/kv-cache/quantization">上一个子章节</a>把 quantization 当作一根杠杆：每个 weight 用更少的
        bit，要搬运的就更少，memory-bound 的 decode 也更快。这一节把盖子掀开。一个 weight 在 bit 这一层<em>究竟</em>是什么？为什么{' '}
        <code>bf16</code> 能装下 10³⁸ 这么大的数，而 <code>fp16</code> 一过 65,504 就溢出？「用更少的 bit 存一个
        weight」到底发生了什么？归根结底是两件事——float 怎么切分它的 bit，以及一张格点怎样替代一个连续的数。
      </p>

      <h2>一个 weight 不过是一串 bit</h2>
      <p>
        一个 floating-point 数就是 <span className="font-mono">sign × mantissa × 2^exponent</span>，bit 被切成这三个字段。这个切法就是全部关键：<strong>exponent</strong>{' '}
        字段换来 <strong>dynamic range</strong>（一个数能多大、能多小），<strong>mantissa</strong> 字段换来{' '}
        <strong>precision</strong>（数值之间排得多密）。integer 根本没有 exponent——它是一张<strong>均匀</strong>的格点，每一步都一样大，本身没有任何
        dynamic range。这一个对比，是下面一切的地基：
      </p>
      <NumberLineGrid />
      <p>
        integer 把档位均匀铺满整个范围。float 把它们挤在 0 附近——大多数 weight 真正待的地方——再让它们朝最大值一路稀疏开来。同样数量的档位、相反的布局，而且只有
        float 带着 range。把一个 bit 花在 exponent 上，你就能够到 10³⁸；花在 mantissa 上，你就能分辨更细的差别。两者不可能同时免费拥有。
      </p>

      <h2>同样的 16 个 bit，两种切法</h2>
      <p>最干净的示范就是两个 16-bit 格式。它们有<em>相同</em>的 bit 预算，却切得正好相反——这一个选择就决定了各自擅长什么：</p>
      <SixteenBitFork />
      <p>
        <code>fp16</code> 保留了较大的 mantissa（精度细），但 exponent 只有 5 个 bit，所以最大只到 65,504，且很容易下溢——用{' '}
        <code>fp16</code> 训练需要 loss scaling 来防止梯度消失。<code>bf16</code> 保留了 fp32 完整的 8-bit
        exponent，因此 range 与 fp32 一样大（只是步长更粗），通常不需要 loss scaling。这正是 <code>bf16</code>{' '}
        成为默认训练精度的原因——也正是<strong>本 demo 所采用的</strong>。你在这里。
      </p>

      <h2>格式动物园</h2>
      <p>低于 16 bit，字段怎么切就成了一个真正的设计抉择，所以每一种 bit 宽度都不止一个版本。这就是整条梯子——点任意一行，看看它能表示什么：</p>
      <FormatZooLadder />
      <p>
        注意 <code>fp8</code> 有两个版本：<strong>E4M3</strong>（mantissa 更多，max 448——用于 weight 与
        activation）和 <strong>E5M2</strong>（exponent 更多，max 57,344——用于 gradient）。这和 16-bit 的那一对是同一个
        岔路口，只是低了一档——而且 fp8 训练<em>两半都用</em>：
      </p>
      <Fp8Fork />
      <p>
        把所有格式画到 range 对 precision 的平面上，「为什么每种宽度都有两个」就一目了然了：在固定的 bit
        宽度下，你必须选一个<em>角</em>。
      </p>
      <RangePrecisionScatter />
      <p>
        一个会绊倒 fact-checker 的坑：<code>fp8</code> E4M3 的最大值 448 是 ML（OCP）的约定——一个朴素的 IEEE
        式 8-bit float 只能到 240。E4M3 把 infinity 和大部分 NaN 的 bit 模式回收过来扩展了 range，而更小的{' '}
        <code>fp6</code>/<code>fp4</code> 干脆彻底丢掉 infinity 和 NaN，换回几个宝贵的码点。<strong>subnormal</strong>
        （渐进下溢，隐含的前导 1 变成 0）以降低 precision 为代价，多换来一小撮极小的量级——也就是每种格式还能叫得出名字的最小的数。
      </p>

      <h2>把真实数吸附到格点上</h2>
      <p>
        量化一个 weight，不过是把它四舍五入到格点最近的一格。格点的间距就是 <strong>scale factor</strong>{' '}
        <span className="font-mono">s</span>；剩下的零头就是 quantization error，最多半格：
      </p>
      <QuantizeRoundTrip />
      <p>
        整个机制就这些：<span className="font-mono">q = round(x / s)</span> 选最近的档位，
        <span className="font-mono">x̂ = s · q</span> 把它读回来，误差永远不超过 <span className="font-mono">s / 2</span>
        。格点越粗（bit 越少）步长越大、误差越大。但 weight 并不总是以 0 为中心——这就轮到 <strong>zero-point</strong> 出场：
      </p>
      <SymmetricAsymmetric />
      <p>
        weight 大致以零为均值，所以用<strong>对称（symmetric）</strong>格点（zero-point 为 0）。activation
        是偏斜、单边的（想想 ReLU 之后，全都 ≥ 0），所以用<strong>非对称（asymmetric）</strong>格点——zero-point
        把整张格点平移，让真实的 0 依然恰好落在一个整数码上（<span className="font-mono">q = round(x / s) + z</span>
        ）。这也是更深一层的原因：为什么 weight-only 量化是轻松的第一步，而量化 activation（<code>W8A8</code>
        ）才是难处——activation 带着狂野、随输入而变的 outlier，单一一个 scale 兜不住。
      </p>

      <h2>一个 scale 管多少个 weight？</h2>
      <p>
        整个 tensor 共用一个 scale，存起来便宜但很松——一个 outlier weight 就把所有人的格点拉宽了。给每个
        output channel、或每 32–128 个 weight 一个块各自一个 scale，每张格点就能更贴合自己的那批值。这就是{' '}
        <strong>granularity</strong>（粒度）的梯子：
      </p>
      <GranularityZoom />
      <p>
        粒度越细，误差越低——但 scale 也要占 bit，所以一个每 64 个值共享一个 scale 的「4-bit」weight，其实约为
        <strong>4.25 bit</strong>。Microscaling（MX）标准把这一点固定在 32 个元素一个块上，这也是通往最小格式的桥。
      </p>

      <h2>块格式把 range 借回来</h2>
      <p>
        单看一个 4-bit float，它只能叫出寥寥几个量级——<span className="font-mono">{'{0, 0.5, 1, 1.5, 2, 3, 4, 6}'}</span>
        。这本身没什么用，直到一<em>整块</em>它们<em>共享</em>一个负责提供量级的 scale。这就是为什么 <code>fp4</code> 和{' '}
        <code>fp6</code> <em>永远只是</em>块格式——而同一套 microscaling recipe 也会把 <code>fp8</code> 包起来。切换课程点名的这三种：
      </p>
      <MxBlockAnatomy />
      <p>
        OCP 的 <strong>MX</strong> 家族用同一条规则覆盖三种宽度，始终配一个二的幂 <code>E8M0</code> scale（8 个 exponent
        bit、没有 mantissa，跨度 2⁻¹²⁷ … 2¹²⁷），由一个 32-元素的 block 共享。<strong>MXFP8</strong> 把 <code>fp8</code> 包成
        ≈ <strong>8.25</strong> bits/elem——fp8 本身就有 range，所以这是 near-lossless 的那一端。<strong>MXFP4</strong> 在{' '}
        <code>fp4</code> 上用同样的 block，≈ <strong>4.25</strong> bits/elem，这里共享 scale 正是让 4-bit 能用起来的关键。NVIDIA 的{' '}
        <strong>NVFP4</strong> 把块收紧到 16 个元素，换上更丰富的 <code>fp8</code> E4M3 block scale（一个真正的 float，带
        mantissa，比二的幂更细），再加一个 per-tensor 的 <code>fp32</code> scale（≈ <strong>4.50</strong> bits/elem）——两级
        scaling，把 range 和精度都抠回来。
      </p>

      <h2>聪明的 4-bit：NF4</h2>
      <p>
        如果 pretrained weight 大致是钟形分布，那为什么还要把档位均匀排开？<strong>NF4</strong> 不这么做。它把 16
        个档位摆在正态分布的<em>分位数</em>上——weight 真正密集的地方密，尾巴上稀：
      </p>
      <NF4BellCurve />
      <p>
        均匀的 <code>int4</code> 把它 16 个码里同样多的份额花在罕见的尾巴和拥挤的中心；NF4 顺着数据走，于是它更多的分辨率落在
        weight 所在之处。它对正态分布的 weight 是信息论意义上最优的，含一个精确的 0，且本质上是一张 16 项的<em>查找表</em>
        而非一张算出来的格点——也就是 <strong>QLoRA</strong> 在其上做微调的那个 4-bit 底座。
      </p>

      <h2>挑一套配方</h2>
      <p>
        这些零件——一个格式、一个 scale、一种粒度——组合成有名字的方法。它们落在两条轴上：你量化多少（只量 weight，还是
        weight <em>和</em> activation 都量），以及你愿意重训多少（<strong>PTQ</strong>，只做 calibration；还是{' '}
        <strong>QAT</strong>，连着 fake-quant 一起重训）：
      </p>
      <QuantMethodsMap />
      <p>
        对开放权重你几乎总是做 PTQ——QAT 需要一整套训练循环。绝大多数本地推理的量化（<strong>GPTQ</strong>、
        <strong>AWQ</strong>、<strong>GGUF</strong>、<strong>NF4</strong>）都是 weight-only：它压缩显存、加速 memory-bound
        的 decode，却不碰 activation。有两个值得点名的坑：<strong>AWQ</strong> 是 weight-only，<em>尽管</em>名字叫
        「Activation-aware」（activation 只是<em>挑出</em>哪些 weight 通道要保护），而一个 GGUF <code>Q4_K</code> 算上它存的块
        scale 之后，其实约为 <strong>4.5</strong> bit 每 weight，而不是 4（其它 <code>Q4*</code> 变体各不相同）。
      </p>

      <h2>你真正下载到的：GGUF k-quants</h2>
      <p>
        在 Hugging Face、Ollama 或 LM Studio 上，你挑的不是「int4」——你挑的是一个 <strong>GGUF Q-type</strong>。它们是
        llama.cpp 的文件格式，带「<code>_K</code>」的那些是 <em>k-quants</em>：256 个权重一个 super-block，带两级 scale——每个
        super-block 一或两个 float（第二个是 <code>dmin</code>，用于带非对称零点的类型），再加每 16 或 32 个权重（视具体类型而定）的
        sub-block 一个量化后的小 scale。这正是你刚在 MX 里看到的 block-format 把戏，也正
        因如此，它们的代价是<em>带小数</em>的 bits/weight，永远不是个整数。下面每一档都是<strong>纯类型</strong>的 bits/weight，直接
        从 <code>ggml-common.h</code> 里的 C struct 读出来：
      </p>
      <GgufQuantLadder />
      <p>
        两个要说实话的坑。第一，大家真正下载的名字——<code>Q4_K_M</code>、<code>Q3_K_M</code>、<code>Q5_K_M</code>——并不是纯类型，而是
        <strong>混合</strong>：<code>_M</code> 把最敏感的张量保持在更高位宽——总是有 <code>attn_v</code> 和 <code>ffn_down</code>，
        <code>Q3_K_M</code> 还会加上 <code>attn_o</code>——所以<em>文件</em>会比纯数字略高（一个 8B 的 <code>Q4_K_M</code> ≈{' '}
        <strong>4.9</strong> bpw，而不是 4.5）。第二，<code>IQ</code>
        那几档（<strong>I-quant</strong>）根本不是 round 的格点——每个权重是一个固定 codebook 的索引，由一个 <strong>importance
        matrix</strong> 帮着挑：拿一点 calibration 文本过一遍模型，看哪些 weight 最能改变输出，就把 bits 花在那里。低于 ~3 bit 时，正是这个
        imatrix 才让文件勉强能用。
      </p>
      <p>
        把「保持敏感张量高位」这个想法推广到<em>每一层</em>，你就得到了 <strong>dynamic</strong> 量化——也就是 Unsloth Dynamic 2.0
        GGUF 的核心。同样的大小预算，按 calibration 说了算的地方去花：
      </p>
      <DynamicQuantMix />
      <p>
        Unsloth 的 <strong>Dynamic 2.0</strong> 从一个 &gt; 1.5M token 的 calibration 集里，<em>逐层、逐模型</em>地挑量化类型，并用相对原
        模型的 <strong>KL-divergence</strong> 来评判，而不是 perplexity（perplexity 的误差会自己抵消——见「Accuracy is Not All You
        Need」，<span className="font-mono">arXiv 2407.09141</span>）。推到极限，同样的思路把一个 671B 的 DeepSeek-R1 从约 720&nbsp;GB
        压到 <strong>131&nbsp;GB</strong>，平均约 1.58-bit 的 <code>IQ1</code>——把少数 dense 与 attention 层保持在 4–6 bit，而 MoE
        experts 降到约 1.5。有一点要分清楚：这个「1.58-bit」是<em>训练后</em>的 dynamic 量化，和 <strong>BitNet b1.58</strong>
        （<span className="font-mono">arXiv 2402.17764</span>）不是一回事——后者是从头<em>训练</em>出 {'{−1, 0, 1}'} 的 ternary
        权重，两者只是共用了 log₂3 ≈ 1.58 这个数字。
      </p>

      <h2>硬件在哪里划线</h2>
      <p>
        每一种格式之所以存在，都是因为前一种对某类工作负载用尽了 range 或 precision——而硅片随之跟上。NVIDIA{' '}
        <strong>Turing</strong>（2018）最早加入了 <code>int8</code>/<code>int4</code> Tensor Core，用于推理；
        <strong>Ampere</strong>（A100）加入了 <code>tf32</code>/<code>bf16</code>/<code>fp64</code> Tensor Core
        支持与结构化稀疏（structured sparsity）；<strong>Hopper</strong>
        （H100）加入了原生 <code>fp8</code>（E4M3/E5M2）；<strong>Blackwell</strong>（B200）加入了 <code>fp4</code>
        （NVFP4/MXFP4），而每当精度减半，Tensor Core 的吞吐大致<em>翻倍</em>。Apple-Silicon GPU——以及 NVIDIA Blackwell
        之前（RTX 50 系列以前）的消费级显卡——没有原生的低于 8-bit 的 <em>float</em> 单元，所以这些设备上的低比特推理要靠{' '}
        <code>int4</code>/<code>int8</code> 整数运算来跑——这也正是为什么激进的整数量化是本地推理的故事，而非数据中心的。
      </p>
      <p className="text-muted-foreground">
        诚实的脚注，和入门篇一样：这个浏览器标签页对 weight 和 KV cache <em>都</em>采用 <strong>bf16</strong>——完全没有
        quantization，是保真度最高的选择。这一页上的一切，是你<em>可以</em>拿这份保真度去换的菜单：同一个模型的一个更小、更快的副本，用一张更粗的格点买来。你在这里，站在梯子的顶端，往下看。
      </p>
    </Prose>
  );
}

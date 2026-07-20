// 第 4 章（自注意力）两个子章节正文的简体中文译文，与英文版 ../attention.tsx
// 平行维护：同名导出 AttentionMemoryWallSection / AttentionFlashAttentionSection。
// 与英文版一样是纯静态阅读内容（SSR 安全，可被预渲染器直接渲染为静态 HTML）；
// 运行时相关的技术断言均以真实 WebGPU 后端为依据，翻译不改动任何数字与结论。

import { Prose } from '../../Prose';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { Flash2ParallelismDiagram } from '../../widgets/Flash2ParallelismDiagram';
import { Flash3AsyncDiagram } from '../../widgets/Flash3AsyncDiagram';
import { FlashAttentionTimeline } from '../../widgets/FlashAttentionTimeline';
import { FlashTilingDiagram } from '../../widgets/FlashTilingDiagram';
import { FlashTrafficDiagram } from '../../widgets/FlashTrafficDiagram';
import { Fp8QuantDiagram } from '../../widgets/Fp8QuantDiagram';
import { GpuHierarchyDiagram } from '../../widgets/GpuHierarchyDiagram';
import { MemoryWallDiagram } from '../../widgets/MemoryWallDiagram';
import { NativeAttentionDiagram } from '../../widgets/NativeAttentionDiagram';
import { RooflineDiagram } from '../../widgets/RooflineDiagram';
import { StreamingSoftmaxDiagram } from '../../widgets/StreamingSoftmaxDiagram';

/** 子章节 4.1——为什么 N×N 分数矩阵是开销中心。 */
export function AttentionMemoryWallSection() {
  return (
    <Prose>
      <h1>显存带宽墙</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 4 章 自注意力——<code>softmax(QKᵀ)</code> 到底会有多大。
      </p>
      <p>
        再看一眼 <code>Q · Kᵀ</code>：它是一个 <code>[seq_len, seq_len]</code> 矩阵——每一对 token 一个分数。对一条 4,096
        token 的提示词，这就是<em>每个头、每一层</em>约 1,680 万个分数。8 个 query 头加起来，单个注意力层就是约 1.34
        亿个数——在 bf16（每个 2 字节）下约 268
        MB。模型从来不需要同时用到全部这些数，朴素做法却把整个矩阵都构造出来、放在一旁。
      </p>

      <h2>先认识两种 GPU 内存</h2>
      <p>
        要明白这为什么疼，你需要知道 GPU 构造上的一个事实。它有<em>两个</em>截然不同的地方存放数字：
      </p>
      <ul>
        <li>
          <strong>SRAM</strong>——紧贴计算单元的一小条内存，真正的数学运算就在这里发生。它快得惊人，但<em>极小</em>
          （几个 MB）。把它想成你干活用的小桌面。
        </li>
        <li>
          <strong>HBM</strong>——GPU 的主内存，权重和激活都住在这里。它很大（数
          GB），但相对遥远、访问慢。把它想成走廊尽头的仓库。
        </li>
      </ul>
      <p>
        桌面放不下的东西只能存进仓库，需要时再搬回来。N×N 分数矩阵远远超出 SRAM 的容量——所以它住在 HBM
        里，每一步触碰它的操作都要为这趟搬运付费。
      </p>

      <h2>代价在带宽，不在存储</h2>
      <p>
        <em>带宽</em>指的是数据在 HBM 与计算单元之间能移动得有<em>多快</em>
        ——而不是能装多少。朴素的注意力做法对带宽极其残忍：它把完整的 N×N 矩阵写出到 HBM，读回来做
        softmax，把归一化后的分数再写回去，然后第三次读出来给 <code>V</code>{' '}
        加权。矩阵在慢速内存总线上往返大约四趟，而中间的算术相比之下微不足道。注意力的大部分时间花在
        <em>搬运那个矩阵</em>上，而不是计算——所以我们说它受显存带宽限制（memory-bandwidth-bound）。
      </p>
      <p>看它如何发生——拖动序列长度，感受 N² 的爆炸：</p>

      <MemoryWallDiagram />

      <p className="text-muted-foreground">
        诚实起见的一个限定：这只在 <em>prefill</em>（预填充——一趟处理很多个提示词位置，无论是一趟还是拆成几个 chunked
        pass）阶段、且只在 6 个全量注意力层上才会咬人。逐 token 的解码（每次只有一行新
        query，也就是你在本章演示里看的部分）和 18 个 GatedDeltaNet 层从头到尾都不会构造 N×N 矩阵。
      </p>
      <p>而这恰好就是下一个子章节绕开的问题：在完全不把 N×N 矩阵写进 HBM 的情况下，算出同样的答案。</p>
    </Prose>
  );
}

/** 子章节 4.2——Flash Attention、在线 softmax，以及 Qwen 实际在跑什么。 */
export function AttentionFlashAttentionSection() {
  return (
    <Prose>
      <h1>Flash Attention</h1>
      <p className="text-muted-foreground">深入阅读 · 第 4 章 自注意力——同样的结果，却没有 N×N 矩阵。</p>
      <p>
        <strong>Flash Attention</strong> 计算的仍然是同一个 <code>softmax(QKᵀ/√d)·V</code>——<em>精确</em>
        的注意力运算本身，而不是稀疏注意力、低秩注意力那样的近似——但从不显式构造 N×N 矩阵。它把 <code>Q</code>、
        <code>K</code>、<code>V</code> 切成小块（tile），对每个 query 块流式扫过各个 key 块，同时为每个 query
        行维护一份很小的滚动状态：两个标量——迄今的最大分数 <code>m</code> 和指数的滚动和 <code>ℓ</code>
        ——外加一个滚动的输出<em>向量</em> <code>o</code>，每个输出维度一个分量。
      </p>
      <p className="text-muted-foreground">
        这里的“精确”说的是算法，不是比特：因为它按不同顺序累加各个 key
        块，输出与朴素路径只在浮点舍入意义上一致，并非逐比特相同。这是常规的浮点注意事项——改变求和顺序会改变最后几位——而不是对注意力本身的近似。
      </p>
      <p>
        诀窍在于<em>在线 softmax（online softmax）</em>
        ：当后来的块抬高了滚动最大值时，把已经累积的量重新缩放，归一化就始终严格正确。下面的更新式里，
        <MathDisplay inline latex={String.raw`\tilde\ell`} /> 和 <MathDisplay inline latex={String.raw`\tilde o`} /> 是
        <em>这一块</em>自己针对更新后的最大值 <MathDisplay inline latex={String.raw`m_i`} />{' '}
        计算的和与加权值——因此只要滚动总量乘上 <MathDisplay inline latex={String.raw`e^{\,m_{i-1}-m_i}`} />
        ，新块就能直接并入，不需要任何额外因子。
      </p>
      <MathDisplay
        latex={String.raw`m_i = \max(m_{i-1}, \tilde m), \quad \ell_i = e^{\,m_{i-1}-m_i}\,\ell_{i-1} + \tilde\ell, \quad o_i = e^{\,m_{i-1}-m_i}\,o_{i-1} + \tilde o`}
      />
      <p>
        这些小块住在 GPU
        的快速片上内存（SRAM）里，巨大的矩阵从头到尾不去慢速内存。同样的答案，一小部分的内存流量——这种技术叫“IO
        感知”（IO-aware）。
      </p>

      <h2>一段简史</h2>
      <p>
        Flash Attention 不是凭空出现的——它是“计算 softmax
        而不必把整行同时放进内存”这条思想链的最终回报。逐步看这条脉络：
      </p>

      <FlashAttentionTimeline />

      <p className="text-muted-foreground">
        同样的想法很快传到了原始代码之外：xformers 的 <code>memory_efficient_attention</code>、OpenAI 的 Triton flash
        内核、NVIDIA 的 cuDNN 融合注意力，以及 PyTorch 的 <code>scaled_dot_product_attention</code>
        （它会自动分派到最快的实现）。而 PagedAttention（来自 vLLM）是表亲，不是某个版本——它是围绕分页 KV
        缓存布局构建的注意力内核，解决的是内存高效的推理服务，与 FlashAttention 的稠密 SRAM
        内分块是不同的问题。你浏览器里的 Qwen 在 prefill 阶段运行的融合内核是 <strong>FlashAttention-2 风格</strong>
        的分块内核——v3 的技巧专属于 NVIDIA 的 Hopper GPU 和 FP8，在这里用不上。
      </p>

      <h2>每个版本到底是怎么干的</h2>
      <p>上面的时间线是地图；下面是地形。每一步都修掉了上一步的瓶颈，所以按顺序读最容易理解——一次一个机制。</p>

      <h3>地基：流式 softmax</h3>
      <p>
        一切始于一个技巧，所以我们放慢脚步把它吃透。先看 softmax 本身，用真实数字。softmax
        通过对每个分数取指数、再除以总和，把一行分数变成和为 1 的权重。拿这一行 <code>[2, 1, 3]</code> 来说：指数分别是{' '}
        <MathDisplay inline latex={String.raw`e^{2} \approx 7.39`} />、
        <MathDisplay inline latex={String.raw`e^{1} \approx 2.72`} />、
        <MathDisplay inline latex={String.raw`e^{3} \approx 20.09`} />
        ；它们的和是 <code>30.19</code>；逐个除一下，权重就是 <code>0.245, 0.090, 0.665</code>
        。最大的分数赢走大部分权重——这就是它的全部工作。
      </p>
      <p>
        一个实践层面的麻烦：
        <MathDisplay inline latex={String.raw`e^{x}`} /> 会爆炸。
        <MathDisplay inline latex={String.raw`e^{89}`} /> 已经超出 32 位浮点数的表示范围（上限约为{' '}
        <MathDisplay inline latex={String.raw`e^{88.7}`} />
        ），而原始注意力分数可能很大。标准解法是在取指数之前给每个分数减去该行的最大值——最大的那一项变成{' '}
        <MathDisplay inline latex={String.raw`e^{0} = 1`} />
        ，什么都不会溢出。除以总和时这个平移会被抵消，所以权重完全相同。但注意这个修复的代价：你必须先拿到
        <em>整行的最大值</em>，才能对<em>任何一个数</em>取指数。于是要扫两遍——一遍找最大值，一遍累加指数。
      </p>
      <p>
        <strong>在线 softmax</strong>（2018）把它压缩成一遍：维护一个<em>滚动</em>最大值和一个<em>滚动</em>
        和；当后来的值超过迄今的最大值时，给已经累出来的和打个<em>补丁</em>——乘以{' '}
        <MathDisplay inline latex={String.raw`e^{\,\text{old max}\; -\; \text{new max}}`} />
        。为什么这恰好是正确的补丁？你累加过的每一项都形如{' '}
        <MathDisplay inline latex={String.raw`e^{\,\text{score}\; -\; \text{old max}}`} />
        ，而
      </p>
      <MathDisplay
        latex={String.raw`e^{\,s - m_{\text{old}}} \cdot e^{\,m_{\text{old}} - m_{\text{new}}} = e^{\,s - m_{\text{new}}}`}
      />
      <p>
        ——一次乘法就把每个旧项重新表达到新最大值之下，仿佛你早就知道它一样。具体地，在下面的动画里，第一块{' '}
        <code>[2, 1, 3]</code> 结束时最大值是 <code>3</code>；接着第二块里出现了 <code>5</code>
        ，于是滚动和与滚动输出都先乘上 <MathDisplay inline latex={String.raw`e^{\,3 - 5} \approx 0.135`} />
        ，新的项再并入。2021 年的 “memory-efficient attention” 论文注意到，这个流程可以按 key 和 value 的<em>块</em>
        来跑——于是你永远不需要把整行同时放进内存。
      </p>

      <StreamingSoftmaxDiagram />

      <h3>FlashAttention——把整件事分块塞进 SRAM</h3>
      <p>
        GPU 有两种内存：<strong>HBM</strong>，遥远而慢的大主存；<strong>SRAM</strong>
        ，紧贴计算单元的微型暂存区，大约快十倍。朴素做法在 HBM 里构造完整的 N×N
        分数矩阵并来回拖拽——就是上一个子章节的“显存带宽墙”。<strong>FlashAttention</strong>（2022）让分数完全不进
        HBM。它把 <code>Q</code>、<code>K</code>、<code>V</code> 切成小块；对每个 query 块流式扫过 key/value 块，
        <em>在 SRAM 内部</em>算出每个小分数块，直接并入滚动
        softmax，然后丢掉。只有算完的输出行才写回。巨大的分数矩阵在慢速内存里从未存在过——答案一模一样，流量只剩一个零头。
      </p>
      <p>
        块要多大？刚好让所有在途数据都摆得上桌面。以常见的 64 宽的头为例，一个 128 行的 query 块是 <code>128 × 64</code>{' '}
        个数，在 fp32 下 ≈ 32 KB。同样形状的一个 key 块和一个 value 块再各占 32 KB，小小的 <code>128 × 128</code>{' '}
        分数块是 64 KB——滚动输出 <code>o</code> 自己也是一个 <code>128 × 64</code> 的块，又 32 KB（只有 <code>m</code>{' '}
        和 <code>ℓ</code> 是每行一个标量）。加起来：≈ 190 KB，正好顶到计算单元旁边那 ~200 KB SRAM
        的边缘——这恰恰是块取这个尺寸、不能更大的原因。生产内核还会用 16 位的块和更窄的 key
        块进一步压榨，但决定块尺寸的，正是“SRAM 能装下什么”这笔预算账。
      </p>
      <p className="text-muted-foreground">
        一个诚实的脚注：v1 确立的不变量是 N×N 分数矩阵永不触碰 HBM。这里展示的<em>循环顺序</em>——query 在外层、key/value
        在内层流式扫过、每个输出行只写一次——是 <strong>v2</strong> 才定下来的更干净的调度。最初的 v1
        内核实际上反着循环（key/value 在外、query
        在内），一路上反复回写输出。下图画的是现代顺序，因为那才是今天的内核真正运行的方式。
      </p>

      <FlashTilingDiagram />

      <p>
        这到底省下多少流量？在 4,096 token 时，朴素做法让分数矩阵在慢速 HBM 总线上往返四趟——写出原始分数、读回做
        softmax、写出概率、再读回给 <code>V</code> 加权。对这个模型的 8 个头、bf16 而言，<em>每个全量注意力层</em>就是 ≈
        1.07 GB 的总线流量——而这些分数用一次就被扔掉。分块内核移动的这类字节恰好为零。拖动序列长度，看差距：
      </p>

      <FlashTrafficDiagram />

      <p>
        v1 还有容易被忽略的后一半：<strong>训练</strong>。反向传播（计算梯度）通常需要<em>再次</em>
        用到注意力权重——而把 N×N 矩阵存起来留给后面用，会让整件事前功尽弃。FlashAttention 的答案是
        <em>重计算（recomputation）</em>：只保留输出和每行两个小统计量（还是那个最大值 <code>m</code> 和和{' '}
        <code>ℓ</code>），等反向传播需要时，再在 SRAM 里由 Q 和 K <em>重新推导</em>出每个分数块。这是有意用<em>更多</em>
        算术去换<em>更少</em>的内存流量——而且是赚的，因为在现代 GPU 上数学便宜、去 HBM
        的往返不便宜。这笔反直觉的交易正是整个“IO 感知”思想的核心。
      </p>

      <h3>30 秒逛完 GPU</h3>
      <p>
        接下来的两个版本讲的都是<em>把机器填满</em>，所以你需要一张机器的图。GPU
        不是一颗处理器——它更像一个办公园区。NVIDIA A100 有 <strong>108 个 SM</strong>（streaming
        multiprocessor，流式多处理器）：彼此独立的小处理器，各自带着自己的 SRAM 暂存区（~200 KB）和自己的{' '}
        <strong>Tensor Core</strong>——专用的矩阵乘法单元。工作以<em>线程块（thread block）</em>
        的形式到达——每个块被指派给一个 SM；块内部的线程以 32 个一捆的 <em>warp</em>{' '}
        为单位锁步执行。两条推论与本文相关：没有分到块的 SM <em>什么都不做</em>
        ；而存在多少个块，由内核说了算。逐级缩放看看：
      </p>

      <GpuHierarchyDiagram />

      <h3>FlashAttention-2——让每个核心都忙起来</h3>
      <p>
        FlashAttention-2（2023）一点数学都没改——它改的是工作如何铺满整块芯片。带着办公园区的画面算一笔账。原始内核大致按（batch
        × 注意力头）各启动一个线程块：一条提示词乘 8 个头是 <strong>8 个块</strong>——在 108 个 SM 的 A100 上，约 7%
        的芯片在干活，100 个 SM 黑着灯。v2 的头号修复是把 query 序列<em>也</em>切成块：4,096 token 的提示词按 128
        行一块是 32 个 query 块，于是 8 头 × 32 块 = <strong>256 个块</strong>——绰绰有余，点亮每一个 SM。
      </p>
      <p>
        它还重新划分了块<em>内部</em>的工作：不再让每个 warp 各算每行的一个切片、再经共享内存合并结果，而是让每个 warp
        整体拥有一段 query 行——没有任何跨 warp 的对账。它还削减了非矩阵乘法的算术，因为普通运算比 Tensor Core
        慢得多：滚动输出保持<em>未归一化</em>、只在最后除一次 <code>ℓ</code>
        ，而不是每一步都归一化（按 <MathDisplay inline latex={String.raw`e^{\,m_{i-1}-m_i}`} />{' '}
        做的滚动最大值重缩放每一块仍然发生——只是把对 <code>ℓ</code> 的除法推迟了）；对因果掩码而言（未来 token
        反正不能被注意），注定被完全掩掉的分数块干脆不算——单这一项实测约 1.7–1.8×。合起来：比 v1
        快约一倍，注意力内核最高达到 A100 理论 FP16 峰值的 73%。（这种占用率思路，正是下文解码故事的驱动力。）
      </p>

      <Flash2ParallelismDiagram />

      <h3>FlashAttention-3——重叠与低精度</h3>
      <p>
        FlashAttention-3（2024）为 NVIDIA 的 Hopper GPU（H100）调校，它的起点是硬件里一处悬殊得离谱的失衡。H100 的
        Tensor Core 提供约 <strong>989 TFLOPS</strong> 的 fp16 矩阵乘法——而计算指数（每个 softmax
        都要用）的那些小单元只有约 <strong>3.9 TFLOPS</strong>。注意力来回切换的这两种数学之间，有 <strong>256×</strong>{' '}
        的速度差。如果严格按 matmul → softmax → matmul
        的顺序轮流跑，芯片上最昂贵的硅就要花大量时间，等便宜的那部分干完。
      </p>
      <p>
        FlashAttention-3 的答案是不再轮流。Hopper 允许 warp <em>分工（specialize）</em>：<strong>producer</strong> warp
        只负责指挥 <strong>TMA</strong>（一个专用拷贝引擎）从 HBM 取下一批块，<strong>consumer</strong> warpgroup
        只负责算。然后两个 consumer warpgroup 打<em>乒乓（pingpong）</em>：一个在指数单元上跑自己的 softmax
        时，另一个的矩阵乘法持续喂饱 Tensor Core——然后互换。每个块其实是两次矩阵乘法——先是分数 <code>QKᵀ</code>
        ，然后是 value 混合 <code>P·V</code>——softmax 夹在中间，被重叠起来的正是它们：
      </p>

      <Flash3AsyncDiagram />

      <p>
        第二根杠杆是 <strong>FP8</strong>——它值得放慢看，因为“8 位浮点”听上去人畜无害，直到你看清代价。一个 fp8 数（E4M3
        格式）只有 256 种位模式，最大值 448；fp8 矩阵乘法约比 fp16
        快一倍。问题在于：可表示的值这么少，一切都押在把你的数据映射上去的<em>缩放因子（scale factor）</em>
        上。真实的注意力输入偶尔会有<em>离群值（outlier）</em>
        ——一片小数中的一个巨大值。给整个张量挑一个缩放，那个离群值就把网格拉得太开，所有小值坍缩到寥寥几级上，它们的信息就没了。FlashAttention-3
        两次拯救 fp8：<em>块量化（block quantization）</em>
        给每个小块自己的缩放，一个离群值只会糟蹋自己那个块——<em>不相干处理（incoherent processing）</em>
        则用一个固定的随机旋转去乘 Q 和
        K，在舍入之前把离群值的能量抹匀到每个维度上。在任何舍入之前，这个旋转在数学上是免费的——两边同转，点积不变。fp8
        舍入之后没有什么是严格保持的；旋转的职责是让舍入变得<em>便宜</em>
        ，它省下的误差正是下面这个小部件度量的东西：
      </p>

      <Fp8QuantDiagram />

      <p>
        故事还在继续：<strong>FlashAttention-4</strong>（2026）把同一个不变量带到 NVIDIA 的下一代——Blackwell
        B200——那里矩阵乘法吞吐又差不多翻倍，而芯片的其余部分没有——最高达到 1,613 TFLOPS（71%
        利用率）。我们的深入阅读停在 v3，因为 v4 的技巧是 Blackwell 专属的，但请注意贯穿每个版本的模式：
        <em>算法的不变量从未改变</em>——N×N 矩阵永不触碰慢速内存——每个版本只是把这个不变量重新调校到最新硅片的瓶颈上。
      </p>

      <h2>你浏览器里的 Qwen 用它吗？</h2>
      <p>
        用了一部分——而诚实的答案才是有意思的那个。运行这个模型的 WebGPU 后端<em>确实有</em>一个带上面在线 softmax
        的融合分块 flash-attention 内核，它在 6 个全量注意力层的 <em>prefill</em> 阶段运行。但你在演示里看到的逐 token{' '}
        <em>解码</em>故意<em>不</em>用它：在 0.8B、8 个 query 头的情况下，融合内核只会启动 8 个 GPU
        workgroup，让芯片大部分闲置，所以一个占用率闸门把解码路由回普通的三步路径（matmul → softmax →
        matmul），它能撒出多得多的并行工作，在这里实测快约 90%。而且 24 层中只有 6 层运行 softmax 注意力——其余 18 层是
        GatedDeltaNet。所以：prefill 用 flash，解码用普通路径，皆是有意为之。
      </p>
      <p>在两者之间切换，看分派器把调用路由到真正运行的那个内核：</p>

      <NativeAttentionDiagram />
    </Prose>
  );
}

/** 子章节 4.3——把「memory-bound」从一个词变成一个数字：roofline。 */
export function AttentionRooflineSection() {
  return (
    <Prose>
      <h1>roofline 屋顶线</h1>
      <p className="text-muted-foreground">深入阅读 · 第 4 章 自注意力——给「memory-bound」配上一个真正的数字。</p>
      <p>
        <a href="/zh/chapters/attention/memory-wall">显存带宽墙</a>一直靠一个词——<em>memory-bound</em>
        ——来解释为什么注意力、以及一般意义上的逐 token decode，时间都花在等内存而不是算数上。那是个故事。
        <strong>roofline</strong>
        （屋顶线）把它变成一个可以画出来的数字，而这个数字会精确告诉你：故事从什么时候起不再成立。
      </p>

      <h2>两道天花板，你先撞上哪一道</h2>
      <p>
        GPU 会因为两个彼此独立的原因走到尽头，你先够到哪一个，哪一个就是你的天花板。一个是 <strong>compute</strong>
        ——Tensor Core 每秒能做多少次数学运算（H100 的 fp16 上限约 <strong>989 TFLOPS</strong>）。另一个是{' '}
        <strong>memory bandwidth</strong>——你每秒能从 HBM 里搬出多少字节（同一块 H100：<strong>3.35 TB/s</strong>
        ）。一个封住数学，另一个封住搬运的字节。引擎再快，供油跟不上也没用。
      </p>
      <p>
        到底哪道天花板困住你，取决于<em>工作负载</em>，而不是芯片：它的 <strong>arithmetic intensity</strong>
        （算术强度）——每搬一个字节所做的 FLOPs 工作量，单位是 <code>ops:byte</code>。
      </p>
      <MathDisplay
        latex={String.raw`\text{intensity} = \frac{\text{work (FLOPs)}}{\text{memory traffic (bytes)}}, \qquad \text{ridge} = \frac{\text{peak compute}}{\text{bandwidth}} = \frac{989\ \text{TFLOP/s}}{3.35\ \text{TB/s}} \approx 295\ \text{ops:byte}`}
      />
      <p>
        那道位于约 295 ops:byte 的 <strong>ridge</strong>（脊线）就是收支平衡点。每字节做的运算<em>少于</em>
        295，你会远在下一批字节到来之前就把数学算完——你是 <strong>memory-bound</strong>，在 ridge 左侧干等。做得
        <em>更多</em>，字节到达的速度反而比你能嚼完的还快——你就是 <strong>compute-bound</strong>，在右侧。roofline
        把可达性能对算术强度画出来：一条对角的 bandwidth 上限不断爬升，恰好在 ridge 处与水平的 compute 上限相交。
      </p>

      <h2>prefill 和 decode 各自落在哪</h2>
      <p>
        推理的两个阶段分坐两侧。<strong>prefill</strong> 一次就拿到整条提示词，所以它一趟能处理<em>很多个</em>
        位置，读进来的每个权重都被这些位置共同复用：每字节一座算术大山。而 decode 每一趟只能吐一个
        token。（一趟里装多少个位置是调度的选择，不是固定属性，见{' '}
        <a href="/zh/chapters/kv-cache/batching">chunked prefill</a>；两种情况下它都仍然是 compute-bound。）它稳稳地落在
        ridge 右侧，<strong>compute-bound</strong>。<strong>decode</strong> 在 batch 1 时是另一个极端：为了产出
        <em>一个</em> token，它把整个模型的权重恰好流过芯片一遍，几乎不做什么数学。它跌下最左侧的悬崖，
        <strong>深度 memory-bound</strong>——正是本课程一再撞上的那种「闲着等内存」。
      </p>
      <p>
        拖动 batch 滑块，看 decode 点行进。把 <code>N</code> 个请求批在一起，能让这 <code>N</code>
        个都复用同一次权重读取，于是 intensity 大约按 <code>×N</code> 攀升，点沿对角线向右滑向
        ridge——只有当每字节终于有了足够的数学量去喂饱 FLOPs 时，它才从 memory-bound 翻成 compute-bound：
      </p>

      <RooflineDiagram />

      <p>
        所以「memory-bound」从来不是模型的一个固定属性——它是<em>你如何运行它</em>的属性。一个用户等一个
        token，就坐在悬崖上；一台把数百个用户批在一起的服务器，则沿对角线爬升，直到 compute
        终于变成那个值得优化的东西。这段攀爬正是后面
        <a href="/zh/chapters/kv-cache/batching">batching</a> 子章节的全部主题——roofline 就是它读的那张地图。
      </p>
    </Prose>
  );
}

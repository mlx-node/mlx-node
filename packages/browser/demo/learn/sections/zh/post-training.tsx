// 子章节 15.1（知识蒸馏）正文的简体中文译文，与英文版 ../post-training.tsx
// 平行维护：同名导出 PostTrainingDistillationSection。
// 与英文版一样是纯静态阅读内容（SSR 安全，可被预渲染器直接渲染为静态 HTML）；
// 翻译不改动任何数字、引文或结论，只翻译周围的讲解文字。
//
// 建立在第 15 章对 SFT 的既有框定之上——“同样的下一个 token 损失，换成更小、
// 更精选的数据集”——本节追问的是：这个数据集的标签究竟从哪来。蒸馏的答案：来自
// 一个更大的 teacher 模型，而不只是（或除了）人类标注者。两个时代，均以一手来源
// 为依据：
//   - 经典：Hinton, Vinyals & Dean，《Distilling the Knowledge in a Neural
//     Network》，arXiv:1503.02531（2015）——用温度 T 软化 teacher 的
//     softmax，训练 student 去匹配这个软分布（再加一个较轻的硬标签项）；
//     宝马/垃圾车/胡萝卜的引文，以及 MNIST（146 → 74 个错误）/ 语音
//     （>80% 的 ensemble 增益被迁移）的数字，均直接引自论文。
//   - 现代：DeepSeek-R1，arXiv:2501.12948（2025），附录 F——没有软标签，
//     没有 KL 散度：就是在 (prompt, teacher 生成的 response) 对上做普通 SFT。
//     “we apply only SFT and do not include an RL stage”（逐字引用）。
//     附录 F.1 自己的数字：蒸馏进 Qwen2.5-32B 达到 AIME 72.6，而直接对同等
//     规模的基础模型做 RL、不蒸馏，只有 47.0。
//   - 桥接：Qwen3 技术报告，arXiv:2505.09388，第 4.5 节，“Strong-to-Weak
//     Distillation”——一个官方先例，证明经典与现代两种做法可以在同一条
//     pipeline 里先后共存（先 off-policy SFT，再 on-policy KL 逻辑匹配），
//     但这是针对 QWEN3 的小模型。对 Qwen3.5-0.8B 并未证实：Qwen3.5 的官方
//     材料（HF model card、GitHub README、qwen.ai 博客）都把这一系列的训练
//     归因于预训练加 “Scaled Reinforcement Learning”，从未提到蒸馏——所以结
//     尾部分直接如实说明，而不是类比断言。

import * as React from 'react';

import { Prose } from '../../Prose';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { DistillationPipelines } from '../../widgets/DistillationPipelines';

/** 子章节 15.1——知识蒸馏：由 teacher 而不是人类来写标签的那种 SFT。 */
export function PostTrainingDistillationSection() {
  return (
    <Prose>
      <h1>知识蒸馏</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 15 章 从基础模型到助手——训练一个小模型去模仿一个更大的模型，而不只是（或除了）模仿人类。
      </p>

      <h2>同样的配方，标签换了个来源</h2>
      <p>
        <ChapterLink chapterId="post-training">本章</ChapterLink>
        把指令微调（SFT）讲成了模型一直在用的那个下一个 token 损失，只是把它指向了一小批精心整理的
        <strong>（指令，回复）</strong>对——这些回复通常由人类编写或审核。<strong>知识蒸馏</strong>
        保留这套设置原封不动，只换了一件事：回复不再由人类写出，而是由一个更大、更强的
        <strong>teacher 模型</strong>写出（在下面的经典版本里，甚至由它提供整个目标分布）。
        <strong>student 模型</strong>——真正被训练的那个模型，通常比 teacher 小得多——学的是模仿
        teacher，而不只是（或除了）模仿人类标注者。
      </p>
      <p>
        这仍然完全是<ChapterLink chapterId="training">训练一章</ChapterLink>引入的那个损失：
        <MathDisplay inline latex={String.raw`-\log p(\text{target token})`} />
        ，在每个计分位置上取平均——在初始化时是{' '}
        <MathDisplay inline latex={String.raw`\ln(248{,}320) \approx 12.4`} /> 纳特，也就是在这个模型的词表上均匀乱猜。
        蒸馏从不触碰这个公式，它只改变什么算<em>目标</em>。
      </p>

      <h2>经典配方：软标签与「暗知识」</h2>
      <p>
        这项技术比 LLM 时代要老得多——Geoffrey Hinton、Oriol Vinyals 和 Jeff Dean 在 2015
        年提出了它，起名<strong>蒸馏（distillation）</strong>（建立在 Rich Caruana
        及其合作者更早的一个想法之上，那个想法是直接匹配原始 logits——论文证明这正是 Hinton
        方法在高温极限下的一个特例）。他们的洞察是：一个训练好的分类器，其 softmax
        会给<em>每一个</em>类别都分配一点概率，哪怕是错误的类别，而这些小概率并不是噪声——它们编码了这个模型倾向于如何泛化。他们自己给的例子，逐字引用：
      </p>
      <p>
        <em>
          「例如一张宝马汽车的图片，被误认成垃圾车的概率或许非常小，但这个概率仍然比被误认成胡萝卜的概率要高出许多倍。」
        </em>
        ——Hinton, Vinyals &amp; Dean，《Distilling the Knowledge in a Neural Network》，arXiv:1503.02531，第 1 节
      </p>
      <p>
        一个 one-hot 标签（「这是一辆宝马」）把这种模式彻底丢掉了。要保住它，就用一个
        <strong>temperature</strong> <MathDisplay inline latex={String.raw`T`} /> 去软化 teacher 的 softmax：
      </p>
      <MathDisplay latex={String.raw`q_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}`} />
      <p>
        在 <MathDisplay inline latex={String.raw`T = 1`} /> 时这就是普通 softmax；调高{' '}
        <MathDisplay inline latex={String.raw`T`} />
        会把分布压平，让近似错误（垃圾车）和离谱错误（胡萝卜）彼此分开，而不是都四舍五入到零。随后
        student 会在两个交叉熵的加权组合上训练：一个是在同样高的{' '}
        <MathDisplay inline latex={String.raw`T`} /> 下，对 teacher 的<strong>软标签</strong>；另一个较轻的，是在{' '}
        <MathDisplay inline latex={String.raw`T = 1`} /> 下，对普通的<strong>硬标签</strong>。有一个细节值得记住：因为软标签项的梯度会按{' '}
        <MathDisplay inline latex={String.raw`1/T^2`} /> 缩小，Hinton 的论文把这一项乘以{' '}
        <MathDisplay inline latex={String.raw`T^2`} />，这样它的贡献不会随着 <MathDisplay inline latex={String.raw`T`} /> 调高而悄悄消失。
      </p>
      <p>
        效果并不含糊。在 MNIST 上，一个 800 单元的小网络按普通方式训练会犯 146 个测试错误；同样大小的网络只训练去匹配一个更大的、带
        dropout 正则化的网络给出的软标签（在 <MathDisplay inline latex={String.raw`T = 20`} /> 下），只犯 74
        个——大约减半。在一个商用语音模型上（约 8500 万参数，2000 小时音频），一个 10 模型的
        ensemble 把帧准确率从 58.9% 提到 61.1%；从这个 ensemble 蒸馏出的单个模型达到
        60.8%——论文写道这「转移了超过 80%」的 ensemble 增益，而且只用了一个模型，不是十个。
      </p>

      <h2>现代配方：就是 SFT，只是文本来源换了个更聪明的</h2>
      <p>
        今天规模最大的一次 LLM 蒸馏实践，把软分布那一整套机制全丢掉了。DeepSeek 的 R1
        论文把自己冗长的思维链推理蒸馏进六个小得多的稠密模型（Qwen2.5-Math-1.5B/7B、Qwen2.5-14B/32B、Llama-3.1-8B、Llama-3.3-70B），用的是
        800,000 条 (prompt, response) 对，其中每一条 response 都由 DeepSeek-R1
        自己生成（大约 60 万条推理轨迹，加 20 万条通用示例）。论文附录里逐字写的方法：
        <em>「we apply only SFT and do not include an RL stage」</em>
        （我们只应用 SFT，不包含 RL 阶段）。完全没有 temperature，没有 KL 散度，也没有任何 teacher
        logits——teacher 生成的文本就被当成一个普通的 SFT 目标来处理，正是本页最开始讲的那套配方。
      </p>
      <p>
        这个头条结果想说的是：蒸馏不只是比从零训练一个小型推理模型更便宜——它还可能<em>更好</em>。直接对一个
        Qwen2.5-32B 基础模型做强化学习、完全不蒸馏（「Qwen2.5-32B-Zero」），AIME 2024 pass@1 达到
        47.0。而只是在同等规模的基础模型上，对 DeepSeek-R1 生成的回复做微调——纯 SFT，完全没有
        RL——达到 72.6。就连蒸馏出来最小的那个模型 DeepSeek-R1-Distill-Qwen-1.5B，在这项基准上都超过了
        GPT-4o 和 Claude-3.5-Sonnet。把两个时代放在一起比一比：
      </p>

      <DistillationPipelines />

      <h2>两者并不互斥：Qwen 自己的 strong-to-weak 蒸馏</h2>
      <p>
        上面这两套配方不是竞争关系——Qwen 自己（上一代）的 Qwen3 技术报告记录了一条先后用上两者的
        pipeline，用在 Qwen3 的小型稠密模型和 MoE 模型上。先是一个 <strong>off-policy</strong>
        阶段：对一个更大的 teacher（Qwen3-32B 或 Qwen3-235B-A22B）生成的回复做普通 SFT——就是上面的现代配方。然后是一个
        <strong>on-policy</strong> 阶段：小的 student 模型生成自己的 rollout，再被微调去最小化它自己的
        logits 与 teacher 的 logits 之间的 <strong>KL 散度</strong>——这字面上就是 Hinton 2015
        年的方法，只是跑在 student 自己采样的样本上，而不是一个固定的数据集。Qwen 报告说这条路线达到了比他们完整的四阶段训练
        pipeline 更高的基准分数，同时「只用了 1/10 的 GPU 时数」。这是一个真实、有据可查的例子，正好把本页刚走过的经典
        vs 现代这条分界线，共存在了同一次训练里。
      </p>

      <h2>你浏览器里的 Qwen 是这么来的吗？</h2>
      <p>
        未经证实——与其按类比默认成立，不如老实说清楚。上一段讲的是 Qwen3，也就是<em>上一代</em>。而
        Qwen3.5——本课程实际教授、包括这个标签页里正在跑的 0.8B 模型在内的、基于 Qwen3-Next
        的这一系列——没有任何官方来源提到过蒸馏。Qwen3.5-0.8B 的 model card、QwenLM/Qwen3.5 的 GitHub
        README，以及 qwen.ai 的旗舰发布博客，都把这一系列的训练描述成预训练之后接
        <strong>「Scaled Reinforcement Learning」</strong>——在大量 agent 环境上跑强化学习——针对
        Qwen3.5，哪里都没提到 teacher 模型、「strong-to-weak」pipeline，或者蒸馏。所以诚实的答案是：Qwen
        确实公开记录过，在它<em>上一代</em>的小模型上做过正是本页描述的这件事。而这套手法、或它的某个变体，是否也碰过这个具体的
        0.8B checkpoint，没有任何公开来源可以证实——所以本课程不会这么断言。
      </p>
    </Prose>
  );
}

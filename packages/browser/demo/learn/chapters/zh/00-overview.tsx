import { ChapterLink } from '../../scaffolding/ChapterLink';

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { GenerationLoop } from '../../widgets/GenerationLoop';
import { LineOfBestFit } from '../../widgets/LineOfBestFit';
import { LogitsToSoftmax } from '../../widgets/LogitsToSoftmax';
import { TokenJourney } from '../../widgets/TokenJourney';
import { WeightsVsActivations } from '../../widgets/WeightsVsActivations';

/**
 * 第 1 章（overview）的简体中文翻译 — 与 ../00-overview.tsx 平行维护。
 * 仅翻译正文与 learning 数据；小部件共享英文版实现。
 */

const VOCAB = 248_320; // Qwen3.5-0.8B vocab_size — matches sibling chapters

export const learning: ChapterLearningData = {
  chapterId: 'overview',
  objective: '用一句话说出 LLM 计算的是什么——并追踪这一个计算如何在循环中跑出整段文本。',
  problem: '后面的每一章都聚焦于某一个组件。不先建立全局图景，这些机器零件就无处安放。',
  minutes: 6,
  glossary: [
    {
      term: 'language model',
      definition: '一个函数：给定到目前为止的 token，为每个可能的下一个 token 的可能性打分。',
    },
    {
      term: 'token',
      definition: 'LLM 读写的基本单位——一个子词片段，既不是字符，也不是完整的词。',
    },
    {
      term: 'logits',
      definition: '模型输出的原始无界分数——词表中每个条目各一个——softmax 把它们变成概率之前的样子。',
    },
    {
      term: 'softmax',
      definition:
        '把每个 logit 取指数，再除以它们的总和：e^(z_i) / Σ_j e^(z_j)。将一组原始分数变成总和为 1 的概率分布。',
    },
    {
      term: 'forward pass',
      definition: '模型在一个 token 序列上的一次完整计算，产生下一个位置的 logits。',
    },
    {
      term: 'weights',
      definition:
        '模型存储的 852,985,920 个数字（权重）——训练时定下，之后冻结。对每个请求都相同；它们就是模型本身。',
    },
    {
      term: 'activations',
      definition: '前向传播中为你的提示词现场算出的中间向量（激活值），这趟计算一结束就被丢弃。',
    },
    {
      term: 'autoregressive',
      definition: '一次生成一个 token，把每个输出再喂回去，作为下一步的输入。',
    },
    {
      term: 'sampling',
      definition: '按 softmax 概率加权，随机挑出下一个 token。“贪心”（greedy）解码则永远直接取分数最高的那一个。',
    },
    {
      term: 'generation loop',
      definition: '前向传播 → 采样一个 token → 追加 → 重复，直到遇到停止 token 或达到长度上限。',
    },
  ],
  takeaways: [
    'LLM 就是一个函数：输入一串 token，输出词表中每个可能的下一个 token 的分数。',
    '文本是一次一个 token 生成的——同一个前向传播在循环里反复运行，每一轮多看到一个 token。',
    '推理阶段的全部工作就是“预测下一个 token”；能力是这个目标逼着模型学会的东西。',
  ],
  exercise: {
    prompt: '在生成循环的动画里观察伪代码。当一个新 token 加入 token 条时，恰好是哪一行被高亮？',
    answer:
      'tokens.push(next)——追加这一步。token 条只有在前向传播采样出一个 token、并把它 push 进列表之后才会变长；随后循环在变长了的序列上再跑一轮。',
  },
  quiz: [
    {
      id: 'q1-one-pass',
      prompt: '从机制上讲，LLM 在一次前向传播中计算的是什么？',
      options: [
        { id: 'a', label: '它最终会写出的整句话。' },
        { id: 'b', label: '为词表中的每个 token 给出一个分数（logit），只针对下一个位置。' },
        { id: 'c', label: '对“目前为止的文本是否合法”的一个是/否判断。' },
      ],
      correctId: 'b',
      explanation: '一次前向传播只产出针对下一个 token 的、覆盖整个词表的一个分布。更长的输出来自把这个过程放进循环里跑。',
    },
    {
      id: 'q2-loop',
      prompt: 'LLM 是怎么写出一整段话，而不是只有一个 token 的？',
      options: [
        { id: 'a', label: '它先规划好整段话，然后一次性写出来。' },
        { id: 'b', label: '它把同一个前向传播放进循环，每次采样一个 token 并追加进去。' },
        { id: 'c', label: '它从训练数据里检索最接近的一段话。' },
      ],
      correctId: 'b',
      explanation: '生成是自回归的：预测一个 token，追加，再在变长了的序列上跑一遍。',
    },
    {
      id: 'q3-logits',
      prompt: '什么是 “logits”？',
      options: [
        { id: 'a', label: '已经归一化、总和为 1 的概率。' },
        { id: 'b', label: '原始的无界分数——词表中每个 token 各一个——尚未经过 softmax 归一化。' },
        { id: 'c', label: '分词器产出的 token id。' },
      ],
      correctId: 'b',
      explanation: 'softmax 把 logits 变成概率分布；原始 logits 本身还没有被归一化。',
    },
  ],
};

export function OverviewChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>什么是大语言模型？</h1>
        <p>
          剥去神秘的外衣，大语言模型其实就是一个函数。你交给它一串 token——也就是到目前为止的文本——它会为词表中的
          <em>每一个</em> token 返回一个分数：对“接下来是什么”的猜测。这就是全部计算。本课程余下的内容，要么讲这个函数是怎么构建的，要么讲它是怎么被使用的。
        </p>

        <h2>一次调用：输入 token，为每个候选的下一个 token 打分</h2>
        <p>
          具体来说，输入只是一个整数列表——每个 token 对应一个 id，由{' '}
          <ChapterLink chapterId="tokenization">
            分词器
          </ChapterLink>{' '}
          产生——输出则是一个很大的小数数组，词表中每个词对应一个分数：
        </p>
        <pre>
          <code>{`tokens: number[]          // a list of integers, one id per token,
                          // e.g. [760, 7993, 7338, 383, 279] = "The cat sat on the"
   │
   ▼  one forward pass
logits: Float32Array(${VOCAB.toLocaleString()})   // an array of decimals — one raw score per vocab word`}</code>
        </pre>
        <p>
          这些原始分数就叫 <strong>logits</strong>。把它们过一遍 softmax（归一化指数函数），就得到“接下来是什么”的概率分布——对
          Qwen3.5-0.8B 来说，是覆盖它认识的全部 <code>{VOCAB.toLocaleString()}</code> 个 token
          的分布。模型从不直接输出某个词；它为每个<em>可能</em>输出的词各打一个分，再由一条单独的规则挑出一个。
        </p>
        <p>
          那个“过一遍 softmax”的步骤小到可以手算——下面就在一个只有五个 token
          的玩具词表上演示，让你在把规模扩大到全部 <code>{VOCAB.toLocaleString()}</code>{' '}
          个之前，先看清原始分数是怎么变成概率的：
        </p>
        <LogitsToSoftmax />
        <p>
          <em>Sharpen ↔ flatten</em>（锐化 ↔ 压平）滑块会在 softmax 之前对原始 logits
          做缩放：往上拖，模型对自己的第一候选押得更狠（分布出现尖峰）；往下拖，概率在各个选项之间摊得更平。这正是生成文本时{' '}
          <ChapterLink chapterId="sampling">
            <strong>temperature</strong>
          </ChapterLink>{' '}
          （温度）所调的那个旋钮——在这里它表现为温度的倒数（数值越高分布越尖）。
        </p>

        <h2>两类数字：权重 vs 激活值</h2>
        <p>
          再往下走之前，先把这里出现的数字分进两个桶——分清楚了，后面的一切都会顺理成章。第一个桶是<strong>权重</strong>
          （weights）：那 852,985,920 个存下来的数字，它们<em>就是</em> Qwen3.5-0.8B
          本身。它们在训练时被定下，从那以后一直冻结——推理时只读，而且无论是你的提示词还是别人的，它们都逐位相同。第二个桶是
          <strong>激活值</strong>（activations）：前向传播<em>专为你这串 token</em>
          算出的中间向量，随着数据流过各层而产生。它们在你的请求到来时诞生，在这趟计算结束的那一刻就被丢掉。（有一小片会跨循环步幸存——{' '}
          <ChapterLink chapterId="kv-cache">KV 缓存</ChapterLink>
          ——但那只是对话内部的一项优化，不是对你的记忆。）
        </p>
        <WeightsVsActivations />
        <p>
          这个划分也是整门课的地图：<em>训练</em>（{' '}
          <ChapterLink chapterId="training">训练一章</ChapterLink>
          ）是改写左边那个桶的过程；而<em>推理</em>——本课程接下来做的一切——永远只是往右边那个桶里填数、再倒空。它还解释了一个常让人意外的事实：模型对每个用户都是同一件制品，调用与调用之间什么也不记得，因为你的提示词算出的任何东西都不会被写回权重。
        </p>

        <h2>这些数字从哪来？是“拟合”出来的，不是写出来的</h2>
        <p>
          那么，852,985,920 个具体的数字是从哪来的？不是谁一个个写出来的。传统软件是程序员亲手写明的规则：<em>如果邮件里出现“免费领钱”，就标记为垃圾邮件。</em>
          每一种行为都是某个人想清楚、再敲出来的一行代码。而大语言模型恰恰是反着造的。没有谁能手写出“任意句子里的下一个词是什么”这样一条规则——于是改用另一种办法：先把这些数字随机初始化，再给模型看海量真实文本，反复微调这些数字，让它的猜测一点点逼近真正出现的下一个词。行为从来不是被写明的；它是被<em>拟合</em>到样例上的。（这个拟合过程——{' '}
          <ChapterLink chapterId="training">训练一章</ChapterLink>
          ——对模型里的每一个数字都是同一套思路，无论它落在哪一种层里。）
        </p>
        <p>
          这件事最小的、能装进脑子里的版本，就是一条直线。直线 <code>y = a·x + b</code> 恰好有两个旋钮：斜率{' '}
          <code>a</code> 和截距 <code>b</code>。给它一些散落的样例点，“训练”无非就是拧动这两个旋钮，直到这条线尽可能贴近这些点：
        </p>
        <LineOfBestFit />
        <p>
          这就是全部的把戏，只是放大到了几乎难以置信的程度。一条直线有 2 个旋钮，Qwen3.5-0.8B 有{' '}
          <strong>852,985,920</strong> 个——大约是它的 <strong>4.26 亿</strong> 倍（852,985,920 ÷ 2 ={' '}
          426,492,960）。同样的动作，只是旋钮多得无法想象：不再是用一个斜率和一个截距去弯折一条线，而是去拟合数以亿计的数字，让那一个庞大的函数尽可能贴近“人类真正写出的下一个
          token”——覆盖人们几乎所有写下来的东西。
        </p>

        <h2>生成循环</h2>
        <p>
          一次前向传播只给出一个 token 的预测。要写出一句话，就得循环调用这个函数——采样出一个
          token，追加到末尾，再对这串稍微变长的列表跑一遍。（“采样”的意思就是：按概率加权随机挑一个
          token——概率最高的那个通常会赢，但不是每次都赢。“贪心”模式则跳过掷骰子，永远拿第一名。）
        </p>
        <GenerationLoop />
        <p>
          这就是“生成文本”的含义：它是<strong>自回归</strong>（autoregressive）的。每个新 token
          都来自把整个模型在比上次长一个 token 的序列上重新跑一遍。（每一步都重跑全部计算听起来很浪费——{' '}
          <ChapterLink chapterId="kv-cache">
            KV 缓存一章
          </ChapterLink>{' '}
          会讲它是怎么被做便宜的。）
        </p>

        <h2>一口气看完整条流水线</h2>
        <p>
          在那一次前向传播内部，一个 token 的旅程是：文本 → <strong>token</strong> → <strong>向量</strong>
          （嵌入）→ 一摞很深的<strong>注意力 + MLP</strong> 块 → 为每个 token 给出的<strong>最终分数</strong>（LM
          head）→ <strong>采样</strong>出一个 → 追加，然后循环。（这摞“深堆叠”是混合的：24 层中有 6
          层用的是注意力一章要讲的注意力；另外 18 层用一种更省算力的捷径，在 KV
          缓存一章中介绍。）后面的每一章都会打开其中一个盒子；{' '}
          <ChapterLink chapterId="architecture">
            架构一章
          </ChapterLink>{' '}
          再把它们拼回到同一页上。
        </p>
        <p>
          更妙的是，下面就是这趟旅程的可<em>逐步播放</em>版本——一个具体的 token，从原始文本一路走到下一个词，每一步都标出数据的形状：
        </p>
        <TokenJourney />

        <h2>为什么“预测下一个 token”就够了</h2>
        <p>
          这听起来简单得近乎无趣。关键在于它被要求达到的标准：要在全人类的文本上——文章、代码、对话、翻译、算术——把下一个
          token 预测<em>好</em>，模型就不得不内化语法、事实和推理模式，因为正是这些东西让下一个 token
          变得可预测。能力是把一个狭窄目标做到极致的副产品。
        </p>
        <p>
          先把一句实话说在前面：我们刚才描述的是<em>基础模型</em>——纯粹的自动补全。从它跳到一个会听从你指令的
          ChatGPT 式助手，是另一段<em>训练</em>的故事，留到后面的{' '}
          <ChapterLink chapterId="post-training">
            “从基础模型到助手”
          </ChapterLink>
          一章再讲。
        </p>
        <p>
          第二句实话：这种逐 token 的猜测读起来有多<em>连贯</em>，很大程度上取决于规模。要拟合的旋钮越多，越多的语法、事实和推理模式才能被捕捉进去——所以更大的模型往往能在更长的篇幅里不跑题。本课程里跑的这个模型是
          <strong>刻意做小的——0.85B 参数，专门挑成能完全跑在你浏览器标签页里的大小。</strong>
          它不是任何模型的缩水版或过时版：它就是 Qwen3.5-0.8B，发布于 2026 年，24 层里混合了 6 个 full-attention
          层和 18 个更省算力的 <code>GatedDeltaNet</code> 层。但在这个尺寸下、又没有经过指令微调，它会<em>跑偏</em>是意料之中的——开头一句还像样，接着就飘到某个奇怪的地方去了。给你一个它所处量级的感觉：这个模型约
          0.85B 参数，一个小型开源模型大概 ~1.5B，而最初的 GPT-3 约为 ~175B——比这里跑的这个模型又大了大约 200 倍（不是比那个 1.5B 的模型）。你可以亲眼看着我们这个
          0.85B 模型跑偏，并用{' '}
          <ChapterLink chapterId="sampling">
            采样一章
          </ChapterLink>
          里的温度控件去调节它跑得有多野。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

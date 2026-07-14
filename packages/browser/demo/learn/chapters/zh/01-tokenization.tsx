import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { BpeMergeWalkthrough } from '../../widgets/BpeMergeWalkthrough';
import { MultilingualTokenMap } from '../../widgets/MultilingualTokenMap';
import { TrickyCases } from '../../widgets/TrickyCases';

/**
 * 第 2 章（tokenization）的简体中文翻译 — 与 ../01-tokenization.tsx 平行维护。
 * 仅翻译正文与 learning 数据；`TokenizerDemo` 等演示组件保留在英文模块中，
 * 由英文模块渲染（演示界面保持英文）。
 */

export const learning: ChapterLearningData = {
  chapterId: 'tokenization',
  objective: '解释 LLM 为什么把文本切成子词 token，而不是字符或完整的词。',
  problem: '模型需要一个有限的词表，在序列长度与词表大小之间取得平衡。',
  minutes: 7,
  glossary: [
    {
      term: 'token',
      definition: 'Transformer 真正读取的单位——一个指向模型词表的整数 id。',
    },
    {
      term: 'BPE',
      definition:
        'Byte-Pair Encoding（字节对编码）：从字节出发，反复合并出现频率最高的相邻片段对，直到词表达到目标大小。',
    },
    {
      term: 'sub-word',
      definition: '比词小、比字符大的 token，例如 "ization" 或 " the"。',
    },
    {
      term: 'vocabulary',
      definition: '模型认识的全部 token 的集合。Qwen3.5 自带约 248,320 个条目。',
    },
    {
      term: 'special token',
      definition: '像 <|im_start|> 这样的保留 token，用来标记结构（对话轮次、文本结束），而不是字面文本。',
    },
    {
      term: 'chars/token',
      definition: '字符数除以 token 数——衡量一段字符串与分词器词表契合程度的快捷指标。',
    },
  ],
  takeaways: [
    '成本随 token 数而非字符数增长：计费、KV 缓存、延迟都是按 token 计的。',
    '前导空格是 token 的一部分——"cat" 和 " cat" 通常是两个不同的词表条目。',
    '不同模型家族对同一字符串的分词结果不同，所以 token 数只有在同一个分词器内才可比。',
  ],
  exercise: {
    prompt: '用分词器演示找一句话，让 chars/token 比值尽可能高；再找另一句，让它尽可能低。',
    answer:
      '很长的英文单词（比如 "unbelievably"、"internationalization"）比值非常高，因为 BPE 会把常见的长词干合并成单个 token。低的一端是 emoji、CJK 标点或罕见文字——它们会退化成字节级碎片，比值经常掉到 1 char/token 以下。',
  },
  quiz: [
    {
      id: 'q1-why-subword',
      prompt: '现代 LLM 为什么用子词 token，而不是完整的词？',
      options: [
        { id: 'a', label: '子词总是比词短。' },
        {
          id: 'b',
          label:
            '词级词表必须穷举每一个名字、错别字和生造词，会让嵌入矩阵（按 token 查询的查找表——下一章讲）大到不切实际。',
        },
        {
          id: 'c',
          label: '词级分词器无法表示标点，所以根本没法用。',
        },
      ],
      correctId: 'b',
      explanation:
        '词表大小必须有限。按词切意味着数百万个条目（长尾）；按字符切意味着序列变得非常长。子词是实用的折中。',
    },
    {
      id: 'q2-chars-per-token',
      prompt: '更高的 chars/token 比值通常说明什么？',
      options: [
        {
          id: 'a',
          label: '这段字符串用的是分词器学得很好的常见模式，所以每个 token 覆盖更多字符。',
        },
        {
          id: 'b',
          label: '模型在这种输入上更容易产生幻觉。',
        },
        {
          id: 'c',
          label: '这段文本使用的是模型无法理解的语言。',
        },
      ],
      correctId: 'a',
      explanation: '更高的 chars/token 意味着分词器学到的合并规则与这段文本契合得很好——通常是覆盖充分的语言里的散文。',
    },
    {
      id: 'q3-leading-space',
      prompt: '句首的 "cat" 和句中的 " cat"，它们的 token id 相同吗？',
      options: [
        { id: 'a', label: '相同——BPE 在编码前会去掉空白。' },
        {
          id: 'b',
          label: '不同——BPE 把前导空格当作 token 的一部分，所以它们通常是两个不同的词表条目。',
        },
      ],
      correctId: 'b',
      explanation: 'BPE 在包含空格的原始字节流上工作，所以带空格前缀的词和裸词是两个不同的符号。',
    },
  ],
};

export function TokenizationChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>分词（Tokenization）：把文本变成模型的词表</h1>
        <p>
          在 Transformer 能对你的句子做任何事情之前，句子必须先被切成模型认识的片段。这些片段叫作{' '}
          <strong>token</strong>（词元），而决定从哪里下刀的规则叫作<em>分词器</em>。token
          既不是字符，也不是词——它介于两者之间，由一种叫<strong>字节对编码（Byte-Pair Encoding，BPE）</strong>
          的算法选出。这里的小部件在你的浏览器里运行 Qwen3.5 的原版分词器。敲几个字，看色块实时刷新。
        </p>

        <h2>为什么是子词？</h2>
        <p>
          分词器处在一个尴尬的中间地带。如果让每个<em>词</em>都当一个
          token，词表得有数百万个条目才能覆盖名字、错别字和生造词的长尾——嵌入矩阵（为每个词表条目存一个向量的大查找表——下一章讲）也会跟着膨胀。如果反过来让每个
          <em>字符</em>当一个 token，模型就得啃比按词计长五到十倍的序列，而每一层的开销至少与序列长度成线性关系。
        </p>
        <p>
          一句固定的话能把这个权衡讲得很具体。拿{' '}
          <code>"The quick brown fox jumps over the lazy dog"</code> 来说：按字符是 <strong>43</strong> 个
          token；按整词是 <strong>9</strong> 个；子词分词器恰好也是 <strong>9</strong>{' '}
          个——和按词一样短，但用的是一个永远不会“缺词”的词表。
        </p>
        <p>
          BPE 找到了一条中间路线：从原始字节（或字符）出发，反复把出现频率最高的相邻片段对合并成一个新符号，直到词表达到你想要的大小。像{' '}
          <code>the</code> 这样的常见词最终成为单个 token；像 <code>Antidisestablishmentarianism</code>{' '}
          这样的罕见词则分解成若干更短的熟悉片段。分词器不懂词义——它只知道哪些字节序列在训练语料中统计上更常见。由于基础字母表就是
          256 种可能的字节值，分词器可以表示<em>任何</em>输入——任何 Unicode 字符、emoji
          或文字，哪怕训练时从未见过——因此不存在“未知”或词表外的 token；陌生文本只是被切成更多、更小的（字节级）token。
        </p>

        <h2>Qwen3.5 的具体做法</h2>
        <p>
          Qwen3.5 自带一个约 <strong>248,320</strong> 个条目的 BPE 分词器。有几个细节值得了解：
        </p>
        <ul>
          <li>
            <strong>空格住在 token 里面。</strong>BPE 算法把带空格前缀的词和裸词当作两个不同的符号，所以句首的{' '}
            <code>"cat"</code> 和句中的 <code>" cat"</code>{' '}
            通常是两个不同的 token。这里的色块用一个小圆点（<code>·</code>
            ）标出前导空格，让你能看见这件原本不可见的事。
          </li>
          <li>
            <strong>特殊 token 标记结构。</strong>像 <code>&lt;|im_start|&gt;</code> 和{' '}
            <code>&lt;|im_end|&gt;</code> 这样的序列不是模型要生成的文本——它们是 Qwen3.5
            对话模板使用的轮次边界。只要某个 token 解码后的文本里含有 <code>&lt;|…|&gt;</code>
            ，小部件就会给它画上虚线边框，方便你认出来。
          </li>
          <li>
            <strong>多语言覆盖并不均匀。</strong>拉丁字母的词通常是一两个 token；CJK 字符往往一字一个
            token，但罕见文字可能分解成字节级碎片。试试多语言示例，体会其中的差别。
          </li>
        </ul>

        <h2>为什么这件事重要</h2>
        <p>
          语言模型的每一项下游成本都是<em>按 token</em> 支付的：API 计费、上下文窗口上限（模型一次能容纳多少
          token 的限制）、KV 缓存内存（生成期间为每个 token
          保留的工作记忆——后面章节讲），以及生成的真实耗时。一段在人看来很短的提示词，如果用了罕见标点、代码，或分词器训练时见得不多的语言，对模型来说可能很长。统计栏里的
          <strong>每 token 字符数（chars/token）</strong>是感受这一点的最简单方式：覆盖良好的语言里的散文通常落在
          4 chars/token 附近，代码接近 2-3，而满是 emoji 或罕见文字的字符串可能掉到 1 以下。
        </p>
        <p>
          不同模型家族使用不同的分词器，同一字符串可能切出长度天差地别的结果。这就是为什么 LLM
          的成本和容量都用 “token” 而不是“字符”或“词”来谈。
        </p>

        <h2>看着 BPE 拼出一个 token</h2>
        <p>
          下面的小部件在输入 <code>"unbelievable"</code> 上演示一个风格化的 BPE
          过程。每一步把相邻的一对片段合并成一个新符号——BPE
          训练时正是对语料库把同样的操作做上千百万次，从而学出它的合并顺序。
        </p>

        <BpeMergeWalkthrough />

        <h2>值得亲眼一看的分词怪癖</h2>
        <p>
          初学者在 LLM 成本或行为上遇到的大多数意外，都能追溯到下面这几条。每一行展示一个小输入，以及它通常分解成的
          token。
        </p>

        <TrickyCases />

        <MultilingualTokenMap />

        <p className="mt-6 text-muted-foreground">
          下一章（<em>词嵌入</em>）我们会看到这些 token id
          如何各自变成一个高维向量，让模型真正能拿来计算——这是从整数通往注意力所在的连续空间的桥梁。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

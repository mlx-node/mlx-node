// 中文（简体）版第 10 章正文 — LM head 与权重共享。
// Demo 组件保持英文，仍从英文模块 ../09-lm-head.tsx 渲染；本文件只翻译
// 正文 Body 与 learning 数据。术语表的 term 字段保持英文原样。

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { InnerProductLogit } from '../../widgets/InnerProductLogit';
import { LmHeadWalkthrough } from '../../widgets/LmHeadWalkthrough';
import { TiedUntiedLedger } from '../../widgets/TiedUntiedLedger';
import { WeightTyingVisual } from '../../widgets/WeightTyingVisual';

const TOP_K = 12;

export const learning: ChapterLearningData = {
  chapterId: 'lm-head',
  objective:
    '解释最后的隐藏状态如何变成一个 logits 向量，以及为什么对 Qwen3.5 来说，LM head 实际上就是被复用的嵌入矩阵。',
  problem: '第 9 章结束于一个隐藏状态；第 11 章开始于 logits。中间必须有一步把前者变成后者。',
  minutes: 6,
  glossary: [
    {
      term: 'LM head',
      definition:
        '模型最后的线性投影。把最后的隐藏状态乘以一个 [d, V] 矩阵，为词表中的每个 token 得到一个 logit。',
    },
    {
      term: 'logit',
      definition:
        '词表中每个条目对应的一个无界实数分数——越高表示模型越偏好该 token。它们还不是概率；采样一章会用 softmax 把它们变成概率。',
    },
    {
      term: 'weight tying',
      definition:
        '把 embed_tokens.weight 复用为 LM head，让输入查表和输出投影共享同一个张量（在小模型上能省下相当大比例的参数）。',
    },
    {
      term: 'vocabulary size (V)',
      definition: '分词器能产出的不同 token 的数量。Qwen3.5 使用 V = 248,320。',
    },
    {
      term: 'hidden dim (d)',
      definition: '残差流的宽度。Qwen3.5-0.8B 使用 d = 1024——每个 token 在每一层都携带 1024 个浮点数。',
    },
    {
      term: 'inner product',
      definition:
        '逐元素乘积之和。logit_j = <last_hidden, W[j, :]>：模型给 token j 的分数，就是隐藏状态与该 token 在 LM head 中那一行的对齐程度。',
    },
  ],
  takeaways: [
    'LM head 只是一次矩阵乘法：last_hidden @ embed_tokens.T → logits。词表中每个条目得到一个分数。',
    '权重矩阵的每一行是某个词表 token 学到的“指纹”；logit 最高的那个，就是与隐藏状态最对齐的那个。',
    'Qwen3.5 把 embed_tokens.weight 与 lm_head.weight 绑定——同一个 ~254M 浮点数的张量在输入查表和输出投影两处使用。',
  ],
  exercise: {
    prompt:
      "对 'The cat sat on the' 点击 Run，查看 top-K logits 面板。最高的柱子就是模型的选择。现在在 top-K 靠后的位置中找一个在你看来\"不像合理续写\"的 token（例如像 ' his' 这样的代词混进了名词的位置）。它出现在 top-K 里，说明 LM head 的几何结构是什么样的？",
    answer:
      '即便是你认为“不太可能”的 token 也会进入 top-K，因为它们在 LM head 中的行向量与隐藏状态指向大致相同的方向。模型对"`the` 之后该接什么"的表征是宽泛的——任何可坐的、可命名的、像“物体所在位置”的东西在几何上都很接近。采样中的 temperature 和 top-p 才是把这团宽泛的云修剪成一个具体选择的工具。',
  },
  quiz: [
    {
      id: 'q1-shape',
      prompt: 'Qwen3.5-0.8B 的 LM head 矩阵的形状是什么？',
      options: [
        { id: 'a', label: '[1024, 1024]——每个 token 一个隐藏向量。' },
        { id: 'b', label: '[248320, 1024]——每个词表 token 一行，宽度等于隐藏维度。' },
        { id: 'c', label: '[248320, 248320]——一个词表 × 词表的转移矩阵。' },
      ],
      correctId: 'b',
      explanation:
        '矩阵的每一行对应一个词表条目，每一列对应一个隐藏维度；与 [1, 1024] 的隐藏状态相乘，得到 [1, 248320] 的 logit 向量。',
    },
    {
      id: 'q2-tying',
      prompt: "Qwen3.5 配置中的 'tie_word_embeddings = true' 是什么意思？",
      options: [
        {
          id: 'a',
          label: '嵌入值被钳制到 [-1, 1] 以保持稳定。',
        },
        {
          id: 'b',
          label: '同一个张量同时用于输入嵌入查表和输出 LM head——不再单独分配 lm_head.weight。',
        },
        {
          id: 'c',
          label: '模型只能输出它之前见过的 token。',
        },
      ],
      correctId: 'b',
      explanation:
        '启用绑定后，前向传播在底部读取 `embed_tokens.weight` 完成 token id → 向量，在顶部复用它的转置完成向量 → logits。同样的 ~254M 个浮点数，用了两次。',
    },
    {
      id: 'q3-direction',
      prompt: '关于 LM head 各行的说法，哪一项是对的？',
      options: [
        {
          id: 'a',
          label: '每一行是历史上所有曾产生该 token 的提示词的平均隐藏状态。',
        },
        {
          id: 'b',
          label: '每一行是一个学到的向量——token j 的 logit 就是该行与最后隐藏状态的内积。',
        },
        {
          id: 'c',
          label: '每一行是对应某个词表 id 的 one-hot 向量。',
        },
      ],
      correctId: 'b',
      explanation:
        '把每一行想成与某个词表 token 关联的、学到的“隐藏空间中的方向”。logit 就是当前隐藏状态与该方向的对齐程度。',
    },
  ],
};

export function LmHeadChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>LM head：隐藏状态 → logits</h1>
        <p>
          第 9 章把我们留在残差流的顶端：经过最后的 RMSNorm 之后，每个 token 一个宽度为 <code>d = 1024</code>{' '}
          的隐藏向量。第 11 章（采样）将从一个 <strong>logits</strong> 向量出发——词表中每个 token
          一个实数分数。连接两者的步骤就是一次矩阵乘法，它有自己的名字：<strong>LM head</strong>。
        </p>

        <h2>一行数学</h2>
        <MathDisplay latex={String.raw`\text{logits} = h_{\text{last}} \cdot W_{\text{lm}}^\top`} />
        <p>
          <code>h_last</code> 是最后一个 token 的隐藏状态——当模型逐个 token 生成时（这一步叫<em>解码</em>
          ，decode——后面的章节会讲），形状为 <code>[1, 1024]</code>。<code>W_lm</code> 是 LM head 的权重矩阵——对
          Qwen3.5-0.8B 来说形状为 <code>[V, d] = [248,320, 1024]</code>。转置后变成 <code>[1024, 248,320]</code>
          ；相乘得到形状为 <code>[1, 248,320]</code> 的输出——词表中每个 token 一个 logit。
        </p>
        <p>
          有个问题值得停下来想一想：层堆叠为提示词里的<em>每个</em> token 都产出了隐藏向量，为什么只有最后一个进入 LM
          head？因为每个位置预测的是<em>它自己的</em>下一个 token——而模型在生成时，唯一需要的预测就是最后那个 token
          之后的那个。其他位置的预测并没有浪费：训练会一次性给它们全部打分，训练一章会展示这一点。
        </p>
        <p>
          动画逐格演示这次矩阵乘法。扫描光束一次高亮一个输出列，旁边同时点亮产生它的矩阵列。每个输出条目都是一次独立的内积——这也是这个运算能在
          GPU 上如此干净地并行化的原因。矩阵以转置形式绘制，所以每个 token 的指纹显示为一<em>列</em>
          ——光束下方的标签标出当前正在打分的是哪个 token 的列。
        </p>

        <LmHeadWalkthrough />

        <h2>
          每个输出单元格<em>意味着</em>什么
        </h2>
        <p>
          把 <code>W_lm</code> 的每一<strong>行</strong>想成某个词表 token 学到的“指纹”。token <code>j</code> 的
          logit 就是 <code>h_last</code> 与该行的内积：
        </p>
        <MathDisplay latex={String.raw`\text{logit}_j = \langle h_{\text{last}},\, W_{\text{lm}}[j, :] \rangle`} />
        <p>
          所以，<em>logit 高 ⇔ 隐藏状态与该 token 的行指向大致相同的方向</em>。决定下一个 token
          分布的是隐藏状态的几何结构，而 LM head 的各行就是模型用来读出这份几何结构的字典。
        </p>
        <p>
          内积可以展开为 <code>h · w = |h|·|w|·cos θ</code>，其中 <code>cos θ</code> 是两个向量夹角的<em>余弦</em>
          ——同向时为 <code>+1</code>，正交时为 <code>0</code>，反向时为 <code>−1</code>。当每个 token
          的行长度相当时，logit 实际上就是隐藏状态与该行之间<em>夹角</em>
          的读数。切换下面的三种情形——对齐、正交、相反——观察各维度的乘积（它们的和就是 logit）与余弦一起变化。
        </p>

        <InnerProductLogit />

        <p>
          这就是为什么你有时会在 top-K 顶部看到一些出乎你意料、却看起来还算合理的
          token：它们的行恰好与隐藏状态对齐，即使模型最终未必会采样它们。点击 Run 之后，右侧的 top-K
          面板会具体展示这一点。
        </p>

        <h2>权重共享：同一个矩阵，用两次</h2>
        <p>
          接下来是让人意外的部分。上面公式里的矩阵 <code>W_lm</code> 并不是一个单独学习的张量。对
          Qwen3.5（以及大多数 7B 以下的现代解码器 LLM）来说，它<em>就是</em>第 3 章的嵌入矩阵——在堆叠底部把 token id
          映射成向量的那个 <code>[248,320, 1024]</code> 浮点数网格，在顶部（转置后）被复用。
        </p>

        <WeightTyingVisual />

        <p>
          <ChapterLink chapterId="embeddings">第 3 章</ChapterLink>提到这省下了第二份拷贝——这一份拷贝，画出来就是这样。
        </p>

        <TiedUntiedLedger />

        <p>
          这就是<strong>权重共享</strong>（weight tying），由配置中的 <code>tie_word_embeddings = true</code>{' '}
          控制。它背后有个干净的直觉：模型读入 token 用的那套“字母表”，也应该是它写出 token 用的字母表。如果{' '}
          <code>embed_tokens.weight</code> 的第 <code>j</code> 行是 <em>"the"</em>{' '}
          作为隐藏向量的样子，那么每当残差流长得像这一行时，LM head 就应该给 <em>"the"</em> 打出高分——而{' '}
          <code>row_j · h</code> 恰好就是这个检验。
        </p>
        <p>
          实际收益：一个参数量不足 10 亿的模型靠不复制这个矩阵省下约 254M
          参数（接近其总量的三分之一）。代价：在非常大的规模上有少量质量损失，这也是为什么几十亿参数以上的 GPT
          风格模型有时会<em>解开</em>（untie）这个头。对 Qwen3.5-0.8B 来说，省下的参数占了上风。
        </p>

        <h2>右侧你将看到什么</h2>
        <p>
          该面板执行一次 inspector 调用：对提示词分词、运行一次前向传播、在最后一个位置捕获 top-K
          logits、把它们渲染成柱状图。每一行是 logit 最高的 <code>{TOP_K}</code> 个词表 token
          之一；以主题色高亮的那一行是模型实际采样到的 token（此阶段为贪心——第 11 章会引入
          temperature（温度）和 top-p）。
        </p>
        <p className="text-muted-foreground">
          <em>值得注意：</em>面板对这些 logits 做了 softmax，所以柱子的<em>高度</em>读作概率（每个面板的柱子之和为
          1）。但 LM head 输出的原始分数是 <em>logits</em>——可正可负的任意实数，其大小在不同提示词之间不可比。第 11
          章将讲到 temperature 和 top-p 如何把这个 softmax 重塑成模型的最终选择。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

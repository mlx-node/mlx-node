import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import { ChapterLink } from '../../scaffolding/ChapterLink';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { CosineSimilarityTool } from '../../widgets/CosineSimilarityTool';
import { EmbeddingDirections } from '../../widgets/EmbeddingDirections';
import { EmbeddingLookupDiagram } from '../../widgets/EmbeddingLookupDiagram';
import { ShapeProblem } from '../../widgets/ShapeProblem';
import { VectorCosine } from '../../widgets/VectorCosine';

/**
 * 第 3 章（embeddings）的简体中文翻译 — 与 ../02-embeddings.tsx 平行维护。
 * 仅翻译正文与 learning 数据；`EmbeddingsDemo`（PCA 散点图、最近邻浏览器）
 * 及其全部辅助代码保留在英文模块中，由英文模块渲染（演示界面保持英文）。
 */

export const learning: ChapterLearningData = {
  chapterId: 'embeddings',
  objective: '解释一个整数 token id 如何变成高维向量，以及为什么向量相近意味着 token 相关。',
  problem: 'Transformer 需要每个 token 的连续表示，这样才能表达相似度，梯度也才能流动。',
  minutes: 7,
  glossary: [
    {
      term: 'embedding',
      definition: 'token id 被映射到的那个向量——嵌入矩阵的一行，网络其余部分就在它上面做计算。',
    },
    {
      term: 'embedding matrix',
      definition: '形状为 [vocab_size, hidden_dim]。对 Qwen3.5-0.8B 来说约为 248,320 × 1,024——约 254M 个参数。',
    },
    {
      term: 'hidden_dim',
      definition: '每个 token 的向量在模型中流动时的宽度。Qwen3.5-0.8B 为 1,024。',
    },
    {
      term: 'tied embeddings',
      definition:
        '让输入嵌入矩阵与输出 lm_head 共享，使同一个向量空间既负责读取提示词，也负责写出下一个 token 的 logit。',
    },
    {
      term: 'PCA',
      definition: '主成分分析——把高维点投影到方差最大的两个方向上，从而画出 2D 图。',
    },
    {
      term: 'cosine similarity',
      definition: '两个向量的点积除以各自的范数。衡量方向的一致程度而忽略长度；接近 1 表示非常相似。',
    },
    {
      term: 'semantic direction',
      definition:
        '像 E(woman) − E(man) 这样的向量差，近似编码某一个特征（性别、单复数、所属国家），可以加到其他嵌入上或从中减去。',
    },
  ],
  takeaways: [
    '嵌入矩阵约占一个小模型参数量的三分之一——它不是免费的查找表。',
    'PCA 适合发现聚类，但它的距离会骗人，所以相似度要在完整的 hidden_dim 空间里算。',
    '输入/输出嵌入绑定迫使一个矩阵身兼两职，这会约束其几何结构能编码的内容。',
  ],
  exercise: {
    prompt:
      "点击 Load embeddings，然后点一个动物，比如 'tiger'。再用输入框添加自定义词 'banana'。它的 top-5 最近邻列表偏向哪个已有类别？为什么？",
    answer:
      'banana 的近邻偏向食物聚类（apple、bread、cheese 排得很靠前），而不是颜色或动物，因为嵌入对“可食用名词”的编码比“黄色物体”更强——尽管 banana 两者都是。',
  },
  quiz: [
    {
      id: 'q1-why-vectors',
      prompt: '模型为什么把 token 嵌入成向量，而不是直接把整数 id 往后传？',
      options: [
        {
          id: 'a',
          label: '整数没法参与 GPU 的矩阵乘法。',
        },
        {
          id: 'b',
          label:
            '整数不携带任何相似性的概念——相邻的 id 之间没有关系，而模型需要一个“相近即相关”的空间，梯度才能把东西挪来挪去。',
        },
        {
          id: 'c',
          label: '向量比原始 id 更省内存。',
        },
      ],
      correctId: 'b',
      explanation:
        '嵌入的全部意义就在于把离散 token 变成模型可以在其中做几何的连续空间。内存实际上是变多了，不是变少。',
    },
    {
      id: 'q2-pca-distances',
      prompt: '演示为什么在完整的 1024 维空间里找最近邻，而不是用 2D 的 PCA 坐标？',
      options: [
        {
          id: 'a',
          label: '前两个主成分只能解释百分之几的方差，2D 距离丢掉了定义相似度的大部分结构。',
        },
        {
          id: 'b',
          label: 'PCA 距离比在 1024 维里算余弦更慢。',
        },
        {
          id: 'c',
          label: 'PCA 只能处理整数 id，不能处理浮点向量。',
        },
      ],
      correctId: 'a',
      explanation:
        'PCA 投影到方差最大的方向，但这两个方向只解释 1024 维点云的一小部分。这张图适合看聚类；上面的距离会误导人。',
    },
    {
      id: 'q3-tied-embeddings',
      prompt: 'Qwen3.5 把输入嵌入与输出 lm_head “绑定”（tie）是什么意思？',
      options: [
        {
          id: 'a',
          label:
            '两层共享同一个 [vocab_size, hidden_dim] 矩阵——一个矩阵读取提示词，同一个矩阵把最终隐藏状态变回词表上的 logits。',
        },
        {
          id: 'b',
          label: '模型用相同的学习率训练它们。',
        },
        {
          id: 'c',
          label: '它强制每个 token 的嵌入都具有单位 L2 范数。',
        },
      ],
      correctId: 'a',
      explanation: '绑定 = 两个方向真的共用同一个权重张量。既省参数，又迫使每个 token 只有一套连贯的表示。',
    },
    {
      id: 'q4-analogy-honest',
      prompt: '在 Qwen3.5 的原始嵌入矩阵里，king − man + woman 实际证明了什么？',
      options: [
        {
          id: 'a',
          label: "它凭空变出了 ' queen'——不做算术的话，queen 和 king 毫无关系。",
        },
        {
          id: 'b',
          label:
            "它把 ' queen' 从 king 的第 4 名原始近邻提升到第 1 名，越过 king 的大小写和复数变体——这个方向把一个本就很近的向量推到了榜首。",
        },
        {
          id: 'c',
          label: '它证明矩阵里存着一条精确的性别轴：woman − man 与 queen − king 指向完全相同。',
        },
      ],
      correctId: 'b',
      explanation:
        'queen 本来就很近（king 的第 4 名原始近邻）；算术真正的功劳是赢过 King/kings 这些变体。而且这条轴并不精确：cosine(woman − man, queen − king) 只有 0.292。',
    },
  ],
};

export function EmbeddingsChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>词嵌入（Embeddings）：把 token 变成向量</h1>
        <p>
          分词给了模型一串整数——每个 token 一个 id。但整数不携带模型能计算的含义：id <code>1234</code> 并不比{' '}
          <code>1235</code> 更大、更小，或在任何有用的意义上更<em>相似</em>。下一步是把每个 id
          映射成网络其余部分能做数学运算的连续向量。这个向量就是该 token 的<strong>嵌入</strong>（embedding）。
        </p>

        <h2>为什么是向量？</h2>
        <p>
          连续的表示让模型能够表达相似的“程度”。核心想法是：向量就是<em>空间中的一个方向</em>
          ，含义相近的词最终会指向相近的方向——于是“相关”变成了模型可以用夹角度量的东西。著名的（虽然有点理想化的）word2vec
          例子——<code>king − man + woman ≈ queen</code>
          ——表明学到的嵌入甚至能把“关系”捕捉成可以做算术的方向。现代 LLM 的嵌入结构比 word2vec
          的纠缠得多（一个模型要跨多种语言和代码风格服务几十种任务），所以这种干净的类比算术变弱了——但底层思想没变：
          <strong>向量相近意味着 token 相关</strong>，而 Transformer
          各层的整个前向传播，做的就是以符合上下文的方式挪动这些向量。
        </p>
        <p>
          这里的“相近”指的是<em>指向相同的方向</em>，而不是“尺子量出来的距离短”。在把真实的 1024
          维向量投影成图片之前，先在一个可以上手拖动的平面 2-D 玩具里建立核心直觉：含义住在向量的<em>方向</em>
          里，两个向量之间的<em>夹角</em>度量它们有多相关。
        </p>
        <p>
          所有度量只用到两小块数学，而且都能写在一行里。<strong>点积</strong>就是“对应位置相乘，再加起来”：对{' '}
          <code>a = [3, 4]</code> 和 <code>b = [4, 3]</code>，<code>a·b = 3×4 + 4×3 = 24</code>。
          <strong>范数</strong> <code>‖a‖</code> 是向量的长度：<code>‖a‖ = √(3² + 4²) = 5</code>，同样{' '}
          <code>‖b‖ = 5</code>。余弦相似度只是把长度除掉之后的点积：<code>cos = 24 / (5 × 5) = 0.96</code>
          ——这两个向量几乎指向同一个方向。这一个配方——点积除以范数——就是本章里所有的相似度数字。
        </p>

        <VectorCosine />

        <h2>嵌入表</h2>
        <p>
          嵌入查询是一次矩阵乘法（或者等价地，一次按行取数）：给定整数 id，取嵌入矩阵的第 <code>id</code>{' '}
          行。矩阵的形状是 <code>[vocab_size, hidden_dim]</code>。对 Qwen3.5-0.8B 来说，光这一张表就有约{' '}
          <code>248,320 × 1,024 ≈ 254 million</code>{' '}
          个参数（一个参数就是一个学到的数字；整个模型一共约 853M 个）——约占整个模型的三分之一。
        </p>
        <p>
          值得停下来想一想：这 2.54 亿个数字一开始全都是<em>随机噪声</em>。没有人坐下来把 <code>tiger</code> 安排在{' '}
          <code>lion</code> 旁边、画一条性别轴，或者把 <code>king − man + woman ≈ queen</code>{' '}
          接好线。本章里你会去拨弄的每一个聚类、每一条语义方向、每一个类比，都是在海量文本上拟合同一个目标——预测下一个
          token——所<em>涌现</em>出来的副产品。这套几何结构是
          <ChapterLink chapterId="training">训练</ChapterLink>留下的化石，而不是谁手工搭起来的字典。
        </p>
        <p>
          下面把这次查询画成可以逐步播放的图。一个 token id 进来，一行向量出去——中间什么也没有计算：
        </p>

        <EmbeddingLookupDiagram />
        <p>
          现代 LLM（包括 Qwen3.5）常常把输入嵌入与输出“反嵌入”<strong>绑定</strong>（tie）——后者就是把最终隐藏状态变回词表
          logits 的 <code>lm_head</code>，我们会在 LM head
          一章细讲；现在你只需把它读作“给下一个 token
          打分的那一层”。绑定这两个矩阵既省参数，也迫使这套表示在两个方向上都好用：<em>读取</em>
          提示词的那个向量空间，同时也要<em>写出</em>下一个 token 的
          logit。右侧的小部件画的正是这同一个共享矩阵里的行。
        </p>

        <ShapeProblem
          question="Qwen3.5 的嵌入矩阵有多大？嵌入绑定又省下了多少？"
          formula="params = vocab_size × hidden_dim"
          values={[
            { label: 'vocab_size', value: '248,320' },
            { label: 'hidden_dim', value: '1024' },
          ]}
          answer="248,320 × 1024 ≈ 254.3M 个参数——约占 Qwen3.5-0.8B 全部 ~853M 的三分之一。嵌入绑定把这同一个矩阵复用为输出投影（unembed），再省下 254.3M。如果不绑定，光嵌入+反嵌入加起来就差不多有一个小型独立模型那么大。"
        />

        <h2>PCA：通往 1024 维空间的一扇 2D 窗户</h2>
        <p>
          想象给一座 3-D 雕塑拍照：你会把它转到特征铺得最开的角度，好看清结构。PCA 就是为 1024
          维嵌入空间挑出那个“最佳机位”——而且和照片一样，它把纵深压扁了，所以屏幕上的距离只是粗略的参考，不是精确的尺子。
        </p>
        <p>
          我们看不见 1024 个维度，所以要投影。<strong>主成分分析</strong>
          找出数据中方差最大的方向，并投影到前两个（或前三个）方向上。结果是线性压扁所能做到的最好程度：投影后的点保留了一张
          2D 图所能承载的最多原始分布信息。
        </p>
        <p>
          问题在于“尽可能多”仍然很少。1024
          维点云的前两个主成分通常只解释总方差的百分之几——其余都在被丢掉的方向里。所以 PCA 很适合发现
          <em>聚类</em>（在嵌入空间中总体方向相近的 token 会落在一起），但两根轴装不下 1024 根轴的几何。
        </p>

        <h2>你应该看到什么</h2>
        <p>
          点击 <strong>Load embeddings</strong>。小部件会把来自六个类别（动物、数字、颜色、国家、动词、食物）的约
          110 个词分词，通过模型 worker 查出每个词第一个 token 的嵌入，并在你的浏览器里运行
          PCA。每个词只用它的<em>第一个</em>子词 token
          来画，所以被切成多块的词在图上只由开头那一块代表。可以期待看到：
        </p>
        <ul>
          <li>
            <strong>按类别聚类。</strong>动物挨着动物，数字挨着数字，依此类推。尽管只是 1024 维空间的一个 2D
            小切片，聚类通常仍然清晰。
          </li>
          <li>
            <strong>聚类内部的细结构。</strong>
            数字常常排成大致的梯度。颜色分成暖色/冷色两个子区域。食物把咸味和甜味分开。
          </li>
          <li>
            <strong>看起来“懂语义”的最近邻。</strong>点任何一个点：完整 hidden-dim 空间里的 top-5
            近邻会被高亮，它们通常是同类词，再加上几个出人意料的跨类别“擦边球”。
          </li>
        </ul>

        <h2>为什么用余弦，而不是欧氏距离？</h2>
        <p>
          余弦相似度度量两个向量之间的<em>夹角</em>，而夹角正是嵌入空间中编码语义<em>方向</em>
          的东西。欧氏距离度量完整的向量差，把方向和长度混在了一起。训练得到的向量长度取决于优化动态，并没有语义含义——两个同义词的范数可以差很远，方向却几乎相同。这就是余弦成为嵌入相似度标准的原因。有些检索系统会在预先归一化的向量上用
          L2，那在数学上是等价的。
        </p>

        <h2>散点图用来定向，不用来测量</h2>
        <p>
          散点图里你看到的一切都受一条前提约束：它把 1024 维向量投影到仅仅两根轴上，丢掉了 1022
          个维度的结构。屏幕上看着很近的两个点，在真实空间里可能离得很远，反之亦然。颜色分组依然会聚在一起（方差最大的几个方向往往能抓住它们），但
          <em>距离</em>会骗人——所以永远不要从图上读相似度。这正是最近邻浏览器在完整的 1024
          维空间、而不是投影后的 2D 坐标里计算余弦相似度的原因。把散点图当成找聚类的导航图，而不是尺子。
        </p>

        <h2>余弦相似度，并排看</h2>
        <p>
          下面的面板完全跳过投影：每根条形给出该模型家族的一个大致余弦相似度——右侧的实时散点图用的才是真实嵌入。条形的排序就是要教的结论——完全相同
          &gt; 同义词 &gt; 相关 &gt; 反义词 &gt; 跨语言 &gt; 不相关。
        </p>

        <CosineSimilarityTool />

        <h2>方向承载含义</h2>
        <p>
          到目前为止，我们一直把相似度当作两个完整向量之间的一个数字。但这个空间还有更细的结构：嵌入之间的<em>差</em>
          可以表现得像特征。如果 <code>E(woman) − E(man)</code> 大致捕捉了“性别轴”，那么把这个差加到别的向量上，就应该让它沿同一条轴移动——这正是经典的
          word2vec 把戏：<code>E(king) − E(man) + E(woman)</code> 应该落在 <code>queen</code>
          附近。减掉“男性”，加上“女性”，“王室”这部分会一路跟着过来。
        </p>
        <p>
          在 2026 年的 LLM 嵌入矩阵里，这一招还灵吗？我们在这个 checkpoint
          上实际测了一遍，诚实的回答是“灵，但要打个星号”。<code>queen</code> 本来<em>就是</em> <code>king</code>{' '}
          的第 4 名原始近邻——排在大小写和复数变体 <code>King</code>、<code>King</code>（无前导空格）、
          <code>kings</code> 之后——所以这个算术并没有凭空召唤出 queen。它真正做到的是
          <strong>把 queen 提升到第 1 名</strong>，越过所有 king
          变体：这个方向把“男性”成分剥掉了足够多，让女性对应词胜出。同一招还能推广：
          <code>E(Paris) − E(France) + E(Japan)</code> 让 <code>Tokyo</code> 排到第 1——而且由于这个 248,320 个 token
          的词表是多语言的，<code>东京</code>和<code>日本</code>会和 <code>Osaka</code>、<code>Kyoto</code> 出现在同一份
          top-10 里。
        </p>
        <p>
          不是每条方向都这么上镜。复数方向 <code>E(cats) − E(cat)</code>{' '}
          确实存在——复数名词与它的点积一致为正，单数徘徊在 0 附近——但数值非常微弱，理想中的“one &lt; two &lt;
          three”递增趋势几乎看不出来。而且这些方向本身也只是大致一致：<code>woman − man</code> 与{' '}
          <code>queen − king</code> 的余弦只有 0.292，远谈不上平行。这并不意外——这个矩阵是为“帮助预测下一个
          token”训练的（输入/输出权重绑定，身兼两职），不是为了通过类比测验。线性的特征方向只是副产品，被这一个矩阵必须编码的其他一切弄花了。
        </p>

        <EmbeddingDirections />

        <h2>为什么需要这么多维度？</h2>
        <p>
          如果只在 3D
          里做这件事，几乎所有概念都会撞在一起。“动物”、“国家”、“颜色”、“动词”、“句法角色”、“语域”、“语言”以及另外十几条含义轴，根本没有空间在一个三轴空间里独立变化。Qwen3.5-0.8B
          使用 <strong>1,024</strong>{' '}
          个嵌入维度，给了模型相当宽裕的“房间”——每个独立特征都可以占据自己的方向，而不挤到别人。
        </p>

        <p className="mt-6 text-muted-foreground">
          接下来（<em>自注意力</em>）我们将看到这些向量在 Transformer 中是如何真正<em>移动</em>
          的——在位置之间搬运信息的那个操作，正是它让 "the cat sat on the mat" 的含义不同于 "the mat sat on the
          cat"。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

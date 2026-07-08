// 子章节 13.1（预训练数据整理）正文的简体中文译文，与英文版 ../training.tsx
// 平行维护：同名导出 TrainingDataCurationSection。
// 与英文版一样是纯静态阅读内容（SSR 安全，可被预渲染器直接渲染为静态 HTML）；
// 每一条事实性断言都以英文版引用的一手资料为依据，翻译不改动任何数字与结论。
// 术语（token、dedup、MinHash、FineWeb、Common Crawl、LLaMA、GPT-3、CCNet、
// KenLM、trafilatura、fastText、Jaccard、LSH 等）按本课程 zh 惯例保留英文。

import * as React from 'react';

import { Prose } from '../../Prose';
import { MathDisplay } from '../../scaffolding/MathDisplay';
import { DataCurationFunnel } from '../../widgets/DataCurationFunnel';

/** 子章节 13.1——"数万亿 token 的通用网页文本" 这句话背后到底做了什么。 */
export function TrainingDataCurationSection() {
  return (
    <Prose>
      <h1>预训练数据整理</h1>
      <p className="text-muted-foreground">
        深入阅读 · 第 13 章 Training &amp; teacher forcing——上一章只说了一句「数万亿 token 的通用网页文本、书籍和代码」。这里展开这句话背后到底是什么。
      </p>
      <p>
        上一章的训练目标——在每个位置最小化 <code>-log p(下一个 token)</code>——对每一个现代 LLM 都是同一行公式。真正有差异、并且默默做了大部分工作的，是<em>你喂给它什么文本</em>。直接拿原始网页数据不加处理去训练，效果会比先清洗一遍差——不是因为目标函数变了，而是因为原始网页文本本身有一堆目标函数本身看不出来的问题：里面充斥着近重复内容，相当一部分是样板文字或垃圾内容，来源和配比也不像一份精心整理过的语料那样讲究。这个子章节就把那句被一笔带过的话，拆成背后的三根支柱：
        <strong>去重（deduplication）</strong>、<strong>质量过滤（quality filtering）</strong>，以及<strong>来源与配比（sourcing &amp; mixture）</strong>。
      </p>

      <h2>去重：网页上的重复比你想的要多</h2>
      <p>
        Common Crawl 及类似的网页抓取并不是一份干净的、由互不相同的文档组成的样本。同一篇文章会被几十个新闻镜像站转载，同一段样板法律声明会出现在成千上万个互不相关的网站上，模板化页面往往只差一句话。Lee、Ippolito、Nystrom、Zhang、Eck、Callison-Burch 和 Carlini 实测了这类内容在标准训练语料里究竟留存了多少（《Deduplicating Training Data Makes Language Models Better》，2021）：C4 中 <strong>3.04%</strong>、RealNews 中 <strong>13.63%</strong>、LM1B 中 <strong>4.86%</strong>、Wiki-40B 中 <strong>0.39%</strong> 的文档是近重复的。极端情况下，C4 里一句仅 61 个词的句子被发现重复了<strong>超过 6 万次</strong>。
      </p>
      <p>
        为什么要费力去掉它？因为在大量重复文本上训练出来的模型，不只是在浪费算力反复学同一句话——它开始<em>记住</em>这句话。Lee 等人发现，去重能把训练好的模型逐字吐出记忆中训练文本的频率降低约 <strong>10 倍</strong>，同时还能把数据集规模缩小最多 19%，并减少意外的训练/测试集重叠（这种重叠曾让超过 4% 的标准验证集上的基准分数虚高）。
      </p>
      <p>
        论文用了两种互补的技术。<strong>Exact-substring deduplication（精确子串去重）</strong>在整个语料上建一棵后缀数组，删除任何在不止一份文档中出现的、长度 50 个 BPE token 以上的逐字片段——能抓住复制粘贴的样板文字和引用段落。<strong>Near-duplicate deduplication（近重复去重，「NearDup」）</strong>是更难的问题：两份文档<em>几乎</em>一样但字节并不完全相同（同一篇文章换了个标题重新发表、模板页面只改了一个字段）。Lee 等人用的工具是 <strong>MinHash</strong>：把每份文档表示成一组重叠的 5-gram 集合，把这组集合哈希成一个紧凑的签名（这里是 9,000 个哈希值，分成 20 组、每组 450 个），当两份文档的签名匹配足够多时就标记为近重复——对应的 Jaccard 相似度大致为
        <MathDisplay inline latex={String.raw`J(A,B) = \frac{|A \cap B|}{|A \cup B|} \gtrsim 0.8`} />
        。基于 MinHash 的 LSH（locality-sensitive hashing）如今已是对网页规模文本去重的行业标准做法——几乎每一条现代开放 pipeline（包括 FineWeb）都在复用它，只是各自调了自己的参数。
      </p>

      <h2>质量过滤：两套经典配方</h2>
      <p>
        去重去掉的是<em>重复</em>的文本；质量过滤去掉的是那些独一无二但<em>质量差</em>的文本——链接农场、堆砌关键词的垃圾内容、自动生成的样板文字、几乎全是导航菜单的页面。多年来主要有两套方法，如今都还在被广泛使用。
      </p>
      <p>
        <strong>Classifier-based filtering（基于分类器的过滤）</strong>训练一个小模型来区分「看起来像好文本」和原始网页抓取内容。GPT-3 的论文描述了这套方法的经典版本：用精选语料（WebText、Wikipedia、一份书籍语料）作为正例、原始 Common Crawl 作为负例训练一个逻辑回归分类器，再用它对 Common Crawl 重新采样，让结果更偏向 Wikipedia/书籍风格的文档。同样的思路两年后又以 <strong>FineWeb-Edu</strong> 的形式重新出现：让 Llama-3-70B-Instruct 给 50 万个网页样本按教育价值打 0-5 分，再从这些评分中蒸馏出一个轻量分类器，用来给 FineWeb 整个 15T token 的语料打分——由此产出一个 1.3T token 的子集，在 MMLU 这类知识密集型基准上明显优于未经筛选的语料。
      </p>
      <p>
        <strong>Perplexity-based filtering（基于困惑度的过滤）</strong>走的是另一条更便宜的路：用一个在已知优质文本上训练的小型语言模型，给每一段文本打「有多令人意外」的分数，然后丢弃困惑度高（令人意外、多半是样板文字或垃圾内容）的那一尾巴。CCNet 是被广泛引用的例子——为每种语言在该语言的 Wikipedia 上训练一个 5-gram KenLM 模型，用它给抓取到的段落打分，保留困惑度低、类 Wikipedia 的段落，后来被 RedPajama 和 Dolma 复用为一道过滤工序。（这一条本项目只通过描述该论文的二手来源确认了其方法，没有直接查证 CCNet 原始论文的正文——这里特意标注一下，因为本页其它每一条断言都能追溯到我们自己抓取过的一手来源。）
      </p>

      <h2>来源与配比：Common Crawl 是底料，不是全部配方</h2>
      <p>
        几乎每一份开放的预训练语料都从同一个地方起步：<strong>Common Crawl</strong>，一份公开的、持续更新的网页抓取，按月发布快照。把一份原始 Common Crawl 转储变成一份能用的预训练数据集，靠的正是上面那套 pipeline——而 <strong>FineWeb</strong>（Penedo 等人，2024）是文档最完整的一个实际范例：URL 黑名单过滤、文本提取（借助 <code>trafilatura</code> 库）、fastText 语言识别、Gopher/C4 风格加自定义启发式质量过滤，最后是 MinHash 近重复去重——把 96 份原始 Common Crawl 快照变成一份 15 万亿 token 的精选英文数据集。切换到下面的「Mixture」视图，能看到来源故事的另一半：各家实验室并不只用网页文本训练，而是<em>刻意</em>把它与书籍、代码、参考文献类文本按固定的非均匀比例混合——LLaMA 已发表的配比表是引用最清楚的例子：稀缺的高质量来源（Wikipedia、书籍、ArXiv）被多轮上采样，而占比大得多的 Common Crawl 部分却几乎只跑了刚过一轮。
      </p>

      <DataCurationFunnel />

      <p className="text-muted-foreground">
        请把 pipeline 视图当作一种收窄的示意，而不是一张精确表格：只有入口（96 份快照）和两个出口（FineWeb 的 15T token，以及经过分类器过滤的 FineWeb-Edu 子集的 1.3T token）是 FineWeb 真正公布过的数字。而它下面的去重统计数据不同——那是 Lee 等人的实测结果，特意单独放在自己的一行里，避免示意性的收窄和实测的事实被混为一谈。
      </p>

      <h2>你浏览器里的 Qwen 会告诉你它的训练数据是怎么清洗的吗？</h2>
      <p>
        不会——而这一次，诚实的答案与其说是关于这个具体的浏览器演示，不如说是关于它背后的整个行业。<strong>Qwen3 Technical Report</strong> 确实披露了一些关于规模的真实数字：三个预训练阶段总共约 <strong>36 万亿 token</strong>（Stage 1，通用阶段，<code>&gt;30T</code> token；Stage 2，推理/知识强化阶段，再加 <code>+≈5T</code>；Stage 3，长上下文阶段，在 32,768 token 序列长度下再加数千亿），外加粗粒度的类别标签——STEM、代码、书籍、多语言文本，以及借助 Qwen2.5-VL（用于对类 PDF 文档做 OCR）、Qwen2.5-Math、Qwen2.5-Coder 抽取和生成的合成数据。但它<strong>没有</strong>披露本页提到的任何具体机制：没有点名任何去重方法或工具，没有描述任何质量过滤方法（分类器或困惑度），也没有网页 vs 书籍 vs 代码 vs 学术文本的百分比拆分——完全没有类似上面 LLaMA 那张表的东西。
      </p>
      <p>
        对这门课真正在跑的这个具体模型来说，情况还要更模糊一些。截至 2026-07-01，在 arXiv 上找不到一篇专门的 <em>Qwen3.5</em> 或 <em>Qwen3-Next</em> technical report——覆盖的是这个浏览器标签页实际运行的、纯文本混合 GatedDeltaNet 这条线，而不是另一条独立的多模态线。2026 年存在两篇 Qwen 论文，但都没能填上这个缺口：<em>Qwen3.5-Omni Technical Report</em> 覆盖的是一个架构上完全不同的全模态（文本+视觉+音频）模型，披露了数据规模（超过 1 亿小时的音视频数据），但同样对整理方法只字未提；更早的
        <em>Qwen3-Coder-Next Technical Report</em>（arXiv:2603.00729，2026 年 2 月）虽然基于 Qwen3-Next 底座，但聚焦的是
        agentic 编程能力——是在已有底座上做的 mid-training 和 RL，不是预训练数据的披露。这两篇都不是把猜测包装成事实——这是真的去搜索之后一无所获的诚实结果。
      </p>
      <p className="text-muted-foreground">
        一份独立审计印证了这一点。斯坦福 CRFM 的 Foundation Model Transparency Index（2025 年 12 月的公司报告）给 Alibaba/Qwen 在以下几项上都打了<strong>0 分——未披露</strong>：获取预训练数据的方法、使用的公开数据集来源、爬虫身份与 robots.txt opt-out 处理、授权数据的具体细节，以及语言/领域配比百分比。这并不是 Qwen 独有的问题：OpenAI、Anthropic、Google、Meta 在自己的前沿模型发布中同样不披露这个层级的细节。所以诚实的全貌是这样的：本页正文里的一切——去重、质量过滤、Common Crawl、FineWeb、LLaMA 的配比——都是真实的、有文档记录、可核实的，因为它们来自选择公开这些内容的实验室和论文。而真正用来清洗这些权重训练语料的配方——这个具体 checkpoint 的规模未被披露，虽然大概率和 Qwen3 公布的约 36 万亿 token 是同一个数量级——和几乎所有其它前沿模型一样，是不公开的。
      </p>
    </Prose>
  );
}

// ui.ts — the typed UI-chrome string dictionary. React-free (same rule as
// lib/i18n.ts) so any pure module can import a locale's strings; the React
// hook lives in ui-react.ts.
//
// Every English value here is the EXACT string previously hardcoded in the
// component it replaces — the English experience must not change. Strings with
// interpolation are functions. Chinese style: 简体中文, established technical
// terms stay recognizable (KV 缓存, token/LLM/logits stay English), numbers and
// units unchanged.

import type { Locale } from '../../lib/i18n';

export type UiStrings = {
  languageSwitcher: {
    /** aria-label for the EN | 中文 toggle group. */
    ariaLabel: string;
  };

  lessonLayout: {
    allChapters: string;
    /** Header breadcrumb, e.g. "Chapter 4 · Self-attention". */
    breadcrumb: (number: number, title: string) => string;
    loadModel: string;
    freeChat: string;
    /** Sidebar heading. */
    chaptersHeading: string;
    /** Badge on not-yet-authored sidebar rows. */
    soonBadge: string;
    /** Right-hand demo column heading. */
    tryItNow: string;
  };

  chapterPage: {
    notAuthored: string;
  };

  sectionPage: {
    notAuthored: string;
  };

  chapterIndex: {
    back: string;
    openFreeChat: string;
    eyebrow: string;
    heading: string;
    description: (n: number) => string;
    /** Card corner chip, e.g. "Ch. 04". */
    chapterChip: (padded: string) => string;
    readyBadge: string;
    comingSoonBadge: string;
    openChapter: string;
    notYetAuthored: string;
    footerNote: string;
  };

  forwardPass: {
    heroTitle: string;
    heroHint: string;
    promptLabel: string;
    run: string;
    rerun: string;
    runningLabel: string;
    replayingLabel: string;
    play: string;
    pause: string;
    speed: string;
    speedAria: string;
    // Status pill.
    statusReady: string;
    statusRunning: string;
    statusReplayingLayer: (layer: number, total: number) => string;
    statusReplayingStage: (stage: string) => string;
    statusReplaying: string;
    statusPausedLayer: (layer: number, total: number) => string;
    statusPaused: string;
    statusDone: string;
    statusAborted: string;
    statusEmptyPrompt: string;
    statusError: (message: string) => string;
    // Stage display names used inside the status pill.
    stageTokenize: string;
    stageEmbedding: string;
    stageFinalNorm: string;
    stageLmHead: string;
    stageSampling: string;
    // Consent / loading gate.
    preparingModel: string;
    consentTitle: string;
    consentBodyLocal: string;
    consentBodyHosted: string;
    ctaRetry: string;
    ctaChooseLocal: string;
    ctaLoadModel: string;
    // Helper line under the status pill.
    helperLine: (totalSeconds: number) => string;
    // Fixed head/tail chips.
    chipTokenize: string;
    chipEmbedding: string;
    chipFinalNorm: string;
    chipLmHead: string;
    chipSampling: string;
    chipChapter: (n: number) => string;
    chipAria: (chapterNum: number, label: string) => string;
    // Card-stack header.
    layersHeader: (layers: number, chapterNum: number) => string;
    layerProgress: (layer: number, total: number) => string;
    decoderLayer: string;
    fullAttnTitle: string;
    linearAttnTitle: string;
    /** Sub-stage row labels, keyed by sub-stage id. Missing keys fall back to
     *  the English label baked into the scene constants. */
    subStages: Record<string, string>;
    // Token reveal.
    nextTokensLabel: string;
    revealCaption: (hasEcho: boolean, n: number) => string;
    // Footer paragraphs (split around inline links / <strong>).
    footerStack: string;
    footerLoop1: string;
    footerLoopKvLink: (chapterNum: number) => string;
    footerLoop2: string;
    footerHonest1: string;
    footerHonestGreedy: string;
    footerHonest2: string;
    footerHonestSamplesLink: string;
    footerHonest3: string;
    footerHonestCapped: string;
    footerHonest4: (cap: number) => string;
    footerHonestWholeModelLink: string;
    footerHonest5: string;
  };

  landing: {
    tag: string;
    /** Hero title around the <em> word: pre + em + post. */
    heroTitlePre: string;
    heroTitleEm: string;
    heroTitlePost: string;
    /** Hero subtitle around the <code>@mlx-node/browser</code> chunk. */
    heroSubBeforeCode: string;
    heroSubAfterCode: string;
    voidAdAria: string;
    voidEyebrow: string;
    voidMark: string;
    voidCopy: string;
    specParams: string;
    specParamsLabel: string;
    specChapters: (n: number) => string;
    specChaptersLabel: string;
    specLocal: string;
    specLocalLabel: string;
    checkingModel: string;
    chooseLocalModel: string;
    openChat: string;
    loadModel: (label: string) => string;
    localPickerTitle: string;
    startLearning: string;
    learnHint: (n: number) => string;
    jspaceLink: string;
    mascotAlt: string;
    builtWith: string;
    hostedOnVoid: string;
    localModelLink: string;
  };

  scaffolding: {
    /** ChapterHeader minutes badge, e.g. "12 min". */
    minBadge: (minutes: number) => string;
    glossaryTitle: string;
    glossaryCount: (n: number) => string;
    takeawaysTitle: string;
    tryThisTitle: string;
    showAnswer: string;
    hideAnswer: string;
    quickCheckTitle: string;
    score: (correct: number, total: number) => string;
    resetAnswers: string;
    correctLabel: string;
    notQuiteLabel: string;
    correctAria: string;
    incorrectAria: string;
    /** SubChapterNav default eyebrow. */
    goDeeper: string;
  };

  pipeline: {
    eyebrow: string;
    chipTokenize: string;
    chipEmbedding: string;
    chipLayers: string;
    chipLayersTitle: string;
    chipFinalNorm: string;
    chipLmHead: string;
    chipSampling: string;
    wholePipeline: string;
    summary: string;
    describeAll: (summary: string) => string;
    describeMeta: (summary: string) => string;
    describeStages: (phrase: string, plural: boolean, summary: string) => string;
    stagePhrases: Record<string, string>;
    /** Joiner for multi-stage phrases, e.g. ' and ' / '和'. */
    phraseJoin: string;
  };
};

const EN: UiStrings = {
  languageSwitcher: {
    ariaLabel: 'Language',
  },

  lessonLayout: {
    allChapters: 'All chapters',
    breadcrumb: (number, title) => `Chapter ${number} · ${title}`,
    loadModel: 'Load model',
    freeChat: 'Free chat',
    chaptersHeading: 'Chapters',
    soonBadge: 'soon',
    tryItNow: 'Try it now',
  },

  chapterPage: {
    notAuthored: 'This chapter is not yet authored.',
  },

  sectionPage: {
    notAuthored: 'This sub-chapter is not yet authored.',
  },

  chapterIndex: {
    back: 'Back',
    openFreeChat: 'Open free chat',
    eyebrow: 'Learn LLMs · powered by Qwen3.5 in your browser',
    heading: 'Chapters',
    description: (n) =>
      `${n} guided lessons that explain how a modern transformer LLM works, using the real model running in your browser via WebGPU. Read the prose on the left, then poke at the live model on the right.`,
    chapterChip: (padded) => `Ch. ${padded}`,
    readyBadge: 'Ready',
    comingSoonBadge: 'Coming soon',
    openChapter: 'Open chapter →',
    notYetAuthored: 'Not yet authored',
    footerNote:
      'Each chapter is self-contained, but the suggested order builds the model from the bottom up — text in, attention through the middle, sampling out.',
  },

  forwardPass: {
    heroTitle: 'One forward pass, end to end — real model run',
    heroHint: 'The card on top is the current layer. Click any sub-stage to open its chapter.',
    promptLabel: 'Prompt',
    run: 'Run',
    rerun: 'Re-run',
    runningLabel: 'Running…',
    replayingLabel: 'Replaying…',
    play: 'Play',
    pause: 'Pause',
    speed: 'Speed',
    speedAria: 'Replay speed',
    statusReady: 'Ready to run.',
    statusRunning: 'Running real inference…',
    statusReplayingLayer: (layer, total) => `Replaying · layer ${layer} / ${total}`,
    statusReplayingStage: (stage) => `Replaying · ${stage}`,
    statusReplaying: 'Replaying…',
    statusPausedLayer: (layer, total) => `Paused · layer ${layer} / ${total}`,
    statusPaused: 'Paused',
    statusDone: 'Done — token revealed below.',
    statusAborted: 'Run cancelled.',
    statusEmptyPrompt: 'Enter a prompt to run.',
    statusError: (message) => `Error: ${message}`,
    stageTokenize: 'tokenize',
    stageEmbedding: 'embedding',
    stageFinalNorm: 'final RMSNorm',
    stageLmHead: 'LM head',
    stageSampling: 'sampling',
    preparingModel: 'Preparing the model…',
    consentTitle: 'Watch the model think — live, in your browser',
    consentBodyLocal:
      'This live demo needs the model. No hosted model is available here — choose a local model directory to run it. Everything runs on your device.',
    consentBodyHosted:
      'Loads the real Qwen3.5-0.8B (~1.6 GB) and runs one forward pass, replaying every layer through a card stack. Runs entirely on your device — nothing leaves your browser. The chapters below are fully readable without it.',
    ctaRetry: 'Retry',
    ctaChooseLocal: 'Choose local model',
    ctaLoadModel: 'Load model',
    helperLine: (totalSeconds) =>
      `Real inference, real per-layer trace — replayed through a card stack at ~0.9–1.0 s/layer so you can watch each sub-stage fire. Total replay ~${totalSeconds}s at 1× speed (½× to slow down for reading, 4× to skim).`,
    chipTokenize: 'Tokenize',
    chipEmbedding: 'Embedding lookup',
    chipFinalNorm: 'Final RMSNorm',
    chipLmHead: 'LM head → 248K logits',
    chipSampling: 'Sampling → next token',
    chipChapter: (n) => `ch ${n}`,
    chipAria: (chapterNum, label) => `Open chapter ${chapterNum}: ${label}`,
    layersHeader: (layers, chapterNum) => `× ${layers} LAYERS · chapter ${chapterNum} (full block)`,
    layerProgress: (layer, total) => `Layer ${layer} / ${total}`,
    decoderLayer: 'decoder layer',
    fullAttnTitle: 'Full softmax attention layer',
    linearAttnTitle: 'Linear-attention layer (GatedDeltaNet)',
    subStages: {},
    nextTokensLabel: 'next tokens →',
    revealCaption: (hasEcho, n) =>
      `${hasEcho ? 'first new token highlighted · ' : 'first token highlighted · '}${n} predicted token${n === 1 ? '' : 's'}`,
    footerStack:
      'Watch the stack: full-attention layers (cyan) compute Q · Kᵀ · softmax over all past tokens; linear layers (amber, GatedDeltaNet) maintain a recurrent state — Qwen3.5 interleaves them 1:3.',
    footerLoop1:
      "This is one turn of the loop. The model appends that next token and runs the whole stack again — but on the next pass it doesn't recompute the prefix: the ",
    footerLoopKvLink: (chapterNum) => `KV cache (chapter ${chapterNum})`,
    footerLoop2:
      " is reused — the prefix isn't recomputed, so each new token is far cheaper than reprocessing the whole context. Generation is this forward pass, looped.",
    footerHonest1: 'Honest framing: this diagram shows ',
    footerHonestGreedy: 'greedy',
    footerHonest2:
      ' completion — it always takes the single highest-logit token, so the same prompt gives the same answer every time here. Real generation usually ',
    footerHonestSamplesLink: 'samples',
    footerHonest3:
      ' from the distribution, so an identical prompt can come out differently from run to run. The revealed continuation is also ',
    footerHonestCapped: 'capped',
    footerHonest4: (cap) =>
      ` at ${cap} tokens — raw greedy decoding loops past that — so this is a window onto the first few predictions, not a full reply. See the `,
    footerHonestWholeModelLink: 'whole-model chapter',
    footerHonest5: ' for the finished picture and its limits.',
  },

  landing: {
    tag: 'Real LLM · 100% Local · WebGPU',
    heroTitlePre: 'How LLMs ',
    heroTitleEm: 'Work',
    heroTitlePost: '',
    heroSubBeforeCode:
      'Run a real 0.8B language model (Qwen3.5) entirely in your browser, then learn exactly how it works — layer by layer. No server, no API keys — powered by ',
    heroSubAfterCode: ' and WebGPU.',
    voidAdAria: 'Hosted on Void Platform',
    voidEyebrow: 'Hosted on',
    voidMark: 'Void Platform',
    voidCopy: 'Edge assets · WebGPU isolation',
    specParams: '0.8B params',
    specParamsLabel: 'Qwen3.5 · runs live',
    specChapters: (n) => `${n} chapters`,
    specChaptersLabel: 'Interactive course',
    specLocal: '100% local',
    specLocalLabel: 'WebGPU · no server',
    checkingModel: 'Checking Model...',
    chooseLocalModel: 'Choose Local Model',
    openChat: 'Open Chat →',
    loadModel: (label) => `Load Model (${label})`,
    localPickerTitle: 'Choose local model directory',
    startLearning: 'Start learning →',
    learnHint: (n) => `New to LLMs? Step through ${n} chapters that explain how this model actually works.`,
    jspaceLink: 'Open J-Space — the layer-by-layer lens →',
    mascotAlt: 'Capybara mascot',
    builtWith: 'Built with',
    hostedOnVoid: 'Hosted on Void',
    localModelLink: 'Local model…',
  },

  scaffolding: {
    minBadge: (minutes) => `${minutes} min`,
    glossaryTitle: 'Glossary',
    glossaryCount: (n) => `${n} term${n === 1 ? '' : 's'}`,
    takeawaysTitle: 'Engineering takeaways',
    tryThisTitle: 'Try this',
    showAnswer: 'Show answer',
    hideAnswer: 'Hide answer',
    quickCheckTitle: 'Quick check',
    score: (correct, total) => `${correct} / ${total} correct`,
    resetAnswers: 'Reset answers',
    correctLabel: 'correct',
    notQuiteLabel: 'not quite',
    correctAria: 'correct answer',
    incorrectAria: 'incorrect',
    goDeeper: 'Go deeper in this chapter',
  },

  pipeline: {
    eyebrow: 'forward pass',
    chipTokenize: 'Tokenize',
    chipEmbedding: 'Embedding lookup',
    chipLayers: '× 24 layers',
    chipLayersTitle:
      'Hybrid stack: 6 full-attention layers + 18 cheaper linear-attention (GatedDeltaNet) layers',
    chipFinalNorm: 'Final RMSNorm',
    chipLmHead: 'LM head',
    chipSampling: 'Sampling',
    wholePipeline: '· whole pipeline',
    summary: 'tokenize → embed → ×24 layers → final norm → LM head → sample',
    describeAll: (summary) => `Forward-pass map: this chapter covers the whole pipeline (${summary}).`,
    describeMeta: (summary) => `Forward-pass map (this chapter zooms out from a single step): ${summary}.`,
    describeStages: (phrase, plural, summary) =>
      `Forward-pass map: you are at the ${phrase} stage${plural ? 's' : ''} (${summary}).`,
    stagePhrases: {
      tokenize: 'tokenize',
      embed: 'embed',
      block: 'transformer block',
      'final-norm': 'final norm',
      'lm-head': 'LM head',
      sample: 'sample',
    },
    phraseJoin: ' and ',
  },
};

const ZH: UiStrings = {
  languageSwitcher: {
    ariaLabel: '语言',
  },

  lessonLayout: {
    allChapters: '全部章节',
    breadcrumb: (number, title) => `第 ${number} 章 · ${title}`,
    loadModel: '加载模型',
    freeChat: '自由对话',
    chaptersHeading: '章节',
    soonBadge: '即将上线',
    tryItNow: '动手试试',
  },

  chapterPage: {
    notAuthored: '本章尚未编写。',
  },

  sectionPage: {
    notAuthored: '本小节尚未编写。',
  },

  chapterIndex: {
    back: '返回',
    openFreeChat: '打开自由对话',
    eyebrow: '学习 LLM · 由浏览器中的 Qwen3.5 驱动',
    heading: '章节',
    description: (n) =>
      `${n} 节循序渐进的课程，讲解现代 Transformer LLM 的工作原理——真实模型通过 WebGPU 在你的浏览器里运行。左边读讲解，右边动手玩真实模型。`,
    chapterChip: (padded) => `第 ${padded} 章`,
    readyBadge: '可学习',
    comingSoonBadge: '即将上线',
    openChapter: '打开本章 →',
    notYetAuthored: '尚未编写',
    footerNote: '每一章都自成一体，但建议按顺序自底向上搭建模型——文本进入，注意力贯穿中间，采样输出。',
  },

  forwardPass: {
    heroTitle: '一次完整的前向传播——真实模型运行',
    heroHint: '最上面的卡片是当前层。点击任意子步骤可打开对应章节。',
    promptLabel: '提示词',
    run: '运行',
    rerun: '重新运行',
    runningLabel: '运行中…',
    replayingLabel: '回放中…',
    play: '播放',
    pause: '暂停',
    speed: '速度',
    speedAria: '回放速度',
    statusReady: '就绪，可以运行。',
    statusRunning: '正在运行真实推理…',
    statusReplayingLayer: (layer, total) => `回放中 · 第 ${layer} / ${total} 层`,
    statusReplayingStage: (stage) => `回放中 · ${stage}`,
    statusReplaying: '回放中…',
    statusPausedLayer: (layer, total) => `已暂停 · 第 ${layer} / ${total} 层`,
    statusPaused: '已暂停',
    statusDone: '完成——预测的 token 已在下方展示。',
    statusAborted: '已取消运行。',
    statusEmptyPrompt: '请输入提示词再运行。',
    statusError: (message) => `出错：${message}`,
    stageTokenize: '分词',
    stageEmbedding: '嵌入',
    stageFinalNorm: '最终 RMSNorm',
    stageLmHead: 'LM head',
    stageSampling: '采样',
    preparingModel: '正在准备模型…',
    consentTitle: '看模型如何思考——就在你的浏览器里实时运行',
    consentBodyLocal: '这个实时演示需要模型。当前没有托管模型——请选择本地模型目录来运行。所有计算都在你的设备上完成。',
    consentBodyHosted:
      '加载真实的 Qwen3.5-0.8B（约 1.6 GB）并运行一次前向传播，用卡片堆叠逐层回放。全部在你的设备上运行——数据不会离开浏览器。下面的章节不加载模型也完全可读。',
    ctaRetry: '重试',
    ctaChooseLocal: '选择本地模型',
    ctaLoadModel: '加载模型',
    helperLine: (totalSeconds) =>
      `真实推理、真实的逐层轨迹——以约 0.9–1.0 秒/层的速度通过卡片堆叠回放，让你看清每个子步骤。1× 速度下完整回放约 ${totalSeconds} 秒（½× 放慢细读，4× 快速浏览）。`,
    chipTokenize: '分词',
    chipEmbedding: '嵌入查表',
    chipFinalNorm: '最终 RMSNorm',
    chipLmHead: 'LM head → 248K 个 logits',
    chipSampling: '采样 → 下一个 token',
    chipChapter: (n) => `第 ${n} 章`,
    chipAria: (chapterNum, label) => `打开第 ${chapterNum} 章：${label}`,
    layersHeader: (layers, chapterNum) => `× ${layers} 层 · 第 ${chapterNum} 章（完整块）`,
    layerProgress: (layer, total) => `第 ${layer} / ${total} 层`,
    decoderLayer: '解码器层',
    fullAttnTitle: '全量 softmax 注意力层',
    linearAttnTitle: '线性注意力层（GatedDeltaNet）',
    subStages: {
      pre_norm: 'RMSNorm（前置）',
      qkv_proj: 'Q · K · V 投影',
      rope: '对 Q · K 应用 RoPE',
      gqa_split: '8 个 Q 头 · 2 个 KV 头（GQA）',
      attn_score: 'Q @ Kᵀ · 缩放',
      causal_mask: '因果掩码',
      softmax: 'Softmax',
      attn_v: 'Attn @ V · 输出投影',
      residual_attn: '+ 残差',
      post_norm: 'RMSNorm（后置）',
      mlp: 'MLP 模块（SwiGLU）',
      residual_mlp: '+ 残差',
      gdn_gates: 'b · a · c · g 门控',
      gdn_qkv: 'q · k · v 投影',
      gdn_recurrent: 'S ← α·S + β·k⊗v（循环状态）',
      gdn_output_gate: '门控输出',
      gdn_out_proj: '输出投影',
    },
    nextTokensLabel: '下一批 token →',
    revealCaption: (hasEcho, n) =>
      `${hasEcho ? '高亮为第一个新 token · ' : '高亮为第一个 token · '}共 ${n} 个预测 token`,
    footerStack:
      '观察堆叠：全量注意力层（青色）对所有历史 token 计算 Q · Kᵀ · softmax；线性层（琥珀色，GatedDeltaNet）维护一个循环状态——Qwen3.5 以 1:3 的比例交错使用它们。',
    footerLoop1: '这是循环的一轮。模型把预测出的 token 追加到末尾，再把整个堆叠跑一遍——但下一轮不会重算前缀：',
    footerLoopKvLink: (chapterNum) => `KV 缓存（第 ${chapterNum} 章）`,
    footerLoop2: '会被复用——前缀无需重算，因此生成每个新 token 都远比重新处理整个上下文便宜。生成就是这个前向传播的循环。',
    footerHonest1: '坦白说明：这张图展示的是',
    footerHonestGreedy: '贪心',
    footerHonest2: '补全——它永远选 logit 最高的那一个 token，所以这里相同的提示词每次给出相同的答案。真实的生成通常会从分布中',
    footerHonestSamplesLink: '采样',
    footerHonest3: '，因此同一个提示词每次运行的结果可能不同。展示的续写也被',
    footerHonestCapped: '截断',
    footerHonest4: (cap) => `在 ${cap} 个 token——原始贪心解码超过这个长度会陷入循环——所以这只是前几个预测的窗口，不是完整回复。完整图景及其局限见`,
    footerHonestWholeModelLink: '全模型章节',
    footerHonest5: '。',
  },

  landing: {
    tag: '真实 LLM · 100% 本地 · WebGPU',
    heroTitlePre: 'LLM 是如何',
    heroTitleEm: '工作',
    heroTitlePost: '的',
    heroSubBeforeCode:
      '在浏览器里完整运行一个真实的 0.8B 语言模型（Qwen3.5），然后逐层弄懂它的工作原理。无服务器、无 API key——由 ',
    heroSubAfterCode: ' 和 WebGPU 驱动。',
    voidAdAria: '托管于 Void Platform',
    voidEyebrow: '托管于',
    voidMark: 'Void Platform',
    voidCopy: '边缘资源 · WebGPU 隔离',
    specParams: '0.8B 参数',
    specParamsLabel: 'Qwen3.5 · 实时运行',
    specChapters: (n) => `${n} 个章节`,
    specChaptersLabel: '交互式课程',
    specLocal: '100% 本地',
    specLocalLabel: 'WebGPU · 无服务器',
    checkingModel: '正在检查模型…',
    chooseLocalModel: '选择本地模型',
    openChat: '打开对话 →',
    loadModel: (label) => `加载模型（${label}）`,
    localPickerTitle: '选择本地模型目录',
    startLearning: '开始学习 →',
    learnHint: (n) => `第一次接触 LLM？跟着 ${n} 个章节一步步弄懂这个模型的真实工作原理。`,
    jspaceLink: '打开 J-Space —— 逐层透视这个模型 →',
    mascotAlt: '水豚吉祥物',
    builtWith: '基于',
    hostedOnVoid: '托管于 Void',
    localModelLink: '本地模型…',
  },

  scaffolding: {
    minBadge: (minutes) => `${minutes} 分钟`,
    glossaryTitle: '术语表',
    glossaryCount: (n) => `${n} 个术语`,
    takeawaysTitle: '工程要点',
    tryThisTitle: '动手练习',
    showAnswer: '显示答案',
    hideAnswer: '隐藏答案',
    quickCheckTitle: '随堂测验',
    score: (correct, total) => `答对 ${correct} / ${total}`,
    resetAnswers: '重置答案',
    correctLabel: '正确',
    notQuiteLabel: '不对哦',
    correctAria: '正确答案',
    incorrectAria: '不正确',
    goDeeper: '深入本章',
  },

  pipeline: {
    eyebrow: '前向传播',
    chipTokenize: '分词',
    chipEmbedding: '嵌入查表',
    chipLayers: '× 24 层',
    chipLayersTitle: '混合堆叠：6 个全量注意力层 + 18 个更省算力的线性注意力层（GatedDeltaNet）',
    chipFinalNorm: '最终 RMSNorm',
    chipLmHead: 'LM head',
    chipSampling: '采样',
    wholePipeline: '· 全流程',
    summary: '分词 → 嵌入 → ×24 层 → 最终归一化 → LM head → 采样',
    describeAll: (summary) => `前向传播地图：本章覆盖整条流水线（${summary}）。`,
    describeMeta: (summary) => `前向传播地图（本章从单个步骤拉远视角）：${summary}。`,
    describeStages: (phrase, _plural, summary) => `前向传播地图：你正处于${phrase}阶段（${summary}）。`,
    stagePhrases: {
      tokenize: '分词',
      embed: '嵌入',
      block: 'Transformer 块',
      'final-norm': '最终归一化',
      'lm-head': 'LM head',
      sample: '采样',
    },
    phraseJoin: '和',
  },
};

export const UI_STRINGS: Record<Locale, UiStrings> = {
  en: EN,
  zh: ZH,
};

import { ChapterLink } from '../../scaffolding/ChapterLink';

import { Prose } from '../../Prose';
import { ChapterFrame } from '../../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../../scaffolding/learning-data';
import { BaseVsInstructDiagram } from '../../widgets/BaseVsInstructDiagram';
import { ChatTemplateExplorer } from '../../widgets/ChatTemplateExplorer';
import { DataScaleBar } from '../../widgets/DataScaleBar';
import { PreferencePair } from '../../widgets/PreferencePair';
import { TrainingStages } from '../../widgets/TrainingStages';

/**
 * 第 15 章（后训练）“从基础模型到助手”的简体中文译文 — 仅包含章节正文与
 * learning 数据。对一个以 ChatGPT 为心智模型的初学者来说，这是最大的认知
 * 缺口：基础模型只会自动补全；指令微调 + 偏好微调 + 对话模板才让它听从
 * 指令。纯讲解 + 静态小组件，不调用模型/worker——对话模板按字面展示。
 */

export const learning: ChapterLearningData = {
  chapterId: 'post-training',
  objective: '解释基础模型如何变成有用的助手——指令微调、偏好微调和对话模板——而架构本身不做任何改动。',
  problem:
    '前面讲的一切都是基础模型：纯粹的自动补全。但你实际使用的 LLM 会回答问题、听从指令。中间一定有什么把两者连了起来。',
  minutes: 8,
  glossary: [
    {
      term: 'pretraining',
      definition: '在数万亿 token 的通用文本上做下一个 token 预测——模型的知识来自这里。',
    },
    {
      term: 'instruction tuning (SFT)',
      definition: '在精心整理的（指令，回复）对上做有监督微调；教会模型以助手的角色作答。',
    },
    {
      term: 'RLHF',
      definition:
        'Reinforcement Learning from Human Feedback（基于人类反馈的强化学习）——用一个基于人类偏好比较训练出的奖励模型来训练主模型。',
    },
    {
      term: 'DPO',
      definition:
        'Direct Preference Optimization（直接偏好优化）——直接在"偏好 vs 拒绝"的回复对上优化，省去单独的奖励模型。',
    },
    {
      term: 'chat template',
      definition: '把多轮对话转换成模型实际读取的单一 token 字符串的格式约定。',
    },
    {
      term: 'special token',
      definition:
        '一种保留 token（如 <|im_start|>、<|im_end|>），用来标记结构——轮次边界、角色、回合结束——而不是字面文本。',
    },
    {
      term: 'tool calling',
      definition:
        '对话模板可以在 <tools>…</tools> 系统块中列出可用工具；模型随后输出结构化的 <tool_call><function=NAME><parameter=NAME>VALUE</parameter></function></tool_call> 而不是普通文本，应用执行它并把结果喂回去。在线对话中它就是 "tools" 开关。',
    },
  ],
  takeaways: [
    '基础模型只会接着写下去；是指令微调让它以助手的角色回应指令。',
    '后训练（先 SFT，再偏好微调）复用同一个模型，损失函数也大体相同——它塑造的是行为，并不增加新架构。',
    '一段聊天对话其实就是一个格式化的字符串：special token 标记轮到谁说话；模型在结尾的 assistant 标记之后开始生成，输出回合结束 token 即停止。',
  ],
  exercise: {
    prompt:
      '把对话模板小组件切换到“模型实际看到的原始文本”。模型开始生成之前紧挨着的是哪个 special token？又是哪个 token 告诉它该停下？',
    answer:
      '生成从 <|im_start|>assistant（外加一个换行）之后立刻开始；模型输出 <|im_end|> 时停止。一个细节：assistant 标记之后紧接的字面 token 并不是自由文本，而是推理块——思考开启时是 <think>\\n，关闭时是预先闭合的 <think>\\n\\n</think>\\n\\n——之后答案才真正开始。正是指令微调教会了模型输出这个回合结束 token，而不是滔滔不绝地编出一段假的下一轮对话。',
  },
  quiz: [
    {
      id: 'q1-base-behavior',
      prompt: '如果你向一个基础（仅预训练的）模型输入一个问题，它会做什么？',
      options: [
        { id: 'a', label: '给出有帮助的回答——回答问题是架构自带的能力。' },
        {
          id: 'b',
          label: '按网络上任何看起来合理的方式继续这段文本，未必是一个有用的回答。',
        },
        { id: 'c', label: '返回错误，因为它需要先有对话模板。' },
      ],
      correctId: 'b',
      explanation:
        '基础模型就是自动补全。没有任何东西教过它“问题后面应该跟一个有用的回答”——这种行为来自指令微调。',
    },
    {
      id: 'q2-sft',
      prompt: '指令微调（SFT）与预训练的区别是什么？',
      options: [
        { id: 'a', label: '它使用完全不同的模型架构。' },
        {
          id: 'b',
          label: '它使用同样的下一个 token 损失，但数据换成一小批精心整理的（指令，回复）对。',
        },
        { id: 'c', label: '它用强化学习取代了下一个 token 预测。' },
      ],
      correctId: 'b',
      explanation: 'SFT 用的是同样的前向传播和同样的交叉熵损失；变的只有数据——精心整理的指令跟随示例。',
    },
    {
      id: 'q3-special-tokens',
      prompt: '在对话模板中，<|im_start|> 和 <|im_end|> 这类 special token 的作用是什么？',
      options: [
        { id: 'a', label: '它们给会话加密，让模型读不到内容。' },
        {
          id: 'b',
          label: '它们标记轮次边界和角色，使一个扁平的 token 字符串能够编码多轮对话。',
        },
        { id: 'c', label: '它们是用户输入的普通词语。' },
      ],
      correctId: 'b',
      explanation:
        '模型看到的永远只是一个扁平的 token 字符串；special token 划定轮次边界，并标明 assistant 回合从哪里开始、到哪里结束。',
    },
  ],
};

export function PostTrainingChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>从基础模型到助手</h1>
        <p>
          这里有一个让几乎所有人都措手不及的事实：
          <ChapterLink chapterId="training">训练一章</ChapterLink>
          描述的那个模型——一个纯粹的下一个 token 预测器——并<em>不会</em>表现得像 ChatGPT。把 "What is 2 + 2?"
          输进一个原始的基础模型，它可能接着写出另一个问题、一串家庭作业题，或者任何在网络上有可能跟在这串文字后面的东西。它不会
          <em>回答</em>
          ，因为从来没有任何东西教过它：问题后面应该跟一个有用的回复。三个后训练阶段解决了这个问题——而且没有一个会动架构。先把这道鸿沟摆在眼前，左右对照：
        </p>

        <BaseVsInstructDiagram />

        <TrainingStages />

        <h2>指令微调（SFT）</h2>
        <p>
          第一步修复最简单。拿到预训练好的模型继续训练——同样的前向传播、同样的下一个 token
          交叉熵——但数据换成一小批由人编写或审核过的<strong>（指令，回复）</strong>
          对。经过几千到几百万条示例之后，模型学会了这个任务的形状：当眼前的文本是一条用户指令时，最可能的续写是一段以助手口吻给出的有用回复。这就是
          <strong>有监督微调</strong>（supervised fine-tuning），简称
          SFT。它相对预训练而言微不足道——几乎不增加新知识；它教的是模型如何<em>使用</em>已有的知识。
        </p>
        <p>
          关于损失的一个细节：<em>用户</em>的 token
          不计损失——它们被掩掉了，和训练一章里不计分的最后一个位置一模一样——只有助手回复的 token
          参与计分。否则模型也会被训练去模仿用户，而不是学会回答用户。
        </p>

        <DataScaleBar />

        <h2>偏好微调（RLHF / DPO）</h2>
        <p>
          SFT 让模型听从指令；<strong>偏好微调</strong>让它听得<em>更好</em>
          ——更有帮助、更诚实、更不容易给出有害或敷衍的回答。人类比较候选回复（"A 比 B
          好"），模型则被训练去偏向人们偏好的回复。先快速给个定义：<strong>强化学习（RL）</strong>=
          尝试各种输出、给每个输出打分、让高分行为更可能出现——从试错中学习，而不是从带标注的目标中学习。
          <strong>RLHF</strong> 用一个独立的<strong>奖励模型</strong>
          来做这件事——那是第二个模型，在人类比较数据上训练，用来预测人们会有多喜欢某个回复，主模型再被优化去在它上面拿高分——并配合强化学习；
          <strong>DPO</strong> 则直接优化偏好本身，跳过奖励模型。DPO 不是完整的 RL，也不是下一个 token
          交叉熵：它是一个作用在<strong>（偏好，拒绝）</strong>
          回复对上的偏好（分类式）损失，单纯地提高模型对偏好回复相对于被拒回复的相对似然。所以这个阶段确实偏离了你一路看到的朴素下一个
          token 损失——但就 DPO 而言，它仍是固定数据集上的有监督损失，而不是经典 RL 那种试错式 rollout。
        </p>
        <p>
          选出人类更偏好的那条回复，连按几次更新，看看“在偏好上训练”到底做了什么。给个尺度感：典型的偏好数据集约有
          10K–1M 个比较对——与 SFT 同一个数量级，和预训练的数万亿 token 完全不是一回事。
        </p>
        <PreferencePair />

        <h2>对话模板：多轮对话如何接线</h2>
        <p>
          在这一切过程中，模型看到的仍然只是一个<em>扁平的 token 字符串</em>
          ——它天生没有“消息”或“角色”的概念。<strong>对话模板</strong>就是把多轮对话压平成那个字符串的约定，用
          <ChapterLink chapterId="tokenization">special token</ChapterLink>
          标记每一轮从哪里开始、是谁在说话，以及一轮在哪里结束。
        </p>
        <ChatTemplateExplorer />
        <p>
          值得留意的是，“聊天机器人”的那种感觉，有相当一部分来自这种框定（framing），而非来自训练。基础模型只会自动补全。但只要你给它——哪怕是一个纯基础模型——一个铺设场景的{' '}
          <strong>system prompt</strong>，比如"the following is a conversation between a curious user and a knowledgeable,
          helpful assistant"，然后开始用户这一轮，最合理的续写就已经是一段助手口吻的回复了——纯粹因为周围的文字暗示了这一点。2020
          年早期的 GPT-3 演示正是这样工作的：GPT-3 是一个基础模型，人们仅凭写一段生动的前言加一个塑造得当的提示，就能从它身上诱导出类似助手的行为；让它变得
          <em>可靠</em>的有监督微调和偏好微调是后来才加上的。你在上面小组件里看到的 Qwen system
          块很简短——只是一句简短的指令，比如"You are a helpful assistant"——所以它只完成了那种框定工作的{' '}
          <em>一部分</em>，剩下的交给 SFT
          和偏好微调。框定能让你拿到第一句合理的回复；而后训练才是让助手每次都准时出现、而不是只在场景恰好铺设妥当时才出现的关键。
        </p>
        <p>
          当你在这个应用里和模型聊天时，它会替你把对话包装成同样的格式，在末尾追加一个{' '}
          <code>{'<|im_start|>assistant'}</code>，然后让模型生成回答——模型一输出 <code>{'<|im_end|>'}</code>{' '}
          就立刻停止。正是 SFT 教会了模型<em>主动输出</em>这个回合结束 token，而不是编造一条假的下一轮用户消息。
        </p>
        <p>
          同一个模板还接入了另外两样东西，你会在在线对话里看到它们以开关（pill）的形式出现。<strong>think</strong>{' '}
          开关控制推理块：在 assistant 标记之后，模板会注入 <code>{'<think>\\n'}</code>
          （思考开启——模型先推理，再在给出答案前闭合 <code>{'</think>'}</code>），或一个预先闭合的{' '}
          <code>{'<think>\\n\\n</think>\\n\\n'}</code>（思考关闭）。<strong>tools</strong> 开关则添加一个{' '}
          <code>{'<tools>…</tools>'}</code>{' '}
          系统块，描述模型可以调用的函数；模型随后输出的不是普通文本，而是一个结构化调用——{' '}
          <code>{'<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>'}</code> ——
          应用执行它，再把结果作为新的一轮喂回去。两者都纯粹是格式约定：网络没有任何改动；是 SFT 教会了模型遵守它们。
        </p>

        <h2>它仍然是同一个模型</h2>
        <p>
          要抓牢的一点是：这一切都没有改动网络。同样的 24 层，同样的注意力和 MLP 块，同样的前向传播为下一个 token 产出
          logits。后训练只是轻推了<em>权重</em>
          ，让这个函数的输出长成我们想要的样子，再给输入裹上一层对话格式。你对话的那个“助手”，就是其他每一章里的那个基础模型——披着一件对话模板，权重被温和地引向“乐于助人”。
        </p>
      </Prose>
    </ChapterFrame>
  );
}

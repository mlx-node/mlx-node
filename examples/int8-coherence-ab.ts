#!/usr/bin/env node
// Validate int8 W8A8 PREFILL coherence now that the metallib garbage is fixed.
// Uses a long (>=256 token) prompt so the int8 prefill path actually fires
// (M = batch*seq >= 256), then greedily decodes. Run by the harness twice:
// once with MLX_INT8_PREFILL[_QKVZ]=1, once unset, to compare coherence.
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const model = process.argv[2] ?? '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16';
// ~300+ token prompt to push prefill M over the 256 int8 threshold.
const longPrompt =
  'Read the following passage carefully and then answer the question at the end.\n\n' +
  'The history of computing hardware spans the development of devices from simple ' +
  'mechanical aids to modern general-purpose computers. Early counting devices such ' +
  'as the abacus gave way to mechanical calculators built by Pascal and Leibniz in ' +
  'the seventeenth century. In the nineteenth century, Charles Babbage designed the ' +
  'Analytical Engine, a mechanical general-purpose computer, while Ada Lovelace wrote ' +
  'what is often considered the first algorithm intended for such a machine. The ' +
  'twentieth century brought vacuum-tube machines like ENIAC, then the transistor, ' +
  'then the integrated circuit, each step shrinking size and cost while multiplying ' +
  'speed. The microprocessor, introduced in the early nineteen-seventies, placed an ' +
  'entire central processing unit on a single chip and made the personal computer ' +
  'possible. Decades of exponential improvement, often summarized by Moore’s law, ' +
  'drove computers from room-sized installations to pocket-sized devices more powerful ' +
  'than the supercomputers of earlier generations. More recently, specialized ' +
  'accelerators such as graphics processing units and neural processing units have ' +
  'reshaped how demanding numerical workloads, especially machine learning, are run.\n\n' +
  'Question: In one short sentence, what single invention of the early 1970s made the ' +
  'personal computer possible?';

const flags = `INT8_PREFILL=${process.env.MLX_INT8_PREFILL ?? 'unset'} QKVZ=${process.env.MLX_INT8_PREFILL_QKVZ ?? 'unset'}`;
const loaded = await loadModel(model);
const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});
const res = await session.send(longPrompt, {
  config: { maxNewTokens: 40, temperature: 0, reportPerformance: true, reuseCache: false },
});
const r = res as any;
console.log(`MODEL ${model}  [${flags}]`);
console.log(`  promptTokens=${r.promptTokens ?? r.numPromptTokens ?? '?'} numTokens=${r.numTokens} finishReason=${r.finishReason}`);
console.log(`  rawText=${JSON.stringify(r.rawText)}`);
const ok = /[A-Za-z].*[A-Za-z].*[A-Za-z]/.test(String(r.rawText)) && r.finishReason !== 'repetition';
console.log(`  VERDICT=${ok ? 'LOOKS-COHERENT' : 'GARBAGE/REPETITION'}`);

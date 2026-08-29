/**
 * Ordered auto-detection table mapping a checkpoint's config.json to the
 * converter model_type. Row order is load-bearing and is NOT the runtime
 * loader's probe order: a raw model_type row that precedes an
 * architecture-probe row wins even when the probe would also match (e.g.
 * model_type 'gemma4_text' with a Gemma4Unified architecture resolves
 * 'gemma4', never 'gemma4_unified'). The convert-detect parity test pins the
 * overlaps and gates every native convertible model_type against this table.
 */
export interface ConvertDetectRow {
  /** Matches when config.model_type is exactly one of these raw strings. */
  readonly rawModelTypes?: readonly string[];
  /** Matches when config.architectures (an array) contains this class name. */
  readonly architecture?: string;
  /** Converter model_type to emit; defaults to the matched raw model_type. */
  readonly out?: string;
}

export const CONVERT_DETECT: readonly ConvertDetectRow[] = [
  // The Nemotron architecture is authoritative (native parser + runtime
  // registry probe): it is checked BEFORE every model_type row so a
  // stale-but-recognized model_type (e.g. qwen3_5) cannot route a Nemotron
  // checkpoint to another family's sanitizer.
  { architecture: 'NemotronHForCausalLM', out: 'nemotron_h' },
  { rawModelTypes: ['paddleocr_vl'], out: 'paddleocr-vl' },
  // Qianfan-OCR checkpoints use InternVL's historical raw model_type.
  // The runtime registry maps exactly both strings to QianfanOCRModel;
  // canonicalize them to the converter recipe name so the native
  // dense-only quantization guard cannot be bypassed when -m is omitted.
  { rawModelTypes: ['internvl_chat', 'qianfan-ocr'], out: 'qianfan-ocr' },
  { rawModelTypes: ['qwen3_asr'] },
  { rawModelTypes: ['qwen3_5_moe', 'qwen3_5'] },
  { rawModelTypes: ['gemma4', 'gemma4_text'], out: 'gemma4' },
  // Pass the raw 'gemma4_unified' string through unchanged. Native
  // recipe_for resolves it to the shared Gemma4Recipe, so every
  // recipe-keyed code path (sym8_supported, sanitizer, embed_quantizable,
  // mtp_policy, etc.) behaves identically to 'gemma4'. Collapsing it to
  // 'gemma4' here would (a) make the native recipe_for("gemma4_unified")
  // arm dead code, and (b) misroute a unified checkpoint that carries
  // gemma-QAT metadata into the E2B-only prequantized importer, whose
  // gate keys on the exact string "gemma4" (and which then hard-errors in
  // validate_e2b_qat_schedule). The exact-"gemma4" gate must not match
  // unified, so the raw string has to reach the native side.
  //
  // The architecture probe (no `model_type`, only
  // `architectures: ['Gemma4UnifiedForConditionalGeneration']`) mirrors
  // the runtime loader, which also flags this shape as unified
  // (model-loader.ts maps it to gemma4; persistence.rs parse_config sets
  // is_unified on EITHER model_type == "gemma4_unified" OR that
  // architecture). Without this, an architecture-only config would leave
  // modelType undefined and skip Gemma4Recipe::sanitize, producing
  // unloadable output. It maps to 'gemma4_unified' (not 'gemma4') so the
  // E2B-importer gate above still cannot match it.
  {
    rawModelTypes: ['gemma4_unified'],
    architecture: 'Gemma4UnifiedForConditionalGeneration',
    out: 'gemma4_unified',
  },
  { rawModelTypes: ['lfm2_moe', 'lfm2'] },
  { rawModelTypes: ['openai_privacy_filter'], out: 'privacy-filter' },
  { rawModelTypes: ['muse_glimmer'] },
  // The architecture row is FIRST in this table (the architecture is
  // authoritative); this row covers checkpoints that declare the plain
  // model_type.
  { rawModelTypes: ['nemotron_h'] },
];

/**
 * Resolve the converter model_type for a parsed config.json, or `undefined`
 * when no row matches — an unrecognized model_type deliberately converts via
 * the generic family-less pass (the qwen3 flow), so this must never throw.
 */
export function detectConvertModelType(config: unknown): string | undefined {
  const cfg = config as { model_type?: unknown; architectures?: unknown } | null | undefined;
  const raw = typeof cfg?.model_type === 'string' ? cfg.model_type : undefined;
  const architectures = Array.isArray(cfg?.architectures) ? cfg.architectures : undefined;
  for (const row of CONVERT_DETECT) {
    const byRaw = raw !== undefined && row.rawModelTypes?.includes(raw) === true;
    const byArch = row.architecture !== undefined && architectures?.includes(row.architecture) === true;
    if (byRaw || byArch) return row.out ?? raw;
  }
  return undefined;
}

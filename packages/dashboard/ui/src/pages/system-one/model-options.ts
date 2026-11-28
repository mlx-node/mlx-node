import type { ModelBaseOption, ModelVariantOption } from '../../components/arc-x/model-picker/model-picker';
import { type ClairicImageRotation, isPlainObject, type MediaAttachment, nextDraftId } from '../../lib/system-one';
import type { CatalogItem, SystemOneModelInfo } from '../../lib/types';

type PickerState = ModelBaseOption['state'];

const STATE_RANK: Record<PickerState, number> = { loaded: 0, loading: 1, idle: 2 };

const FAMILY_ORDER: Readonly<Record<string, number>> = {
  'Clairic Flash': 0,
  Clairic: 1,
  'CLEF Flash': 2,
  CLEF: 3,
};

/** Strip a trailing parenthetical size tag such as `" (9B)"` from a catalog label, without regex. */
function stripParentheticalSuffix(label: string): string {
  const trimmed = label.trim();
  if (!trimmed.endsWith(')')) return trimmed;
  const open = trimmed.lastIndexOf('(');
  if (open <= 0) return trimmed;
  return trimmed.slice(0, open).trim();
}

function splitPathSegments(rawPath: string): string[] {
  const segments: string[] = [];
  let start = 0;
  for (let i = 0; i <= rawPath.length; i += 1) {
    if (i === rawPath.length || rawPath[i] === '/' || rawPath[i] === '\\') {
      if (i > start) {
        const part = rawPath.slice(start, i).trim();
        if (part !== '' && part !== '.' && part !== '..') segments.push(part);
      }
      start = i + 1;
    }
  }
  return segments;
}

function pathSlugCandidates(modelPath: string): string[] {
  const segments = splitPathSegments(modelPath);
  if (segments.length === 0) return [];
  const candidates: string[] = [segments[segments.length - 1]!];
  if (segments.length >= 2) candidates.push(segments[segments.length - 2]!);
  if (segments.length >= 3) candidates.push(segments[segments.length - 3]!);
  return candidates;
}

function matchCatalogByMember(
  bySlug: ReadonlyMap<string, CatalogItem>,
  model: SystemOneModelInfo,
): CatalogItem | undefined {
  const direct = bySlug.get(model.id.trim().toLowerCase()) ?? bySlug.get(model.name.trim().toLowerCase());
  if (direct !== undefined) return direct;
  for (const candidate of pathSlugCandidates(model.path)) {
    const hit = bySlug.get(candidate.toLowerCase());
    if (hit !== undefined) return hit;
  }
  return undefined;
}

function systemOneCatalogIndex(catalog: readonly CatalogItem[]): {
  byLabel: ReadonlyMap<string, CatalogItem>;
  bySlug: ReadonlyMap<string, CatalogItem>;
} {
  const byLabel = new Map<string, CatalogItem>();
  const bySlug = new Map<string, CatalogItem>();
  for (const item of catalog) {
    if (item.category !== 'systemone' || item.hidden === true) continue;
    const stripped = stripParentheticalSuffix(item.label).toLowerCase();
    if (stripped !== '' && !byLabel.has(stripped)) byLabel.set(stripped, item);
    const exact = item.label.trim().toLowerCase();
    if (exact !== '' && !byLabel.has(exact)) byLabel.set(exact, item);
    for (const build of item.builds) {
      const slug = build.slug.trim().toLowerCase();
      if (slug !== '' && !bySlug.has(slug)) bySlug.set(slug, item);
    }
  }
  return { byLabel, bySlug };
}

function stripLeadingSeparators(text: string): string {
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '-' || ch === '_' || ch === ' ') i += 1;
    else break;
  }
  return text.slice(i);
}

function matchesPrefixToken(lower: string, prefix: string): boolean {
  if (lower === prefix) return true;
  if (!lower.startsWith(prefix)) return false;
  const next = lower[prefix.length];
  return next === '-' || next === '_' || next === ' ';
}

/** Split a System One model name into its logical base family and variant label without regex. */
function classifySystemOneModel(model: SystemOneModelInfo): { base: string; variant: string } {
  const raw = model.name.trim();
  const lower = raw.toLowerCase();

  const families: ReadonlyArray<readonly [string, string]> = [
    ['clairic-flash', 'Clairic Flash'],
    ['clairic_flash', 'Clairic Flash'],
    ['clairic flash', 'Clairic Flash'],
    ['clairic', 'Clairic'],
    ['clef-flash', 'CLEF Flash'],
    ['clef_flash', 'CLEF Flash'],
    ['clef flash', 'CLEF Flash'],
    ['clef', 'CLEF'],
  ];

  for (const [prefix, base] of families) {
    if (matchesPrefixToken(lower, prefix)) {
      const rest = stripLeadingSeparators(raw.slice(prefix.length));
      return { base, variant: rest.length > 0 ? rest : 'BF16' };
    }
  }

  return { base: raw || model.id, variant: model.format.toUpperCase() };
}

function formatContextFact(contextWindow: number): string | null {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) return null;
  if (contextWindow >= 1024) return `${String(Math.round(contextWindow / 1024))}k ctx`;
  return `${String(contextWindow)} ctx`;
}

function variantFacts(model: SystemOneModelInfo): string[] {
  const facts: string[] = [model.format.toUpperCase()];
  const ctx = formatContextFact(model.contextWindow);
  if (ctx !== null) facts.push(ctx);
  if (model.supportsImages) facts.push('Vision');
  return facts;
}

/**
 * Group installed System One models into `ModelPicker` base rows & quantization variants.
 * Loaded models sort first, followed by Clairic Flash, Clairic, CLEF Flash, CLEF, then alphabetical.
 */
export function systemOneModelBases(
  models: readonly SystemOneModelInfo[],
  selectedId: string | null,
  running: boolean,
  catalog?: readonly CatalogItem[],
): ModelBaseOption[] {
  const catalogIndex = catalog === undefined ? null : systemOneCatalogIndex(catalog);
  const stateOf = (model: SystemOneModelInfo): PickerState => {
    if (model.resident) return 'loaded';
    if (running && selectedId !== null && model.id === selectedId) return 'loading';
    return 'idle';
  };

  const groups = new Map<string, Array<{ model: SystemOneModelInfo; label: string }>>();
  for (const model of models) {
    const { base, variant } = classifySystemOneModel(model);
    const list = groups.get(base);
    if (list === undefined) groups.set(base, [{ model, label: variant }]);
    else list.push({ model, label: variant });
  }

  const entries: Array<{ option: ModelBaseOption; rank: number; familyRank: number }> = [];

  for (const [base, members] of groups) {
    // Ensure variant labels are unique within their base group.
    const counts = new Map<string, number>();
    for (const item of members) counts.set(item.label, (counts.get(item.label) ?? 0) + 1);
    const labeled = members.map((item) => ({
      model: item.model,
      label: (counts.get(item.label) ?? 0) > 1 ? `${item.label} (${item.model.format.toUpperCase()})` : item.label,
    }));

    const ordered = [...labeled].sort((a, b) => {
      const rankDiff = STATE_RANK[stateOf(a.model)] - STATE_RANK[stateOf(b.model)];
      if (rankDiff !== 0) return rankDiff;
      const al = a.label.toLowerCase();
      const bl = b.label.toLowerCase();
      return al < bl ? -1 : al > bl ? 1 : 0;
    });

    const variants: ModelVariantOption[] = ordered.map(({ model, label }) => ({
      id: model.id,
      label,
      state: stateOf(model),
      facts: variantFacts(model),
    }));

    const loadingVariant = variants.find((v) => v.state === 'loading');
    const loadedVariant = variants.find((v) => v.state === 'loaded');
    const stateVariant = loadingVariant ?? loadedVariant;

    let baseFacts: string[];
    if (variants.length === 1) {
      baseFacts = [...variants[0]!.facts];
    } else {
      baseFacts = [`${String(variants.length)} variants`];
      const maxCtx = Math.max(0, ...members.map((m) => m.model.contextWindow));
      const ctxFact = formatContextFact(maxCtx);
      if (ctxFact !== null) baseFacts.push(ctxFact);
      if (members.some((m) => m.model.supportsImages)) baseFacts.push('Vision');
    }

    const preselect =
      [selectedId, loadedVariant?.id, loadingVariant?.id].find(
        (id): id is string => id !== null && id !== undefined && variants.some((v) => v.id === id),
      ) ?? variants[0]!.id;

    const option: ModelBaseOption = {
      base,
      state: stateVariant?.state ?? 'idle',
      facts: baseFacts,
      variants,
      preselect,
    };
    const catalogEntry =
      catalogIndex === null
        ? undefined
        : (catalogIndex.byLabel.get(base.toLowerCase()) ??
          members.reduce<CatalogItem | undefined>(
            (found, item) => found ?? matchCatalogByMember(catalogIndex.bySlug, item.model),
            undefined,
          ));
    if (catalogEntry !== undefined) {
      option.recommended = true;
      if (catalogEntry.description.trim() !== '') option.note = catalogEntry.description;
    } else if (catalogIndex === null && Object.hasOwn(FAMILY_ORDER, base)) {
      option.recommended = true;
    }
    if (stateVariant !== undefined && variants.length > 1) {
      option.stateVariant = stateVariant.label;
    }

    entries.push({
      option,
      rank: loadedVariant !== undefined ? 0 : loadingVariant !== undefined ? 1 : 2,
      familyRank: FAMILY_ORDER[base] ?? 100,
    });
  }

  return entries
    .sort((a, b) => {
      if (a.rank !== b.rank) return a.rank - b.rank;
      if (a.familyRank !== b.familyRank) return a.familyRank - b.familyRank;
      const al = a.option.base.toLowerCase();
      const bl = b.option.base.toLowerCase();
      return al < bl ? -1 : al > bl ? 1 : 0;
    })
    .map((entry) => entry.option);
}

function normalizeRotation(raw: unknown): ClairicImageRotation {
  if (raw === 90 || raw === 180 || raw === 270) return raw;
  return 0;
}

function estimateBase64Bytes(base64: string): number {
  let padding = 0;
  if (base64.endsWith('==')) padding = 2;
  else if (base64.endsWith('=')) padding = 1;
  return Math.max(0, Math.floor((base64.length * 3) / 4) - padding);
}

function parseDataUrl(text: string): { contentType: string; base64: string } | null {
  if (!text.startsWith('data:')) return null;
  const marker = ';base64,';
  const idx = text.indexOf(marker);
  if (idx === -1) return null;
  const contentType = text.slice(5, idx).trim();
  const base64 = text.slice(idx + marker.length);
  if (contentType === '' || base64 === '') return null;
  return { contentType, base64 };
}

function isOmittedStringPlaceholder(value: string): boolean {
  return value.startsWith('[omitted media payload:');
}

/**
 * Reconstruct retained media attachments from a history entry's `request` payload.
 * Large attachments stripped by history sanitization are counted in `omittedCount`
 * so callers can inform the user without leaving unrelated draft files behind.
 */
export function restoreHistoryMedia(request: Record<string, unknown>): {
  attachments: MediaAttachment[];
  omittedCount: number;
} {
  const attachments: MediaAttachment[] = [];
  let omittedCount = 0;

  const ingest = (items: unknown, kind: 'image' | 'pdf'): void => {
    if (!Array.isArray(items)) return;
    let seq = 0;
    for (const item of items) {
      seq += 1;
      const defaultMime = kind === 'pdf' ? 'application/pdf' : 'image/png';
      const defaultName = kind === 'pdf' ? `document-${String(seq)}.pdf` : `image-${String(seq)}.png`;

      if (typeof item === 'string') {
        if (isOmittedStringPlaceholder(item)) {
          omittedCount += 1;
          continue;
        }
        const parsed = parseDataUrl(item);
        if (parsed !== null) {
          attachments.push({
            id: nextDraftId(kind),
            name: defaultName,
            contentType: parsed.contentType,
            sizeBytes: estimateBase64Bytes(parsed.base64),
            base64: parsed.base64,
            dataUrl: item,
            rotation: 0,
            kind,
          });
        }
        continue;
      }

      if (!isPlainObject(item)) continue;
      if (item.omitted === true) {
        omittedCount += 1;
        continue;
      }

      if (typeof item.base64 === 'string' && item.base64.length > 0) {
        const contentType =
          typeof item.content_type === 'string' && item.content_type.length > 0 ? item.content_type : defaultMime;
        const name = typeof item.name === 'string' && item.name.length > 0 ? item.name : defaultName;
        attachments.push({
          id: nextDraftId(kind),
          name,
          contentType,
          sizeBytes: estimateBase64Bytes(item.base64),
          base64: item.base64,
          dataUrl: `data:${contentType};base64,${item.base64}`,
          rotation: kind === 'image' ? normalizeRotation(item.rotation) : 0,
          kind,
        });
      }
    }
  };

  ingest(request.images, 'image');
  ingest(request.documents, 'pdf');

  return { attachments, omittedCount };
}

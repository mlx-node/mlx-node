import {
  AlertTriangle,
  ArrowRight,
  BookmarkPlus,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  FileText,
  FileUp,
  ImagePlus,
  Layers,
  Play,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  Square,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { type ChangeEvent, type ReactElement, useCallback, useMemo, useRef, useState } from 'react';

import { CodePanel } from '../../components/arc-x/code-panel/code-panel';
import { CompactField } from '../../components/arc-x/compact-field/compact-field';
import { Disclosure } from '../../components/arc-x/disclosure/disclosure';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { MediaDropzone } from '../../components/arc-x/media-dropzone/media-dropzone';
import { Panel } from '../../components/arc-x/panel/panel';
import { Meter } from '../../components/arc-x/split-bar/split-bar';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { WaitOrb } from '../../components/arc-x/wait-orb/wait-orb';
import { Alert } from '../../components/arc/alert/alert';
import { Badge, type BadgeTone } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { Drawer, DrawerContent, useDrawerItem } from '../../components/arc/drawer/drawer';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { Input } from '../../components/arc/input/input';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/arc/popover/popover';
import { Select } from '../../components/arc/select/select';
import { Tooltip } from '../../components/arc/tooltip/tooltip';
import { formatBytes, formatDuration, formatPercent } from '../../lib/format';
import {
  type AnswerRouteResult,
  type BatchRow,
  type BatchSampleDataset,
  buildClairicPayload,
  CLAIRIC_MAX_MEDIA_BYTES,
  type ClefAnswer,
  type ClefQuestionDraft,
  compareExpectedAnswer,
  createBatchRow,
  type DecisionPolicy,
  DEFAULT_DECISION_POLICY,
  evaluateClairicAnswerRoute,
  evaluateClefAnswerRoute,
  exportBatchToCsv,
  fieldAccentColor,
  type FieldDraft,
  formatAnswerSummary,
  isPlainObject,
  type MediaAttachment,
  nextDraftId,
  normalizePolicy,
  parseBatchImportText,
  type RouteKind,
  summarizeRouting,
  SYSTEM_ONE_BATCH_SAMPLES,
  type SystemOneMode,
} from '../../lib/system-one';
import { toast } from '../../lib/toast';
import styles from './batch-view.module.css';
import { ClefBuilder } from './clef-builder';
import { ResultsView } from './results-view';
import { SchemaBuilder } from './schema-builder';

export interface BatchViewProps {
  mode: SystemOneMode;
  modelName: string | null;
  fields: readonly FieldDraft[];
  clefQuestions: readonly ClefQuestionDraft[];
  fieldWarnings: Readonly<Record<string, string[]>>;
  clefWarnings: Readonly<Record<string, string[]>>;
  policy: DecisionPolicy;
  onPolicyChange: (next: DecisionPolicy) => void;
  rows: readonly BatchRow[];
  onRowsChange: (next: BatchRow[] | ((prev: BatchRow[]) => BatchRow[])) => void;
  primaryColumn: string;
  onPrimaryColumnChange: (col: string) => void;
  running: boolean;
  runningRowId: string | null;
  canRun: boolean;
  runTooltip: string;
  onRunAll: () => void;
  onRunRow: (rowId: string) => void;
  onCancel: () => void;
  onLoadSample: (sample: BatchSampleDataset) => void;
  onFieldsChange: (next: FieldDraft[]) => void;
  onClefQuestionsChange: (next: ClefQuestionDraft[]) => void;
  onSaveSpec?: () => void;
}

type FilterMode = 'all' | 'review' | 'errors';

const ACCEPTED_MEDIA_MIME = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf']);

const noop = (): void => {};

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') resolve(reader.result);
      else reject(new Error('Could not read file'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Could not read file'));
    reader.readAsDataURL(file);
  });
}

function normalizeMediaMime(file: File): string {
  if (file.type !== '') return file.type.toLowerCase();
  const lower = file.name.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.pdf')) return 'application/pdf';
  return '';
}

function stripExtension(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
}

function compactPreview(text: string): string {
  const flat = text.split('\r').join(' ').split('\n').join(' ').split('\t').join(' ').trim();
  if (flat.length <= 120) return flat;
  return `${flat.slice(0, 117)}…`;
}

function isClefAnswerObject(value: unknown): value is ClefAnswer {
  return isPlainObject(value) && (value.type === 'noul' || value.type === 'choice' || value.type === 'score');
}

function routeCellAnswer(answer: unknown, mode: SystemOneMode, policy: DecisionPolicy): AnswerRouteResult {
  if (mode === 'decide' && isClefAnswerObject(answer)) {
    return evaluateClefAnswerRoute(answer, policy);
  }
  if (isClefAnswerObject(answer)) {
    return evaluateClefAnswerRoute(answer, policy);
  }
  return evaluateClairicAnswerRoute(answer, policy);
}

function shortRouteLabel(route: RouteKind): string {
  if (route === 'auto-yes') return 'Auto-Yes';
  if (route === 'auto-no') return 'Auto-No';
  if (route === 'auto') return 'Auto';
  if (route === 'missing') return 'Missing';
  return 'Review';
}

function triggerFileDownload(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function isPolicyDefault(policy: DecisionPolicy, mode: SystemOneMode): boolean {
  if (mode === 'decide') {
    return (
      Math.round(policy.noThreshold * 100) === Math.round(DEFAULT_DECISION_POLICY.noThreshold * 100) &&
      Math.round(policy.yesThreshold * 100) === Math.round(DEFAULT_DECISION_POLICY.yesThreshold * 100) &&
      Math.round(policy.minConfidence * 100) === Math.round(DEFAULT_DECISION_POLICY.minConfidence * 100)
    );
  }
  return Math.round(policy.minConfidence * 100) === Math.round(DEFAULT_DECISION_POLICY.minConfidence * 100);
}

interface ComputedRowMeta {
  index: number;
  answers: Record<string, unknown> | null;
  reviewCount: number;
  autoCount: number;
  missingCount: number;
  totalAnswers: number;
  autoRouted: boolean;
  needsReview: boolean;
  expectedMatch: boolean | null;
}

/** System One Batch Evaluation Studio: multi-case execution, live threshold routing, agreement scoring, and CSV/JSON import/export. */
export function BatchView({
  mode,
  modelName,
  fields,
  clefQuestions,
  fieldWarnings,
  clefWarnings,
  policy,
  onPolicyChange,
  rows,
  onRowsChange,
  primaryColumn,
  onPrimaryColumnChange,
  running,
  runningRowId,
  canRun,
  runTooltip,
  onRunAll,
  onRunRow,
  onCancel,
  onLoadSample,
  onFieldsChange,
  onClefQuestionsChange,
  onSaveSpec,
}: BatchViewProps): ReactElement {
  const [filter, setFilter] = useState<FilterMode>('all');
  const [samplesOpen, setSamplesOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState('');
  const [importError, setImportError] = useState<string | null>(null);
  const [specDrawerOpen, setSpecDrawerOpen] = useState(false);
  const [inspectRowId, setInspectRowId] = useState<string | null>(null);

  const textFileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaFileInputRef = useRef<HTMLInputElement | null>(null);
  const datasetEpochRef = useRef(0);

  const activeSpecs = useMemo(() => {
    if (mode === 'decide') {
      return clefQuestions.map((q, idx) => ({
        id: q.id,
        name: q.name.trim(),
        color: fieldAccentColor(idx),
        warnings: clefWarnings[q.id] ?? [],
      }));
    }
    return fields.map((f, idx) => ({
      id: f.id,
      name: f.name.trim(),
      color: fieldAccentColor(idx),
      warnings: fieldWarnings[f.id] ?? [],
    }));
  }, [mode, clefQuestions, fields, clefWarnings, fieldWarnings]);

  const columns = useMemo(() => {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const spec of activeSpecs) {
      if (spec.name !== '' && !seen.has(spec.name)) {
        seen.add(spec.name);
        out.push(spec.name);
      }
    }
    if (out.length === 0) {
      for (const row of rows) {
        if (isPlainObject(row.result) && isPlainObject(row.result.answers)) {
          for (const key of Object.keys(row.result.answers)) {
            if (!seen.has(key)) {
              seen.add(key);
              out.push(key);
            }
          }
        }
      }
    }
    return out;
  }, [activeSpecs, rows]);

  const columnColors = useMemo(() => {
    const map = new Map<string, string>();
    activeSpecs.forEach((spec) => {
      if (spec.name !== '' && !map.has(spec.name)) map.set(spec.name, spec.color);
    });
    columns.forEach((col, idx) => {
      if (!map.has(col)) map.set(col, fieldAccentColor(idx));
    });
    return map;
  }, [activeSpecs, columns]);

  const effectivePrimaryColumn =
    primaryColumn !== '' && columns.includes(primaryColumn) ? primaryColumn : (columns[0] ?? '');

  const requestFieldsMap = useMemo<Record<string, unknown>>(() => {
    if (mode !== 'extract') return {};
    const payload = buildClairicPayload({
      stateText: '',
      referenceDate: '',
      includeQuantiles: false,
      returnStateText: false,
      fields,
      attachments: [],
    });
    return isPlainObject(payload.fields) ? payload.fields : {};
  }, [mode, fields]);

  const rowMetaMap = useMemo(() => {
    const map = new Map<string, ComputedRowMeta>();
    rows.forEach((row, index) => {
      const answers = isPlainObject(row.result) && isPlainObject(row.result.answers) ? row.result.answers : null;
      const summary = summarizeRouting(row.result, policy);
      const isDone = row.status === 'done';
      const needsReview = isDone && summary.reviewCount > 0;
      const autoRouted = isDone && summary.totalCount > 0 && summary.reviewCount === 0;
      const expectedMatch =
        isDone && answers !== null && effectivePrimaryColumn !== '' && row.expected.trim() !== ''
          ? Object.hasOwn(answers, effectivePrimaryColumn)
            ? compareExpectedAnswer(answers[effectivePrimaryColumn], row.expected, policy)
            : false
          : null;
      map.set(row.id, {
        index,
        answers,
        reviewCount: summary.reviewCount,
        autoCount: summary.autoCount,
        missingCount: summary.missingCount,
        totalAnswers: summary.totalCount,
        autoRouted,
        needsReview,
        expectedMatch,
      });
    });
    return map;
  }, [rows, policy, effectivePrimaryColumn]);

  const kpis = useMemo(() => {
    let doneCount = 0;
    let errorCount = 0;
    let autoRoutedRows = 0;
    let needsReviewRows = 0;
    let latencySum = 0;
    let latencyCount = 0;
    let evaluatedExpected = 0;
    let matchedExpected = 0;

    for (const row of rows) {
      const meta = rowMetaMap.get(row.id);
      if (row.status === 'done') doneCount += 1;
      if (row.status === 'error') errorCount += 1;
      if (row.elapsedMs !== null && Number.isFinite(row.elapsedMs)) {
        latencySum += row.elapsedMs;
        latencyCount += 1;
      }
      if (meta !== undefined) {
        if (meta.autoRouted) autoRoutedRows += 1;
        if (meta.needsReview) needsReviewRows += 1;
        if (meta.expectedMatch !== null) {
          evaluatedExpected += 1;
          if (meta.expectedMatch) matchedExpected += 1;
        }
      }
    }

    return {
      doneCount,
      errorCount,
      autoRoutedRows,
      needsReviewRows,
      avgMs: latencyCount > 0 ? latencySum / latencyCount : null,
      autoRoutedRatio: doneCount > 0 ? autoRoutedRows / doneCount : null,
      evaluatedExpected,
      matchedExpected,
      agreementRatio: evaluatedExpected > 0 ? matchedExpected / evaluatedExpected : null,
    };
  }, [rows, rowMetaMap]);

  const filteredRows = useMemo(() => {
    if (filter === 'all') return rows;
    if (filter === 'errors') return rows.filter((row) => row.status === 'error');
    return rows.filter((row) => rowMetaMap.get(row.id)?.needsReview === true);
  }, [rows, filter, rowMetaMap]);

  const samplesForMode = useMemo(() => SYSTEM_ONE_BATCH_SAMPLES.filter((s) => s.mode === mode), [mode]);
  const samplesOtherMode = useMemo(() => SYSTEM_ONE_BATCH_SAMPLES.filter((s) => s.mode !== mode), [mode]);

  const updateRow = useCallback(
    (rowId: string, patch: Partial<BatchRow>): void => {
      const invalidatesOutput = Object.hasOwn(patch, 'stateText') || Object.hasOwn(patch, 'attachments');
      onRowsChange((prev) =>
        prev.map((row) => {
          if (row.id !== rowId) return row;
          if (invalidatesOutput) {
            return {
              ...row,
              ...patch,
              status: 'idle',
              elapsedMs: null,
              result: undefined,
              error: undefined,
            };
          }
          return { ...row, ...patch };
        }),
      );
    },
    [onRowsChange],
  );

  const handleAddCase = (): void => {
    const created = createBatchRow({ label: `Case ${String(rows.length + 1)}` });
    onRowsChange((prev) => [...prev, created]);
    setInspectRowId(created.id);
  };

  const handleDeleteRow = (rowId: string): void => {
    if (inspectRowId === rowId) setInspectRowId(null);
    onRowsChange((prev) => prev.filter((row) => row.id !== rowId));
  };

  const handleClearRows = (): void => {
    datasetEpochRef.current += 1;
    setInspectRowId(null);
    setFilter('all');
    onRowsChange([]);
  };

  const appendParsedEntries = (
    entries: ReadonlyArray<{ label: string; stateText: string; expected: string }>,
    expectedEpoch?: number,
  ): void => {
    if (expectedEpoch !== undefined && datasetEpochRef.current !== expectedEpoch) return;
    if (entries.length === 0) {
      setImportError('No cases found. Paste CSV, JSONL, JSON array, or one line per case.');
      return;
    }
    const created = entries.map((entry) =>
      createBatchRow({
        label: entry.label,
        stateText: entry.stateText,
        expected: entry.expected,
      }),
    );
    onRowsChange((prev) => [...prev, ...created]);
    setImportText('');
    setImportError(null);
    setImportOpen(false);
    toast.success(`Imported ${String(created.length)} ${created.length === 1 ? 'case' : 'cases'}`);
  };

  const handleImportPasted = (): void => {
    setImportError(null);
    const parsed = parseBatchImportText(importText);
    appendParsedEntries(parsed);
  };

  const handleTextFileUpload = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file === undefined) return;
    const epoch = datasetEpochRef.current;
    setImportError(null);
    try {
      const content = await file.text();
      if (datasetEpochRef.current !== epoch) return;
      const parsed = parseBatchImportText(content);
      appendParsedEntries(parsed, epoch);
    } catch {
      if (datasetEpochRef.current === epoch) {
        setImportError(`Could not read "${file.name}".`);
      }
    }
  };

  const handleMediaFilesUpload = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const files = event.target.files;
    if (files === null || files.length === 0) return;
    const incoming = Array.from(files);
    event.target.value = '';
    const epoch = datasetEpochRef.current;
    setImportError(null);

    const createdRows: BatchRow[] = [];
    const errors: string[] = [];

    for (const file of incoming) {
      const mime = normalizeMediaMime(file);
      if (!ACCEPTED_MEDIA_MIME.has(mime)) {
        errors.push(`"${file.name}" is not a supported PNG, JPEG, WebP, or PDF file.`);
        continue;
      }
      if (file.size > CLAIRIC_MAX_MEDIA_BYTES) {
        errors.push(`"${file.name}" (${formatBytes(file.size)}) exceeds the 4 MiB per-file limit.`);
        continue;
      }
      try {
        const dataUrl = await readFileAsDataUrl(file);
        if (datasetEpochRef.current !== epoch) return;
        const comma = dataUrl.indexOf(',');
        const base64 = comma === -1 ? '' : dataUrl.slice(comma + 1);
        const kind = mime === 'application/pdf' ? 'pdf' : 'image';
        const attachment: MediaAttachment = {
          id: nextDraftId(kind),
          name: file.name,
          contentType: mime === 'image/jpg' ? 'image/jpeg' : mime,
          sizeBytes: file.size,
          base64,
          dataUrl,
          rotation: 0,
          kind,
        };
        createdRows.push(
          createBatchRow({
            label: stripExtension(file.name),
            stateText: '',
            attachments: [attachment],
          }),
        );
      } catch {
        errors.push(`Could not read "${file.name}".`);
      }
    }

    if (datasetEpochRef.current !== epoch) return;
    if (createdRows.length > 0) {
      onRowsChange((prev) => [...prev, ...createdRows]);
      toast.success(`Added ${String(createdRows.length)} media ${createdRows.length === 1 ? 'case' : 'cases'}`);
      if (errors.length === 0) setImportOpen(false);
    }
    if (errors.length > 0) {
      setImportError(errors.join(' '));
    }
  };

  const handleExportCsv = (): void => {
    if (rows.length === 0) return;
    const csv = exportBatchToCsv({ rows, columns, mode, policy });
    triggerFileDownload(csv, `system-one-batch-${mode}.csv`, 'text/csv;charset=utf-8');
    toast.success('Exported batch CSV');
  };

  const handleExportJson = (): void => {
    if (rows.length === 0) return;
    const payload = {
      mode,
      model: modelName,
      policy,
      primaryColumn: effectivePrimaryColumn || null,
      exportedAt: new Date().toISOString(),
      rows: rows.map((row, index) => {
        const answers = isPlainObject(row.result) && isPlainObject(row.result.answers) ? row.result.answers : {};
        const routes: Record<string, AnswerRouteResult> = {};
        for (const col of columns) {
          if (Object.hasOwn(answers, col)) {
            routes[col] = routeCellAnswer(answers[col], mode, policy);
          }
        }
        return {
          index: index + 1,
          label: row.label.trim() || `Case ${String(index + 1)}`,
          stateText: row.stateText,
          attachments: row.attachments.map((a) => ({
            name: a.name,
            contentType: a.contentType,
            sizeBytes: a.sizeBytes,
            kind: a.kind,
          })),
          expected: row.expected,
          status: row.status,
          elapsedMs: row.elapsedMs,
          routes,
          result: row.result ?? null,
          error: row.error ?? null,
        };
      }),
    };
    triggerFileDownload(JSON.stringify(payload, null, 2), `system-one-batch-${mode}.json`, 'application/json');
    toast.success('Exported batch JSON');
  };

  const handleNoThresholdChange = (pct: number): void => {
    const nextNo = Math.round(pct) / 100;
    const nextYes = Math.max(policy.yesThreshold, nextNo);
    onPolicyChange(normalizePolicy({ ...policy, noThreshold: nextNo, yesThreshold: nextYes }));
  };

  const handleYesThresholdChange = (pct: number): void => {
    const nextYes = Math.round(pct) / 100;
    const nextNo = Math.min(policy.noThreshold, nextYes);
    onPolicyChange(normalizePolicy({ ...policy, noThreshold: nextNo, yesThreshold: nextYes }));
  };

  const handleMinConfidenceChange = (pct: number): void => {
    const nextConf = Math.round(pct) / 100;
    onPolicyChange(normalizePolicy({ ...policy, minConfidence: nextConf }));
  };

  const inspectedRow = useMemo(
    () => (inspectRowId === null ? null : (rows.find((r) => r.id === inspectRowId) ?? null)),
    [inspectRowId, rows],
  );
  const drawerRow = useDrawerItem(inspectedRow);
  const drawerRowIndex = drawerRow != null ? rows.findIndex((r) => r.id === drawerRow.id) : -1;
  const drawerRowMeta = drawerRow != null ? rowMetaMap.get(drawerRow.id) : undefined;

  const primaryColumnOptions = useMemo(() => columns.map((col) => ({ value: col, label: col })), [columns]);

  const policyModified = !isPolicyDefault(policy, mode);
  const reviewZoneWidth = Math.max(0, Math.round((policy.yesThreshold - policy.noThreshold) * 100));

  return (
    <div className={styles.root}>
      {rows.length > 0 ? (
        <StatGroup label="Batch evaluation metrics" maxColumns={4}>
          <StatTile
            label="Progress"
            value={`${String(kpis.doneCount)} / ${String(rows.length)}`}
            countUp={false}
            hint={
              kpis.avgMs !== null ? `${formatDuration(kpis.avgMs)} avg` : running ? 'Running cases…' : 'Ready to run'
            }
          />
          <StatTile
            label="Auto-routed"
            value={kpis.doneCount > 0 ? formatPercent(kpis.autoRoutedRatio) : '—'}
            countUp={false}
            hint={
              kpis.doneCount > 0
                ? `${String(kpis.autoRoutedRows)} of ${String(kpis.doneCount)} within policy`
                : 'Updates live with thresholds'
            }
            footer={
              // Before the first result the bar is an empty track of the batch size: an unknown value adds a caption
              // line, and the tile would change height when the first case ends.
              <Meter
                label="Auto-routed cases"
                value={kpis.doneCount > 0 ? kpis.autoRoutedRows : 0}
                max={kpis.doneCount > 0 ? kpis.doneCount : rows.length}
                warnAt={1.1}
                dangerAt={1.1}
                size="dense"
              />
            }
          />
          <StatTile
            label="Needs review"
            value={String(kpis.needsReviewRows)}
            countUp={false}
            hint={
              kpis.errorCount > 0
                ? `${String(kpis.errorCount)} ${kpis.errorCount === 1 ? 'error' : 'errors'}`
                : kpis.doneCount > 0
                  ? kpis.needsReviewRows === 0
                    ? 'Zero flagged cases'
                    : 'Flagged by live policy'
                  : 'Waiting for results'
            }
          />
          {kpis.evaluatedExpected > 0 ? (
            <StatTile
              label="Agreement"
              value={formatPercent(kpis.agreementRatio)}
              countUp={false}
              hint={`${String(kpis.matchedExpected)} / ${String(kpis.evaluatedExpected)} on ${effectivePrimaryColumn}`}
            />
          ) : (
            <StatTile
              label="Columns"
              value={String(columns.length)}
              countUp={false}
              hint={mode === 'decide' ? 'Active rubric questions' : 'Active extraction fields'}
            />
          )}
        </StatGroup>
      ) : null}

      <Panel ariaLabel="Batch specification and live routing policy" padding="none" className={styles.specPanel}>
        <div className={styles.specRow}>
          <div className={styles.specLeft}>
            <div className={styles.specTitleRow}>
              <span className={styles.specLabel}>{mode === 'decide' ? 'Decision rubric' : 'Extraction schema'}</span>
              {modelName !== null && modelName !== '' ? (
                <Badge size="sm" outline className={styles.modelTag}>
                  {modelName}
                </Badge>
              ) : null}
            </div>
            <div className={styles.specChips}>
              {activeSpecs.length === 0 ? (
                <span className={styles.specEmpty}>
                  {mode === 'decide'
                    ? 'No questions configured yet — click Edit questions or pick a sample batch.'
                    : 'No fields configured yet — click Edit fields or pick a sample batch.'}
                </span>
              ) : (
                activeSpecs.map((spec) => {
                  const hasWarn = spec.warnings.length > 0;
                  const chip = (
                    <span key={spec.id} className={styles.specChip} data-invalid={hasWarn ? 'true' : undefined}>
                      <span className={styles.specDot} style={{ background: spec.color }} aria-hidden="true" />
                      <span className={styles.specChipName}>{spec.name || 'untitled'}</span>
                      {hasWarn ? (
                        <AlertTriangle
                          size={11}
                          strokeWidth={2}
                          className={styles.specWarnIcon}
                          aria-label={spec.warnings[0]}
                        />
                      ) : null}
                    </span>
                  );
                  return hasWarn ? (
                    <Tooltip key={spec.id} content={spec.warnings[0]}>
                      {chip}
                    </Tooltip>
                  ) : (
                    chip
                  );
                })
              )}
            </div>
          </div>

          <div className={styles.specActions}>
            {onSaveSpec !== undefined ? (
              <Button type="button" variant="ghost" size="sm" onClick={onSaveSpec}>
                <BookmarkPlus size={13} strokeWidth={1.75} aria-hidden="true" />
                Save spec
              </Button>
            ) : null}
            <Button type="button" variant="secondary" size="sm" onClick={() => setSpecDrawerOpen(true)}>
              <SlidersHorizontal size={13} strokeWidth={1.75} aria-hidden="true" />
              {mode === 'decide' ? 'Edit questions' : 'Edit fields'}
            </Button>
            {running ? (
              <Button type="button" variant="danger" size="sm" onClick={onCancel}>
                <Square size={12} strokeWidth={2} aria-hidden="true" />
                Stop batch
              </Button>
            ) : (
              <Tooltip
                content={
                  !canRun ? runTooltip : rows.length === 0 ? 'Add or load cases to run' : 'Run every case sequentially'
                }
              >
                <span className={styles.tooltipWrap}>
                  <Button
                    type="button"
                    variant="primary"
                    size="sm"
                    disabled={!canRun || rows.length === 0}
                    onClick={onRunAll}
                  >
                    <Play size={12} strokeWidth={2} aria-hidden="true" />
                    {rows.length > 0 ? `Run all (${String(rows.length)})` : 'Run all'}
                  </Button>
                </span>
              </Tooltip>
            )}
          </div>
        </div>

        <div className={styles.policyRow}>
          <span className={styles.policyHeading}>Live thresholds</span>

          {mode === 'decide' ? (
            <div className={styles.sliderGroup}>
              <label className={styles.sliderControl}>
                <span className={styles.sliderLabel}>Auto-No ≤</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(policy.noThreshold * 100)}
                  onChange={(e) => handleNoThresholdChange(Number(e.target.value))}
                  className={styles.rangeInput}
                  aria-label="Auto-No threshold percentage"
                />
                <span className={styles.sliderValue}>{`${String(Math.round(policy.noThreshold * 100))}%`}</span>
              </label>

              <label className={styles.sliderControl}>
                <span className={styles.sliderLabel}>Auto-Yes ≥</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(policy.yesThreshold * 100)}
                  onChange={(e) => handleYesThresholdChange(Number(e.target.value))}
                  className={styles.rangeInput}
                  aria-label="Auto-Yes threshold percentage"
                />
                <span className={styles.sliderValue}>{`${String(Math.round(policy.yesThreshold * 100))}%`}</span>
              </label>

              <label className={styles.sliderControl}>
                <span className={styles.sliderLabel}>Confidence ≥</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(policy.minConfidence * 100)}
                  onChange={(e) => handleMinConfidenceChange(Number(e.target.value))}
                  className={styles.rangeInput}
                  aria-label="Minimum confidence threshold percentage"
                />
                <span className={styles.sliderValue}>{`${String(Math.round(policy.minConfidence * 100))}%`}</span>
              </label>

              <span className={styles.policyNote}>
                {reviewZoneWidth > 0
                  ? `Review band ${String(Math.round(policy.noThreshold * 100))}%–${String(Math.round(policy.yesThreshold * 100))}%`
                  : 'No review band'}
              </span>
            </div>
          ) : (
            <div className={styles.sliderGroup}>
              <label className={styles.sliderControl}>
                <span className={styles.sliderLabel}>Auto-accept ≥</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(policy.minConfidence * 100)}
                  onChange={(e) => handleMinConfidenceChange(Number(e.target.value))}
                  className={styles.rangeInput}
                  aria-label="Auto-accept confidence threshold percentage"
                />
                <span className={styles.sliderValue}>{`${String(Math.round(policy.minConfidence * 100))}%`}</span>
              </label>
              <span className={styles.policyNote}>Dragging updates every row verdict and KPI metric live.</span>
            </div>
          )}

          {policyModified ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className={styles.resetPolicyBtn}
              onClick={() => onPolicyChange({ ...DEFAULT_DECISION_POLICY })}
            >
              <RotateCcw size={12} strokeWidth={1.75} aria-hidden="true" />
              Reset
            </Button>
          ) : null}
        </div>
      </Panel>

      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <div className={styles.filterPills} role="group" aria-label="Filter batch cases">
            <button
              type="button"
              className={styles.filterPill}
              aria-pressed={filter === 'all'}
              onClick={() => setFilter('all')}
            >
              <span>All</span>
              <span className={styles.filterCount}>{rows.length}</span>
            </button>
            <button
              type="button"
              className={styles.filterPill}
              data-tone={kpis.needsReviewRows > 0 ? 'warning' : undefined}
              aria-pressed={filter === 'review'}
              onClick={() => setFilter('review')}
            >
              <span>Needs review</span>
              <span className={styles.filterCount}>{kpis.needsReviewRows}</span>
            </button>
            <button
              type="button"
              className={styles.filterPill}
              data-tone={kpis.errorCount > 0 ? 'danger' : undefined}
              aria-pressed={filter === 'errors'}
              onClick={() => setFilter('errors')}
            >
              <span>Errors</span>
              <span className={styles.filterCount}>{kpis.errorCount}</span>
            </button>
          </div>

          {columns.length > 1 ? (
            <div className={styles.primaryColWrap}>
              <span className={styles.primaryColLabel}>Primary field</span>
              <CompactField className={styles.primaryColSelect}>
                <Select
                  label="Primary field"
                  options={primaryColumnOptions}
                  value={effectivePrimaryColumn}
                  onValueChange={onPrimaryColumnChange}
                />
              </CompactField>
            </div>
          ) : null}
        </div>

        <div className={styles.toolbarRight}>
          <Popover open={samplesOpen} onOpenChange={setSamplesOpen}>
            <PopoverTrigger asChild>
              <Button type="button" variant="secondary" size="sm" disabled={running}>
                Sample batch
                <ChevronDown size={13} strokeWidth={1.75} aria-hidden="true" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className={styles.samplePopover}>
              <div className={styles.popoverSectionLabel}>
                {mode === 'decide' ? 'Decide samples' : 'Extract samples'}
              </div>
              {samplesForMode.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  className={styles.sampleItem}
                  onClick={() => {
                    datasetEpochRef.current += 1;
                    setSamplesOpen(false);
                    setFilter('all');
                    onLoadSample(sample);
                  }}
                >
                  <div className={styles.sampleItemHead}>
                    <span className={styles.sampleItemTitle}>{sample.label}</span>
                    <Badge size="sm" outline>
                      {`${String(sample.rows.length)} cases`}
                    </Badge>
                  </div>
                  <p className={styles.sampleItemDesc}>{sample.description}</p>
                </button>
              ))}
              {samplesOtherMode.length > 0 ? (
                <>
                  <div className={styles.popoverDivider} />
                  <div className={styles.popoverSectionLabel}>
                    {mode === 'decide' ? 'Extract samples' : 'Decide samples'}
                  </div>
                  {samplesOtherMode.map((sample) => (
                    <button
                      key={sample.id}
                      type="button"
                      className={styles.sampleItem}
                      onClick={() => {
                        datasetEpochRef.current += 1;
                        setSamplesOpen(false);
                        setFilter('all');
                        onLoadSample(sample);
                      }}
                    >
                      <div className={styles.sampleItemHead}>
                        <span className={styles.sampleItemTitle}>{sample.label}</span>
                        <Badge size="sm" color={sample.mode === 'decide' ? 'orchid' : 'sky'} outline>
                          {sample.mode === 'decide' ? 'Decide' : 'Extract'}
                        </Badge>
                      </div>
                      <p className={styles.sampleItemDesc}>{sample.description}</p>
                    </button>
                  ))}
                </>
              ) : null}
            </PopoverContent>
          </Popover>

          <Button
            type="button"
            variant={importOpen ? 'secondary' : 'ghost'}
            size="sm"
            disabled={running}
            aria-expanded={importOpen}
            onClick={() => {
              setImportError(null);
              setImportOpen((prev) => !prev);
            }}
          >
            <Upload size={13} strokeWidth={1.75} aria-hidden="true" />
            Import
          </Button>

          <Button type="button" variant="secondary" size="sm" disabled={running} onClick={handleAddCase}>
            <Plus size={13} strokeWidth={1.75} aria-hidden="true" />
            Add case
          </Button>

          <Button type="button" variant="ghost" size="sm" disabled={rows.length === 0} onClick={handleExportCsv}>
            <Download size={13} strokeWidth={1.75} aria-hidden="true" />
            Export CSV
          </Button>

          <Button type="button" variant="ghost" size="sm" disabled={rows.length === 0} onClick={handleExportJson}>
            <Download size={13} strokeWidth={1.75} aria-hidden="true" />
            Export JSON
          </Button>

          {rows.length > 0 ? (
            <Button type="button" variant="ghost" size="sm" disabled={running} onClick={handleClearRows}>
              Clear
            </Button>
          ) : null}
        </div>
      </div>

      {importOpen ? (
        <Panel ariaLabel="Import batch cases" className={styles.importPanel}>
          <div className={styles.importHeader}>
            <div className={styles.importHeading}>
              <span className={styles.specLabel}>Import cases</span>
              <span className={styles.importHint}>
                Paste CSV (with headers like label, input, expected), JSONL, JSON array, or one case per line — or pick
                files.
              </span>
            </div>
            <IconButton
              label="Close import panel"
              icon={<X size={14} strokeWidth={1.75} aria-hidden="true" />}
              onClick={() => setImportOpen(false)}
            />
          </div>

          <textarea
            className={styles.importTextarea}
            rows={4}
            value={importText}
            placeholder={
              'label,input,expected\n"Ticket #1","Production API returning 502 since 09:00","critical"\n"Ticket #2","Requesting dark mode toggle","low"'
            }
            onChange={(e) => setImportText(e.target.value)}
            aria-label="Paste CSV, JSONL, JSON, or plain lines to import"
          />

          <input
            ref={textFileInputRef}
            type="file"
            className={styles.hiddenInput}
            accept=".csv,.jsonl,.json,.txt,text/csv,application/json,text/plain"
            onChange={(e) => void handleTextFileUpload(e)}
            aria-label="Upload CSV, JSONL, JSON, or text file"
          />
          {mode === 'extract' ? (
            <input
              ref={mediaFileInputRef}
              type="file"
              className={styles.hiddenInput}
              accept="image/png,image/jpeg,image/webp,application/pdf"
              multiple
              onChange={(e) => void handleMediaFilesUpload(e)}
              aria-label="Upload images or PDFs as batch cases"
            />
          ) : null}

          {importError !== null ? (
            <Alert tone="warning" title="Import notice">
              {importError}
            </Alert>
          ) : null}

          <div className={styles.importFooter}>
            <div className={styles.importFileActions}>
              <Button type="button" variant="secondary" size="sm" onClick={() => textFileInputRef.current?.click()}>
                <FileUp size={13} strokeWidth={1.75} aria-hidden="true" />
                Pick CSV / JSONL / JSON
              </Button>
              {mode === 'extract' ? (
                <Button type="button" variant="secondary" size="sm" onClick={() => mediaFileInputRef.current?.click()}>
                  <ImagePlus size={13} strokeWidth={1.75} aria-hidden="true" />
                  Pick images / PDFs (1 row each)
                </Button>
              ) : null}
            </div>
            <Button
              type="button"
              variant="primary"
              size="sm"
              disabled={importText.trim() === ''}
              onClick={handleImportPasted}
            >
              Add pasted cases
            </Button>
          </div>
        </Panel>
      ) : null}

      {rows.length === 0 ? (
        <Panel ariaLabel="Empty batch workspace" className={styles.emptyPanel}>
          <EmptyState
            icon={<Layers size={22} strokeWidth={1.5} aria-hidden="true" />}
            title="Evaluate a batch of cases"
            description={
              mode === 'decide'
                ? 'Score multiple cases against your decision rubric, tune Yes/No thresholds live, and measure agreement against expected verdicts.'
                : 'Extract structured fields across documents, tickets, or receipt images, flag low-confidence rows, and export clean CSV/JSON.'
            }
            action={
              <div className={styles.emptyActions}>
                <Button type="button" variant="primary" size="sm" onClick={handleAddCase}>
                  <Plus size={13} strokeWidth={1.75} aria-hidden="true" />
                  Add case
                </Button>
                <Button type="button" variant="secondary" size="sm" onClick={() => setImportOpen(true)}>
                  <Upload size={13} strokeWidth={1.75} aria-hidden="true" />
                  Import CSV / JSONL
                </Button>
              </div>
            }
          />

          {samplesForMode.length > 0 ? (
            <div className={styles.sampleSection}>
              <span className={styles.sampleSectionTitle}>Or start with a sample batch</span>
              <div className={styles.sampleGrid}>
                {samplesForMode.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    className={styles.sampleCard}
                    onClick={() => {
                      datasetEpochRef.current += 1;
                      onLoadSample(sample);
                    }}
                  >
                    <div className={styles.sampleCardBody}>
                      <div className={styles.sampleCardHead}>
                        <span className={styles.sampleCardTitle}>{sample.label}</span>
                        <Badge size="sm" outline>
                          {`${String(sample.rows.length)} cases`}
                        </Badge>
                      </div>
                      <p className={styles.sampleCardDesc}>{sample.description}</p>
                    </div>
                    <ArrowRight size={14} strokeWidth={1.75} className={styles.sampleCardArrow} aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </Panel>
      ) : filteredRows.length === 0 ? (
        <Panel ariaLabel="No matching batch rows" className={styles.emptyFilterPanel}>
          <EmptyState
            title={filter === 'errors' ? 'No errors in this batch' : 'No cases need review'}
            description={
              filter === 'errors'
                ? 'Every executed case completed without an error.'
                : 'All completed cases satisfy your current live threshold policy.'
            }
            action={
              <Button type="button" variant="secondary" size="sm" onClick={() => setFilter('all')}>
                Show all cases
              </Button>
            }
          />
        </Panel>
      ) : (
        <div className={styles.tableCard}>
          <div className={styles.tableScroll}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.colIndex}>#</th>
                  <th className={styles.colCase}>Case / Input</th>
                  {columns.map((col) => {
                    const dotColor = columnColors.get(col) ?? fieldAccentColor(0);
                    const isPrimary = columns.length > 1 && col === effectivePrimaryColumn;
                    return (
                      <th key={col} className={styles.colAnswer}>
                        <span className={styles.colHeadInner}>
                          <span className={styles.colHeadDot} style={{ background: dotColor }} aria-hidden="true" />
                          <span className={styles.colHeadName}>{col}</span>
                          {isPrimary ? (
                            <Tooltip content="Expected column compares against this field">
                              <span className={styles.primaryTag}>primary</span>
                            </Tooltip>
                          ) : null}
                        </span>
                      </th>
                    );
                  })}
                  <th className={styles.colExpected}>Expected</th>
                  <th className={styles.colActions}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => {
                  const meta = rowMetaMap.get(row.id);
                  const rowNum = (meta?.index ?? 0) + 1;
                  const isRowRunning = row.status === 'running' || row.id === runningRowId;
                  const firstMedia = row.attachments[0];
                  const answers = meta?.answers ?? null;
                  const statusKind = isRowRunning
                    ? 'running'
                    : row.status === 'error'
                      ? 'error'
                      : row.status === 'done'
                        ? meta?.needsReview === true
                          ? 'review'
                          : 'done'
                        : row.status === 'queued'
                          ? 'queued'
                          : 'idle';

                  return (
                    <tr
                      key={row.id}
                      className={styles.row}
                      data-status={statusKind}
                      data-selected={inspectRowId === row.id ? 'true' : undefined}
                    >
                      <td className={styles.cellIndex}>
                        <div className={styles.indexWrap}>
                          <span className={styles.rowNumber}>{rowNum}</span>
                          {isRowRunning ? (
                            <span className={styles.orbWrap}>
                              <WaitOrb state="think" label={`Case ${String(rowNum)} running`} />
                            </span>
                          ) : (
                            <Tooltip
                              content={
                                statusKind === 'done'
                                  ? 'Completed — auto-routed'
                                  : statusKind === 'review'
                                    ? 'Completed — needs review'
                                    : statusKind === 'error'
                                      ? 'Error'
                                      : statusKind === 'queued'
                                        ? 'Queued'
                                        : 'Idle'
                              }
                            >
                              <span className={styles.statusDot} data-status={statusKind} aria-label={statusKind} />
                            </Tooltip>
                          )}
                        </div>
                      </td>

                      <td className={styles.cellCase}>
                        <div className={styles.caseCell}>
                          <div className={styles.caseHead}>
                            <input
                              type="text"
                              className={styles.inlineLabelInput}
                              value={row.label}
                              placeholder={`Case ${String(rowNum)}`}
                              aria-label={`Case ${String(rowNum)} label`}
                              disabled={running}
                              onChange={(e) => updateRow(row.id, { label: e.target.value })}
                            />
                            {row.elapsedMs !== null ? (
                              <span className={styles.rowLatency}>{formatDuration(row.elapsedMs)}</span>
                            ) : null}
                          </div>

                          <button
                            type="button"
                            className={styles.inputPreviewBtn}
                            onClick={() => setInspectRowId(row.id)}
                          >
                            {firstMedia !== undefined ? (
                              <span className={styles.thumbSlot}>
                                {firstMedia.kind === 'image' ? (
                                  <img src={firstMedia.dataUrl} alt={firstMedia.name} className={styles.miniThumb} />
                                ) : (
                                  <span className={styles.miniPdf}>
                                    <FileText size={13} strokeWidth={1.75} aria-hidden="true" />
                                  </span>
                                )}
                              </span>
                            ) : null}
                            <span className={styles.inputPreviewText}>
                              {row.stateText.trim() !== ''
                                ? compactPreview(row.stateText)
                                : firstMedia !== undefined
                                  ? `${firstMedia.name}${row.attachments.length > 1 ? ` +${String(row.attachments.length - 1)}` : ''}`
                                  : 'Empty input — click to edit'}
                            </span>
                          </button>

                          {row.status === 'error' && row.error !== undefined && row.error !== '' ? (
                            <span className={styles.rowErrorText} title={row.error}>
                              {row.error}
                            </span>
                          ) : null}
                        </div>
                      </td>

                      {columns.map((col) => {
                        const hasAns = answers !== null && Object.hasOwn(answers, col);
                        const rawAns = hasAns ? answers[col] : undefined;
                        if (isRowRunning && !hasAns) {
                          return (
                            <td key={col} className={styles.cellAnswer}>
                              <span className={styles.cellPending}>Running…</span>
                            </td>
                          );
                        }
                        if (!hasAns) {
                          return (
                            <td key={col} className={styles.cellAnswer}>
                              <span className={styles.cellEmpty}>—</span>
                            </td>
                          );
                        }
                        const routed = routeCellAnswer(rawAns, mode, policy);
                        const summary = formatAnswerSummary(rawAns);
                        const confVal = routed.isNull && routed.pNull !== null ? routed.pNull : routed.confidence;
                        const tone: BadgeTone = routed.tone;

                        return (
                          <td key={col} className={styles.cellAnswer}>
                            <div className={styles.answerCell}>
                              <span className={styles.answerVal} title={summary}>
                                {summary}
                              </span>
                              <div className={styles.answerMeta}>
                                <Tooltip content={routed.label}>
                                  <span>
                                    <Badge size="sm" tone={tone} pill>
                                      {shortRouteLabel(routed.route)}
                                    </Badge>
                                  </span>
                                </Tooltip>
                                {confVal !== null ? (
                                  <span className={styles.confText}>{formatPercent(confVal)}</span>
                                ) : null}
                              </div>
                            </div>
                          </td>
                        );
                      })}

                      <td className={styles.cellExpected}>
                        <div className={styles.expectedWrap}>
                          <input
                            type="text"
                            className={styles.expectedInput}
                            value={row.expected}
                            placeholder="optional"
                            aria-label={`Case ${String(rowNum)} expected value`}
                            onChange={(e) => updateRow(row.id, { expected: e.target.value })}
                          />
                          {meta?.expectedMatch === true ? (
                            <Tooltip content={`Matches expected on ${effectivePrimaryColumn}`}>
                              <span className={styles.matchOk} aria-label="Matches expected">
                                <Check size={14} strokeWidth={2.25} aria-hidden="true" />
                              </span>
                            </Tooltip>
                          ) : meta?.expectedMatch === false ? (
                            <Tooltip content={`Differs from expected on ${effectivePrimaryColumn}`}>
                              <span className={styles.matchFail} aria-label="Does not match expected">
                                <X size={14} strokeWidth={2.25} aria-hidden="true" />
                              </span>
                            </Tooltip>
                          ) : null}
                        </div>
                      </td>

                      <td className={styles.cellActions}>
                        <div className={styles.actionGroup}>
                          <IconButton
                            label={`Run case ${String(rowNum)}`}
                            tooltip={!canRun ? runTooltip : 'Run this case'}
                            icon={<Play size={13} strokeWidth={1.75} aria-hidden="true" />}
                            disabled={running || !canRun}
                            onClick={() => onRunRow(row.id)}
                          />
                          <IconButton
                            label={`Inspect case ${String(rowNum)}`}
                            tooltip="Inspect & edit"
                            icon={<Eye size={13} strokeWidth={1.75} aria-hidden="true" />}
                            onClick={() => setInspectRowId(row.id)}
                          />
                          <IconButton
                            label={`Delete case ${String(rowNum)}`}
                            tooltip="Delete case"
                            icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                            disabled={running}
                            onClick={() => handleDeleteRow(row.id)}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <Drawer open={specDrawerOpen} onOpenChange={setSpecDrawerOpen}>
        <DrawerContent
          side="right"
          title={mode === 'decide' ? 'Edit decision rubric' : 'Edit extraction schema'}
          description="Refine fields or questions without leaving Batch mode."
        >
          <div className={styles.drawerStack}>
            {mode === 'decide' ? (
              <ClefBuilder
                questions={clefQuestions}
                warnings={clefWarnings}
                onChange={onClefQuestionsChange}
                disabled={running}
              />
            ) : (
              <SchemaBuilder fields={fields} warnings={fieldWarnings} onChange={onFieldsChange} disabled={running} />
            )}
            <div className={styles.drawerFooter}>
              {onSaveSpec !== undefined ? (
                <Button type="button" variant="secondary" size="sm" onClick={onSaveSpec}>
                  <BookmarkPlus size={13} strokeWidth={1.75} aria-hidden="true" />
                  Save spec
                </Button>
              ) : (
                <span />
              )}
              <Button type="button" variant="primary" size="sm" onClick={() => setSpecDrawerOpen(false)}>
                Done
              </Button>
            </div>
          </div>
        </DrawerContent>
      </Drawer>

      <Drawer
        open={inspectRowId !== null}
        onOpenChange={(open) => {
          if (!open) setInspectRowId(null);
        }}
      >
        <DrawerContent
          side="right"
          title={
            drawerRow != null
              ? drawerRow.label.trim() || `Case ${String(Math.max(1, drawerRowIndex + 1))}`
              : 'Case inspector'
          }
          description={
            drawerRow != null && drawerRow.elapsedMs !== null
              ? `Case ${String(Math.max(1, drawerRowIndex + 1))} of ${String(rows.length)} · ${formatDuration(drawerRow.elapsedMs)}`
              : `Case ${String(Math.max(1, drawerRowIndex + 1))} of ${String(rows.length)}`
          }
        >
          {drawerRow != null ? (
            <div className={styles.drawerStack}>
              <div className={styles.drawerTwoCol}>
                <Input
                  label="Case label"
                  placeholder={`Case ${String(Math.max(1, drawerRowIndex + 1))}`}
                  value={drawerRow.label}
                  disabled={running}
                  onChange={(e) => updateRow(drawerRow.id, { label: e.target.value })}
                />
                <Input
                  label={effectivePrimaryColumn !== '' ? `Expected (${effectivePrimaryColumn})` : 'Expected answer'}
                  placeholder="Optional expected value…"
                  value={drawerRow.expected}
                  onChange={(e) => updateRow(drawerRow.id, { expected: e.target.value })}
                />
              </div>

              <div className={styles.drawerFieldGroup}>
                <span className={styles.drawerSectionLabel}>
                  {mode === 'decide' ? 'Situation / Context' : 'Source input'}
                </span>
                {mode === 'extract' ? (
                  <MediaDropzone
                    attachments={drawerRow.attachments}
                    onChange={(next) => updateRow(drawerRow.id, { attachments: next })}
                    text={drawerRow.stateText}
                    onTextChange={(next) => updateRow(drawerRow.id, { stateText: next })}
                    onClear={() => updateRow(drawerRow.id, { stateText: '', attachments: [] })}
                    disabled={running}
                    placeholder="Paste text, or drop / paste an image or PDF…"
                  />
                ) : (
                  <textarea
                    className={styles.drawerTextarea}
                    rows={6}
                    value={drawerRow.stateText}
                    disabled={running}
                    placeholder="Paste plain text or JSON context to evaluate…"
                    onChange={(e) => updateRow(drawerRow.id, { stateText: e.target.value })}
                    aria-label="Case input context"
                  />
                )}
              </div>

              {drawerRow.status === 'error' && drawerRow.error !== undefined && drawerRow.error !== '' ? (
                <Alert tone="danger" title="Case failed">
                  {drawerRow.error}
                </Alert>
              ) : null}

              {drawerRow.result !== undefined ? (
                <div className={styles.drawerResults}>
                  <div className={styles.drawerResultsHead}>
                    <span className={styles.drawerSectionLabel}>Structured output</span>
                    <div className={styles.drawerResultBadges}>
                      {drawerRowMeta !== undefined ? (
                        drawerRowMeta.needsReview ? (
                          <Badge size="sm" tone="warning" pill>
                            {`${String(drawerRowMeta.reviewCount)} needs review`}
                          </Badge>
                        ) : (
                          <Badge size="sm" tone="success" pill>
                            Auto-routed
                          </Badge>
                        )
                      ) : null}
                      {drawerRowMeta?.expectedMatch === true ? (
                        <Badge size="sm" tone="success" outline>
                          Expected match
                        </Badge>
                      ) : drawerRowMeta?.expectedMatch === false ? (
                        <Badge size="sm" tone="danger" outline>
                          Expected mismatch
                        </Badge>
                      ) : null}
                    </div>
                  </div>

                  <div className={styles.drawerResultsCard}>
                    <ResultsView
                      result={drawerRow.result}
                      requestFields={requestFieldsMap}
                      mode={mode}
                      presets={[]}
                      canRun={false}
                      onRun={noop}
                      onPickPreset={noop}
                      policy={policy}
                      onPolicyChange={onPolicyChange}
                      hidePolicyBar
                    />
                  </div>

                  <Disclosure title="Raw JSON response">
                    <CodePanel
                      code={JSON.stringify(drawerRow.result, null, 2)}
                      label="Raw JSON response"
                      maxLines={18}
                    />
                  </Disclosure>
                </div>
              ) : null}

              <div className={styles.drawerFooter}>
                <div className={styles.drawerNavGroup}>
                  <IconButton
                    label="Previous case"
                    icon={<ChevronLeft size={14} strokeWidth={1.75} aria-hidden="true" />}
                    disabled={drawerRowIndex <= 0}
                    onClick={() => {
                      const prev = rows[drawerRowIndex - 1];
                      if (prev !== undefined) setInspectRowId(prev.id);
                    }}
                  />
                  <IconButton
                    label="Next case"
                    icon={<ChevronRight size={14} strokeWidth={1.75} aria-hidden="true" />}
                    disabled={drawerRowIndex < 0 || drawerRowIndex >= rows.length - 1}
                    onClick={() => {
                      const next = rows[drawerRowIndex + 1];
                      if (next !== undefined) setInspectRowId(next.id);
                    }}
                  />
                </div>

                <Tooltip content={!canRun ? runTooltip : 'Run inference on this case'}>
                  <span className={styles.tooltipWrap}>
                    <Button
                      type="button"
                      variant="primary"
                      size="sm"
                      loading={running && runningRowId === drawerRow.id}
                      disabled={running || !canRun}
                      onClick={() => onRunRow(drawerRow.id)}
                    >
                      <Play size={12} strokeWidth={2} aria-hidden="true" />
                      Run this case
                    </Button>
                  </span>
                </Tooltip>
              </div>
            </div>
          ) : null}
        </DrawerContent>
      </Drawer>
    </div>
  );
}

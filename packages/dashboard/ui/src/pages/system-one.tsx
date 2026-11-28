import { CircleX, Code2, Download, FileDown, History, Play, Square } from 'lucide-react';
import { type ReactElement, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { CodePanel } from '../components/arc-x/code-panel/code-panel';
import { Disclosure } from '../components/arc-x/disclosure/disclosure';
import { MediaDropzone } from '../components/arc-x/media-dropzone/media-dropzone';
import { ModelPicker } from '../components/arc-x/model-picker/model-picker';
import { shownShare } from '../components/arc-x/number-value/number-value';
import { Panel } from '../components/arc-x/panel/panel';
import { type HighlightSpanEntry, SpanHighlighter } from '../components/arc-x/span-highlighter/span-highlighter';
import { Meter } from '../components/arc-x/split-bar/split-bar';
import { StatGroup, StatTile } from '../components/arc-x/stat-tile/stat-tile';
import { WaitLine } from '../components/arc-x/wait-line/wait-line';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { Button } from '../components/arc/button/button';
import { CopyButton } from '../components/arc/copy-button/copy-button';
import { EmptyState } from '../components/arc/empty-state/empty-state';
import { Input } from '../components/arc/input/input';
import SegmentedControl from '../components/arc/segmented-control/segmented-control';
import { Switch } from '../components/arc/switch/switch';
import { Tooltip } from '../components/arc/tooltip/tooltip';
import { WaitClock } from '../components/wait-clock';
import { DashboardApiError, getJson, mutate, subscribeSystemOne, type Subscription } from '../lib/api';
import { getConnectionGeneration, subscribeConnection } from '../lib/connection';
import { errMessage } from '../lib/errors';
import { formatClock, formatCount, formatDurationParts, formatPercent } from '../lib/format';
import { useHost } from '../lib/host';
import { useRuntimeLive } from '../lib/live';
import {
  type BatchRow,
  type BatchSampleDataset,
  buildAutoDetectPayload,
  buildClairicPayload,
  buildClefPayload,
  buildCurlCommand,
  buildNodeSdkSnippet,
  buildPythonSnippet,
  buildSuggestedFields,
  buildTypeScriptSnippet,
  type ClefQuestionDraft,
  createBatchRow,
  createSpecId,
  type DecisionPolicy,
  DEFAULT_DECISION_POLICY,
  DEFAULT_SYSTEMONE_ENDPOINT,
  extractAnswerSpans,
  fieldAccentColor,
  type FieldDraft,
  generateSampleReceiptPng,
  isPlainObject,
  loadSavedSpecs,
  type MediaAttachment,
  normalizePolicy,
  parseClairicPayload,
  parseClefPayload,
  persistSavedSpecs,
  requestedModel,
  type SavedSpec,
  summarizeResult,
  summarizeRouting,
  SYSTEM_ONE_PRESETS,
  type SystemOneMode,
  type SystemOnePreset,
  validateClefQuestions,
  validateFieldDrafts,
} from '../lib/system-one';
import { toast } from '../lib/toast';
import type {
  CatalogResponse,
  CatalogUpdatesResponse,
  ModelsResponse,
  SystemOneHistoryEntry,
  SystemOneJobSnapshot,
  SystemOneModelInfo,
  SystemOneState,
} from '../lib/types';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { type Hotkey, useHotkeys } from '../shell/hotkeys';
import { Page, PageHeader } from '../shell/page';
import { useModelDownloads } from './models/use-model-downloads';
import { ApiGuideDrawer } from './system-one/api-guide-drawer';
import { BatchView } from './system-one/batch-view';
import { ClefBuilder } from './system-one/clef-builder';
import { HistoryDrawer } from './system-one/history-drawer';
import { restoreHistoryMedia, systemOneModelBases } from './system-one/model-options';
import { NoSystemOneModelsCard, SystemOneModelsDrawer } from './system-one/models-drawer';
import { ResultsView } from './system-one/results-view';
import { SchemaBuilder } from './system-one/schema-builder';
import { SpecsPopover } from './system-one/specs-popover';
import styles from './system-one/system-one.module.css';

type WorkspaceMode = 'single' | 'batch';
type CodeFormat = 'json' | 'ts' | 'python' | 'curl' | 'sdk';

/** The last failure of a run or of an auto-detect, shown in the Output status line until the next one ends. */
interface RunFailure {
  seq: number;
  kind: 'run' | 'detect';
  title: string;
  text: string;
  mode: SystemOneMode;
}

export default function SystemOne(): ReactElement {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedModelParam = searchParams.get('model');
  const requestedBuildParam = searchParams.get('build');

  const connectionGen = useSyncExternalStore(subscribeConnection, getConnectionGeneration, getConnectionGeneration);
  const live = useRuntimeLive();
  const stateQuery = useJson<SystemOneState>('/systemone/state');
  const modelsQuery = useJson<ModelsResponse>('/models');
  const catalogQuery = useJson<CatalogResponse>('/catalog');
  const hostQuery = useHost();
  const updatesQuery = useJson<CatalogUpdatesResponse>('/catalog/updates');
  const downloads = useModelDownloads(modelsQuery, catalogQuery, updatesQuery, hostQuery);

  const liveRunning = live === undefined ? undefined : live.processes.some((p) => p.status === 'running');
  const liveResidentKey = useMemo(() => {
    if (live === undefined) return '';
    const names: string[] = [];
    for (const proc of live.processes) {
      if (proc.status !== 'running') continue;
      for (const m of proc.models) names.push(m.name);
    }
    return names.sort().join(',');
  }, [live]);

  // Reload System One state whenever the engine starts/stops or resident models change.
  const prevLiveKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (liveRunning === undefined) return;
    const key = `${String(liveRunning)}:${liveResidentKey}`;
    if (prevLiveKeyRef.current !== null && prevLiveKeyRef.current !== key) {
      stateQuery.reload();
    } else if (
      prevLiveKeyRef.current === null &&
      stateQuery.data !== undefined &&
      stateQuery.data.sidecarRunning !== liveRunning
    ) {
      stateQuery.reload();
    }
    prevLiveKeyRef.current = key;
  }, [liveRunning, liveResidentKey, stateQuery.data, stateQuery.reload]);

  const installedModels = stateQuery.data?.models ?? [];
  const history = stateQuery.data?.history ?? [];
  const residentModel = installedModels.find((m) => m.resident);
  const sidecarOffline =
    liveRunning !== undefined ? !liveRunning : stateQuery.data !== undefined && !stateQuery.data.sidecarRunning;

  const [mode, setMode] = useState<SystemOneMode>('extract');
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>('single');
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [leftView, setLeftView] = useState<'visual' | 'json'>('visual');
  const [rightTab, setRightTab] = useState<'answers' | 'evidence' | 'code'>('answers');
  const [codeFormat, setCodeFormat] = useState<CodeFormat>('json');
  const [focusInvalidNonce, setFocusInvalidNonce] = useState(0);

  // Calibrated routing & confidence thresholds
  const [policy, setPolicy] = useState<DecisionPolicy>(DEFAULT_DECISION_POLICY);

  // Reusable local library of saved schemas & rubrics
  const [savedSpecs, setSavedSpecs] = useState<SavedSpec[]>(() => loadSavedSpecs());

  // Start clean & empty — both text and media are optional; examples are 1-click on the right.
  const [extractStateText, setExtractStateText] = useState<string>('');
  const [decideStateText, setDecideStateText] = useState<string>('');
  const [referenceDate, setReferenceDate] = useState<string>('');
  const [includeQuantiles, setIncludeQuantiles] = useState<boolean>(false);
  const [returnStateText, setReturnStateText] = useState<boolean>(true);
  const [fields, setFields] = useState<FieldDraft[]>([]);
  const [attachments, setAttachments] = useState<MediaAttachment[]>([]);

  // CLEF questions state
  const [clefQuestions, setClefQuestions] = useState<ClefQuestionDraft[]>([]);

  // Batch evaluation state (separate per mode so switching Extract <-> Decide preserves datasets)
  const [extractBatchRows, setExtractBatchRows] = useState<BatchRow[]>([]);
  const [decideBatchRows, setDecideBatchRows] = useState<BatchRow[]>([]);
  const [extractPrimaryCol, setExtractPrimaryCol] = useState<string>('');
  const [decidePrimaryCol, setDecidePrimaryCol] = useState<string>('');
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchRunningRowId, setBatchRunningRowId] = useState<string | null>(null);
  const batchCancelledRef = useRef(false);
  const batchActiveJobIdRef = useRef<string | null>(null);
  const batchActiveSubRef = useRef<Subscription | null>(null);
  const unmountedRef = useRef(false);

  const batchRows = mode === 'decide' ? decideBatchRows : extractBatchRows;
  const setBatchRows = useCallback(
    (updater: BatchRow[] | ((prev: BatchRow[]) => BatchRow[]), targetMode: SystemOneMode = mode): void => {
      if (targetMode === 'decide') {
        setDecideBatchRows((prev) => (typeof updater === 'function' ? updater(prev) : updater));
      } else {
        setExtractBatchRows((prev) => (typeof updater === 'function' ? updater(prev) : updater));
      }
    },
    [mode],
  );

  const primaryColumn = mode === 'decide' ? decidePrimaryCol : extractPrimaryCol;
  const setPrimaryColumn = useCallback(
    (col: string, targetMode: SystemOneMode = mode): void => {
      if (targetMode === 'decide') setDecidePrimaryCol(col);
      else setExtractPrimaryCol(col);
    },
    [mode],
  );

  const stateText = mode === 'decide' ? decideStateText : extractStateText;
  const setStateText = useCallback(
    (next: string, targetMode: SystemOneMode = mode): void => {
      if (targetMode === 'decide') setDecideStateText(next);
      else setExtractStateText(next);
    },
    [mode],
  );

  // Keep primaryColumn synced to first valid field/question if unset or deleted
  const activeColumnNames = useMemo(
    () =>
      mode === 'decide'
        ? clefQuestions.map((q) => q.name.trim()).filter((n) => n !== '')
        : fields.map((f) => f.name.trim()).filter((n) => n !== ''),
    [mode, clefQuestions, fields],
  );

  useEffect(() => {
    if (activeColumnNames.length === 0) return;
    if (primaryColumn === '' || !activeColumnNames.includes(primaryColumn)) {
      setPrimaryColumn(activeColumnNames[0]!, mode);
    }
  }, [activeColumnNames, primaryColumn, mode, setPrimaryColumn]);

  // Bidirectional Raw JSON editor
  const [rawJsonDraft, setRawJsonDraft] = useState<string | null>(null);
  const [rawJsonError, setRawJsonError] = useState<string | null>(null);

  // Active single-run & output state
  const [activeJob, setActiveJob] = useState<SystemOneJobSnapshot | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | undefined>(undefined);
  /** The mode `result` came from, and the mode of the last submitted request: the other mode must not show them. */
  const [resultMode, setResultMode] = useState<SystemOneMode | null>(null);
  const [requestMode, setRequestMode] = useState<SystemOneMode | null>(null);
  const [lastElapsedMs, setLastElapsedMs] = useState<number | null>(null);
  const [lastRequestPayload, setLastRequestPayload] = useState<Record<string, unknown> | null>(null);
  const [lastSubmittedAttachments, setLastSubmittedAttachments] = useState<MediaAttachment[]>([]);
  const [runFailure, setRunFailure] = useState<RunFailure | null>(null);
  const [resultRun, setResultRun] = useState<{ model: string; at: number } | null>(null);
  const runLineRef = useRef<HTMLDivElement | null>(null);

  const failRun = useCallback(
    (kind: RunFailure['kind'], title: string, text: string, failedMode: SystemOneMode): void => {
      setRunFailure((prev) => ({ seq: (prev?.seq ?? 0) + 1, kind, title, text, mode: failedMode }));
      if (kind === 'run') {
        // A failed run has no answers. The answers of an earlier input must not stay below its error.
        setResult(undefined);
        setResultMode(null);
        setResultRun(null);
        setLastElapsedMs(null);
      }
      const line = runLineRef.current;
      if (line === null || shownShare(line) < 1) toast.error(title, { id: 'system-one-run', description: text });
    },
    [],
  );
  const [submitting, setSubmitting] = useState(false);
  const [autoDetecting, setAutoDetecting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);

  // Drawers & popovers
  const [modelsDrawerOpen, setModelsDrawerOpen] = useState(false);
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false);
  const [apiGuideOpen, setApiGuideOpen] = useState(false);
  const [specsOpen, setSpecsOpen] = useState(false);
  const [focusedSpanField, setFocusedSpanField] = useState<string | null>(null);

  const subRef = useRef<Subscription | null>(null);
  const subscribedJobIdRef = useRef<string | null>(null);
  const subscribedConnRef = useRef<number>(connectionGen);
  const lastReconciledKeyRef = useRef<string | null>(null);
  const pendingAutoDetectRef = useRef<{
    jobId: string;
    thenRun: boolean;
    modelId: string;
    stateText: string;
    referenceDate: string;
    includeQuantiles: boolean;
    returnStateText: boolean;
    attachments: MediaAttachment[];
  } | null>(null);
  const launchRunRef = useRef<
    (
      modelId: string,
      payloadToSend: Record<string, unknown>,
      targetMode: SystemOneMode,
      activeAttachments: MediaAttachment[],
    ) => Promise<void>
  >(async () => {});

  // Reload System One models when a download finishes and refreshes `/models`.
  const prevModelCount = useRef<number | null>(null);
  useEffect(() => {
    const count = modelsQuery.data?.models.length;
    if (count === undefined) return;
    if (prevModelCount.current !== null && prevModelCount.current !== count) {
      stateQuery.reload();
    }
    prevModelCount.current = count;
  }, [modelsQuery.data?.models.length, stateQuery.reload]);

  // Compatible models for the active mode (never cross-wire a Clairic model to a CLEF request).
  const modelsForMode = useMemo(
    () => installedModels.filter((m) => (mode === 'decide' ? m.modelType === 'clef' : m.modelType === 'clairic')),
    [installedModels, mode],
  );

  useEffect(() => {
    if (installedModels.length === 0) return;

    if (requestedModelParam !== null && requestedModelParam !== '') {
      const match = requestedModel(installedModels, requestedModelParam, requestedBuildParam);
      if (match !== undefined) {
        setSelectedModelId(match.id);
        setMode(match.modelType === 'clef' ? 'decide' : 'extract');
        setSearchParams(
          (current) => {
            const next = new URLSearchParams(current);
            next.delete('model');
            next.delete('build');
            return next;
          },
          { replace: true },
        );
        return;
      }
    }

    const currentValid = modelsForMode.some((m) => m.id === selectedModelId);
    if (!currentValid) {
      const preferred = modelsForMode.find((m) => m.resident) ?? modelsForMode[0];
      if (preferred !== undefined) {
        setSelectedModelId(preferred.id);
      }
    }
  }, [installedModels, modelsForMode, requestedBuildParam, requestedModelParam, selectedModelId, setSearchParams]);

  const selectedModelForMode: SystemOneModelInfo | undefined =
    modelsForMode.find((m) => m.id === selectedModelId) ?? modelsForMode[0];
  const pickerDisplayModel: SystemOneModelInfo | undefined =
    selectedModelForMode ?? installedModels.find((m) => m.id === selectedModelId) ?? installedModels[0];

  const isRunning = activeJob?.status === 'running' || submitting || batchRunning;

  const catalogItems = catalogQuery.data?.items;
  const modelBases = useMemo(
    () => systemOneModelBases(installedModels, pickerDisplayModel?.id ?? null, isRunning, catalogItems),
    [installedModels, pickerDisplayModel?.id, isRunning, catalogItems],
  );

  // Computed structured payload from visual editors
  const structuredPayload = useMemo(() => {
    if (mode === 'decide') {
      return buildClefPayload({ stateText, questions: clefQuestions });
    }
    return buildClairicPayload({
      stateText,
      referenceDate,
      includeQuantiles,
      returnStateText,
      fields,
      attachments,
    });
  }, [mode, stateText, referenceDate, includeQuantiles, returnStateText, fields, attachments, clefQuestions]);

  const formattedPayloadJson = useMemo(() => JSON.stringify(structuredPayload, null, 2), [structuredPayload]);
  const rawJsonValue = rawJsonDraft ?? formattedPayloadJson;

  // Validation
  const fieldWarnings = useMemo(() => validateFieldDrafts(fields), [fields]);
  const clefWarnings = useMemo(() => validateClefQuestions(clefQuestions), [clefQuestions]);
  const hasValidationErrors =
    mode === 'decide'
      ? clefQuestions.length === 0 || Object.keys(clefWarnings).length > 0
      : fields.length === 0 || Object.keys(fieldWarnings).length > 0;

  const canSaveCurrentSpec =
    mode === 'decide'
      ? clefQuestions.length > 0 && Object.keys(clefWarnings).length === 0
      : fields.length > 0 && Object.keys(fieldWarnings).length === 0;

  const closeSubscription = useCallback((): void => {
    subRef.current?.close();
    subRef.current = null;
    subscribedJobIdRef.current = null;
  }, []);

  const applyJobSnapshot = useCallback(
    (snap: SystemOneJobSnapshot): void => {
      const pending = pendingAutoDetectRef.current;
      if (pending !== null && pending.jobId === snap.id) {
        setActiveJob(snap.status === 'running' ? snap : null);
        if (snap.status === 'running') return;
        pendingAutoDetectRef.current = null;
        setAutoDetecting(false);
        setCancelling(false);
        if (snap.status === 'done') {
          const suggested = buildSuggestedFields(snap.result, pending.stateText);
          setFields(suggested.fields);
          setRawJsonDraft(null);
          setRawJsonError(null);
          setRunFailure(null);
          if (pending.thenRun) {
            toast.info(`Detected ${suggested.categoryLabel} — extracting ${String(suggested.fields.length)} fields…`);
            const nextPayload = buildClairicPayload({
              stateText: pending.stateText,
              referenceDate: pending.referenceDate,
              includeQuantiles: pending.includeQuantiles,
              returnStateText: pending.returnStateText,
              fields: suggested.fields,
              attachments: pending.attachments,
            });
            void launchRunRef.current(pending.modelId, nextPayload, 'extract', pending.attachments);
          } else {
            toast.success(`Detected ${suggested.categoryLabel} — ${String(suggested.fields.length)} fields ready`);
          }
        } else if (snap.status === 'error') {
          failRun('detect', 'Auto-detect failed', snap.error ?? 'The engine gave no reason.', 'extract');
        } else if (snap.status === 'cancelled') {
          toast.info('Auto-detect cancelled');
        }
        return;
      }

      setActiveJob(snap.status === 'running' ? snap : null);
      if (snap.status === 'done') {
        setResult(snap.result);
        setResultMode(snap.mode);
        setResultRun({ model: snap.model, at: snap.startedAt });
        setLastElapsedMs(snap.elapsedMs);
        setRunFailure(null);
        setCancelling(false);
        stateQuery.reload();
      } else if (snap.status === 'error') {
        failRun('run', 'Run failed', snap.error ?? 'The engine gave no reason.', snap.mode);
        setCancelling(false);
        stateQuery.reload();
      } else if (snap.status === 'cancelled') {
        setCancelling(false);
        toast.info('Run cancelled');
        stateQuery.reload();
      }
    },
    [stateQuery.reload, failRun],
  );

  // Subscribe to a running single job
  const attachJobSubscription = useCallback(
    (jobId: string): void => {
      if (subscribedJobIdRef.current === jobId && subscribedConnRef.current === connectionGen) return;
      subRef.current?.close();
      subscribedJobIdRef.current = jobId;
      subscribedConnRef.current = connectionGen;
      subRef.current = subscribeSystemOne(
        jobId,
        (snap) => {
          applyJobSnapshot(snap);
        },
        () => {
          if (subscribedJobIdRef.current === jobId) {
            subscribedJobIdRef.current = null;
          }
        },
      );
    },
    [applyJobSnapshot, connectionGen],
  );

  // Re-attach if backend already has an active job on mount or after RPC reconnect
  useEffect(() => {
    const jobId = stateQuery.data?.activeJobId;
    if (jobId != null && jobId !== '' && batchActiveJobIdRef.current !== jobId) {
      attachJobSubscription(jobId);
    }
  }, [stateQuery.data?.activeJobId, connectionGen, attachJobSubscription]);

  // Reconcile local running job against refreshed state or RPC reconnect so Run is never stranded
  useEffect(() => {
    if (activeJob === null || activeJob.status !== 'running') {
      lastReconciledKeyRef.current = null;
      return;
    }
    const serverActiveId = stateQuery.data?.activeJobId;
    const mismatch = stateQuery.data !== undefined && serverActiveId !== activeJob.id;
    const reconnected = subscribedConnRef.current !== connectionGen;
    if (!mismatch && !reconnected) return;

    const reconcileKey = `${activeJob.id}:${String(connectionGen)}:${String(serverActiveId ?? 'none')}`;
    if (lastReconciledKeyRef.current === reconcileKey) return;
    lastReconciledKeyRef.current = reconcileKey;

    let cancelled = false;
    void getJson<SystemOneJobSnapshot>(`/systemone/jobs/${encodeURIComponent(activeJob.id)}`)
      .then((snap) => {
        if (cancelled) return;
        if (snap.status === 'running') {
          attachJobSubscription(snap.id);
        } else {
          closeSubscription();
          applyJobSnapshot(snap);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof DashboardApiError && (err.status === 404 || err.code === 'E_NOT_FOUND')) {
          closeSubscription();
          pendingAutoDetectRef.current = null;
          setAutoDetecting(false);
          setActiveJob(null);
          setCancelling(false);
          failRun('run', 'Run interrupted', 'The engine no longer has this run. Run it again.', activeJob.mode);
          return;
        }
        if (mismatch) {
          closeSubscription();
          pendingAutoDetectRef.current = null;
          setAutoDetecting(false);
          setActiveJob(null);
          setCancelling(false);
          failRun('run', 'Run interrupted', 'The engine no longer has this run. Run it again.', activeJob.mode);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeJob, stateQuery.data, connectionGen, attachJobSubscription, applyJobSnapshot, closeSubscription]);

  // Clean up subscriptions and abort active batch runs on unmount
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      batchCancelledRef.current = true;
      closeSubscription();
      batchActiveSubRef.current?.close();
      batchActiveSubRef.current = null;
      const activeBatchId = batchActiveJobIdRef.current;
      if (activeBatchId !== null) {
        batchActiveJobIdRef.current = null;
        void mutate<{ ok: true }>('POST', `/systemone/jobs/${encodeURIComponent(activeBatchId)}/cancel`).catch(
          () => {},
        );
      }
    };
  }, [closeSubscription]);

  const applyRawJson = (text: string): void => {
    setRawJsonDraft(text);
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isPlainObject(parsed)) {
        setRawJsonError('Payload must be a JSON object.');
        return;
      }
      setRawJsonError(null);
      if (Object.hasOwn(parsed, 'questions') && !Object.hasOwn(parsed, 'fields')) {
        const next = parseClefPayload(parsed);
        setMode('decide');
        setDecideStateText(next.stateText);
        setClefQuestions(next.questions);
      } else {
        const next = parseClairicPayload(parsed);
        setMode('extract');
        setExtractStateText(next.stateText);
        setReferenceDate(next.referenceDate);
        setIncludeQuantiles(next.includeQuantiles);
        setReturnStateText(next.returnStateText);
        setFields(next.fields);
      }
    } catch (err) {
      setRawJsonError(err instanceof Error ? err.message : 'Invalid JSON');
    }
  };

  const applyPreset = useCallback((preset: SystemOnePreset): void => {
    pendingAutoDetectRef.current = null;
    setAutoDetecting(false);
    setSpecsOpen(false);
    setRawJsonDraft(null);
    setRawJsonError(null);
    setRunFailure(null);
    setFocusedSpanField(null);
    setMode(preset.mode);
    setWorkspaceMode('single');
    if (preset.mode === 'decide') {
      const parsed = parseClefPayload(preset.payload);
      setDecideStateText(parsed.stateText);
      setClefQuestions(parsed.questions);
      setAttachments([]);
    } else {
      const parsed = parseClairicPayload(preset.payload);
      setExtractStateText(parsed.stateText);
      setReferenceDate(parsed.referenceDate);
      setIncludeQuantiles(parsed.includeQuantiles);
      setReturnStateText(parsed.returnStateText);
      setFields(parsed.fields);
      if (preset.needsSampleReceipt === true) {
        const sample = generateSampleReceiptPng();
        setAttachments(sample === null ? [] : [sample]);
      } else {
        setAttachments([]);
      }
    }
    toast.success(`Loaded example: ${preset.label}`);
  }, []);

  const handleSaveCurrentSpec = useCallback(
    (name: string): void => {
      const trimmed = name.trim();
      if (trimmed === '') return;
      const spec: SavedSpec =
        mode === 'decide'
          ? {
              id: createSpecId(),
              name: trimmed,
              mode: 'decide',
              updatedAt: Date.now(),
              questions: (buildClefPayload({ stateText: '', questions: clefQuestions }).questions ?? {}) as Record<
                string,
                unknown
              >,
              policy,
            }
          : {
              id: createSpecId(),
              name: trimmed,
              mode: 'extract',
              updatedAt: Date.now(),
              fields: (buildClairicPayload({
                stateText: '',
                referenceDate,
                includeQuantiles,
                returnStateText,
                fields,
                attachments: [],
              }).fields ?? {}) as Record<string, unknown>,
              options: { referenceDate, includeQuantiles, returnStateText },
              policy,
            };
      const next = [spec, ...savedSpecs.filter((item) => item.name.toLowerCase() !== trimmed.toLowerCase())];
      const persisted = persistSavedSpecs(next);
      setSavedSpecs(next);
      if (persisted) {
        toast.success(`Saved ${mode === 'decide' ? 'rubric' : 'schema'}: ${trimmed}`);
      } else {
        toast.info(`Saved for this session only (browser storage unavailable)`);
      }
    },
    [mode, clefQuestions, fields, referenceDate, includeQuantiles, returnStateText, policy, savedSpecs],
  );

  const handleApplySavedSpec = useCallback((spec: SavedSpec): void => {
    pendingAutoDetectRef.current = null;
    setAutoDetecting(false);
    setSpecsOpen(false);
    setRawJsonDraft(null);
    setRawJsonError(null);
    setRunFailure(null);
    setMode(spec.mode);
    if (spec.mode === 'decide') {
      const parsed = parseClefPayload({ questions: spec.questions ?? {} });
      setClefQuestions(parsed.questions);
    } else {
      const parsed = parseClairicPayload({
        fields: spec.fields ?? {},
        reference_date: spec.options?.referenceDate ?? '',
        options: {
          include_quantiles: spec.options?.includeQuantiles === true,
          return_state_text: spec.options?.returnStateText !== false,
        },
      });
      setFields(parsed.fields);
      setReferenceDate(parsed.referenceDate);
      setIncludeQuantiles(parsed.includeQuantiles);
      setReturnStateText(parsed.returnStateText);
    }
    if (spec.policy !== undefined) {
      setPolicy(normalizePolicy(spec.policy));
    }
    toast.success(`Applied ${spec.name} (kept current input)`);
  }, []);

  const handleDeleteSavedSpec = useCallback((id: string): void => {
    setSavedSpecs((prev) => {
      const next = prev.filter((item) => item.id !== id);
      persistSavedSpecs(next);
      return next;
    });
  }, []);

  const handleLoadBatchSample = useCallback(
    (sample: BatchSampleDataset): void => {
      const preset = SYSTEM_ONE_PRESETS.find((p) => p.id === sample.presetId);
      setMode(sample.mode);
      setWorkspaceMode('batch');
      setRawJsonDraft(null);
      setRawJsonError(null);
      setRunFailure(null);
      if (preset !== undefined) {
        if (sample.mode === 'decide') {
          const parsed = parseClefPayload(preset.payload);
          setClefQuestions(parsed.questions);
        } else {
          const parsed = parseClairicPayload(preset.payload);
          setFields(parsed.fields);
          setReferenceDate(parsed.referenceDate);
          setIncludeQuantiles(parsed.includeQuantiles);
          setReturnStateText(parsed.returnStateText);
        }
      }
      const nextRows = sample.rows.map((r) =>
        createBatchRow({
          label: r.label,
          stateText: r.stateText,
          expected: r.expected,
        }),
      );
      setBatchRows(nextRows, sample.mode);
      setPrimaryColumn(sample.primaryField, sample.mode);
      toast.success(`Loaded batch: ${sample.label}`);
    },
    [setBatchRows, setPrimaryColumn],
  );

  const launchRun = useCallback(
    async (
      modelId: string,
      payloadToSend: Record<string, unknown>,
      targetMode: SystemOneMode,
      activeAttachments: MediaAttachment[],
    ): Promise<void> => {
      setSubmitting(true);
      setFocusedSpanField(null);
      setLastRequestPayload(payloadToSend);
      setRequestMode(targetMode);
      setLastSubmittedAttachments(targetMode === 'extract' ? activeAttachments : []);
      try {
        const res = await mutate<{ jobId: string }>('POST', '/systemone/run', {
          model: modelId,
          payload: payloadToSend,
          mode: targetMode,
        });
        setActiveJob({
          id: res.jobId,
          status: 'running',
          model: modelId,
          mode: targetMode,
          startedAt: Date.now(),
          elapsedMs: 0,
        });
        attachJobSubscription(res.jobId);
      } catch (err) {
        failRun('run', 'Run did not start', errMessage(err), targetMode);
      } finally {
        setSubmitting(false);
      }
    },
    [attachJobSubscription],
  );

  launchRunRef.current = launchRun;

  const startAutoDetect = useCallback(
    async (opts: { thenRun: boolean }): Promise<void> => {
      if (isRunning) return;
      if (selectedModelForMode === undefined || selectedModelForMode.modelType !== 'clairic') {
        setModelsDrawerOpen(true);
        toast.info('Install a Clairic model to auto-detect fields');
        return;
      }
      if (sidecarOffline) {
        toast.info('Inference engine is stopped', {
          description: 'Start it from the Engine page or menu bar before running.',
        });
        return;
      }
      if (stateText.trim() === '' && attachments.length === 0) {
        toast.info('Paste text or attach an image/PDF first');
        return;
      }

      const detectPayload = buildAutoDetectPayload({ stateText, attachments });
      setSubmitting(true);
      setAutoDetecting(true);
      try {
        const res = await mutate<{ jobId: string }>('POST', '/systemone/run', {
          model: selectedModelForMode.id,
          payload: detectPayload,
          mode: 'extract',
          skipHistory: true,
        });
        pendingAutoDetectRef.current = {
          jobId: res.jobId,
          thenRun: opts.thenRun,
          modelId: selectedModelForMode.id,
          stateText,
          referenceDate,
          includeQuantiles,
          returnStateText,
          attachments,
        };
        setActiveJob({
          id: res.jobId,
          status: 'running',
          model: selectedModelForMode.id,
          mode: 'extract',
          startedAt: Date.now(),
          elapsedMs: 0,
        });
        attachJobSubscription(res.jobId);
      } catch (err) {
        pendingAutoDetectRef.current = null;
        setAutoDetecting(false);
        failRun('detect', 'Auto-detect did not start', errMessage(err), 'extract');
      } finally {
        setSubmitting(false);
      }
    },
    [
      isRunning,
      selectedModelForMode,
      sidecarOffline,
      stateText,
      attachments,
      referenceDate,
      includeQuantiles,
      returnStateText,
      attachJobSubscription,
    ],
  );

  // Execute one batch row over the existing async job channel (skipping single-run history).
  const runJobToCompletion = useCallback(
    async (
      modelId: string,
      payloadToSend: Record<string, unknown>,
      targetMode: SystemOneMode,
    ): Promise<SystemOneJobSnapshot> => {
      const { jobId } = await mutate<{ jobId: string }>('POST', '/systemone/run', {
        model: modelId,
        payload: payloadToSend,
        mode: targetMode,
        skipHistory: true,
      });
      if (batchCancelledRef.current || unmountedRef.current) {
        void mutate<{ ok: true }>('POST', `/systemone/jobs/${encodeURIComponent(jobId)}/cancel`).catch(() => {});
        return {
          id: jobId,
          status: 'cancelled',
          model: modelId,
          mode: targetMode,
          startedAt: Date.now(),
          elapsedMs: 0,
        };
      }
      batchActiveJobIdRef.current = jobId;

      return new Promise<SystemOneJobSnapshot>((resolve) => {
        let settled = false;
        let pollTimer: ReturnType<typeof setTimeout> | null = null;
        const finish = (snap: SystemOneJobSnapshot): void => {
          if (settled) return;
          settled = true;
          if (pollTimer !== null) clearTimeout(pollTimer);
          batchActiveSubRef.current?.close();
          batchActiveSubRef.current = null;
          if (batchActiveJobIdRef.current === jobId) batchActiveJobIdRef.current = null;
          resolve(snap);
        };
        const pollUntilTerminal = (): void => {
          if (settled) return;
          if (unmountedRef.current) {
            finish({
              id: jobId,
              status: 'cancelled',
              model: modelId,
              mode: targetMode,
              startedAt: Date.now(),
              elapsedMs: 0,
            });
            return;
          }
          void getJson<SystemOneJobSnapshot>(`/systemone/jobs/${encodeURIComponent(jobId)}`)
            .then((snap) => {
              if (settled) return;
              if (snap.status !== 'running') {
                finish(snap);
              } else {
                pollTimer = setTimeout(pollUntilTerminal, 400);
              }
            })
            .catch((err: unknown) => {
              finish({
                id: jobId,
                status: 'error',
                model: modelId,
                mode: targetMode,
                startedAt: Date.now(),
                elapsedMs: 0,
                error: errMessage(err),
              });
            });
        };
        batchActiveSubRef.current = subscribeSystemOne(
          jobId,
          (snap) => {
            if (snap.status !== 'running') finish(snap);
          },
          () => {
            if (settled) return;
            pollUntilTerminal();
          },
        );
      });
    },
    [],
  );

  const executeBatchQueue = useCallback(
    async (targetRowIds: readonly string[]): Promise<void> => {
      if (isRunning) return;
      if (selectedModelForMode === undefined) {
        setModelsDrawerOpen(true);
        toast.info(`Install a ${mode === 'decide' ? 'CLEF' : 'Clairic'} model first`);
        return;
      }
      if (sidecarOffline) {
        toast.info('Inference engine is stopped', {
          description: 'Start it from the Engine page or menu bar before running.',
        });
        return;
      }
      if (hasValidationErrors) {
        toast.error(
          mode === 'decide'
            ? 'Add at least one valid question before running batch'
            : 'Add at least one valid field before running batch',
        );
        return;
      }

      const activeMode = mode;
      const snapshotRows = batchRows.filter(
        (r) =>
          targetRowIds.includes(r.id) &&
          (r.stateText.trim() !== '' || (activeMode === 'extract' && r.attachments.length > 0)),
      );
      if (snapshotRows.length === 0) {
        toast.info('Add at least one case with text or an attachment first');
        return;
      }

      batchCancelledRef.current = false;
      setBatchRunning(true);
      setCancelling(false);

      const idSet = new Set(snapshotRows.map((r) => r.id));
      setBatchRows(
        (prev) => prev.map((row) => (idSet.has(row.id) ? { ...row, status: 'queued', error: undefined } : row)),
        activeMode,
      );

      let completedCount = 0;
      let failedCount = 0;
      let stoppedEarly = false;
      try {
        for (const row of snapshotRows) {
          if (batchCancelledRef.current || unmountedRef.current) {
            stoppedEarly = true;
            break;
          }
          setBatchRunningRowId(row.id);
          setBatchRows(
            (prev) =>
              prev.map((item) =>
                item.id === row.id
                  ? { ...item, status: 'running', result: undefined, elapsedMs: null, error: undefined }
                  : item,
              ),
            activeMode,
          );

          const payloadToSend =
            activeMode === 'decide'
              ? buildClefPayload({ stateText: row.stateText, questions: clefQuestions })
              : buildClairicPayload({
                  stateText: row.stateText,
                  referenceDate,
                  includeQuantiles,
                  returnStateText,
                  fields,
                  attachments: row.attachments,
                });

          try {
            const snap = await runJobToCompletion(selectedModelForMode.id, payloadToSend, activeMode);
            if (unmountedRef.current) return;
            if (snap.status === 'done') {
              completedCount += 1;
              setBatchRows(
                (prev) =>
                  prev.map((item) =>
                    item.id === row.id
                      ? { ...item, status: 'done', elapsedMs: snap.elapsedMs, result: snap.result, error: undefined }
                      : item,
                  ),
                activeMode,
              );
            } else if (snap.status === 'cancelled') {
              stoppedEarly = true;
              setBatchRows(
                (prev) =>
                  prev.map((item) =>
                    item.id === row.id
                      ? { ...item, status: 'cancelled', result: undefined, elapsedMs: snap.elapsedMs }
                      : item,
                  ),
                activeMode,
              );
              break;
            } else {
              failedCount += 1;
              setBatchRows(
                (prev) =>
                  prev.map((item) =>
                    item.id === row.id
                      ? {
                          ...item,
                          status: 'error',
                          result: undefined,
                          elapsedMs: snap.elapsedMs,
                          error: snap.error ?? 'Run failed',
                        }
                      : item,
                  ),
                activeMode,
              );
            }
          } catch (err) {
            if (unmountedRef.current) return;
            failedCount += 1;
            setBatchRows(
              (prev) =>
                prev.map((item) =>
                  item.id === row.id
                    ? { ...item, status: 'error', result: undefined, elapsedMs: null, error: errMessage(err) }
                    : item,
                ),
              activeMode,
            );
          }
        }

        if (unmountedRef.current) return;
        if (stoppedEarly || batchCancelledRef.current) {
          setBatchRows(
            (prev) => prev.map((item) => (item.status === 'queued' ? { ...item, status: 'idle' } : item)),
            activeMode,
          );
          toast.info(`Batch stopped (${String(completedCount)} completed)`);
        } else if (snapshotRows.length > 1) {
          if (failedCount > 0) {
            toast.info(`Batch finished: ${String(completedCount)} done, ${String(failedCount)} failed`);
          } else {
            toast.success(`Batch complete: ${String(completedCount)} ${completedCount === 1 ? 'case' : 'cases'}`);
          }
        }
      } finally {
        if (!unmountedRef.current) {
          setBatchRunning(false);
          setBatchRunningRowId(null);
          setCancelling(false);
          stateQuery.reload();
        }
      }
    },
    [
      isRunning,
      selectedModelForMode,
      mode,
      sidecarOffline,
      hasValidationErrors,
      batchRows,
      setBatchRows,
      clefQuestions,
      referenceDate,
      includeQuantiles,
      returnStateText,
      fields,
      runJobToCompletion,
      stateQuery.reload,
    ],
  );

  const handleRunBatchAll = useCallback((): void => {
    void executeBatchQueue(batchRows.map((r) => r.id));
  }, [executeBatchQueue, batchRows]);

  const handleRunBatchRow = useCallback(
    (rowId: string): void => {
      void executeBatchQueue([rowId]);
    },
    [executeBatchQueue],
  );

  const handleRun = useCallback(async (): Promise<void> => {
    if (isRunning) return;
    if (workspaceMode === 'batch') {
      handleRunBatchAll();
      return;
    }
    if (selectedModelForMode === undefined) {
      setModelsDrawerOpen(true);
      toast.info(`Install a ${mode === 'decide' ? 'CLEF' : 'Clairic'} model first`);
      return;
    }
    if (sidecarOffline) {
      toast.info('Inference engine is stopped', {
        description: 'Start it from the Engine page or menu bar before running.',
      });
      return;
    }
    if (rawJsonError !== null) {
      toast.error('Fix JSON syntax errors before running');
      return;
    }
    if (leftView !== 'json' && mode === 'extract' && fields.length === 0) {
      if (stateText.trim() === '' && attachments.length === 0) {
        toast.info('Paste text or attach an image/PDF first');
        return;
      }
      await startAutoDetect({ thenRun: true });
      return;
    }
    if (leftView !== 'json' && hasValidationErrors) {
      setFocusInvalidNonce((n) => n + 1);
      toast.error('Fix highlighted field warnings before running');
      return;
    }

    let payloadToSend: Record<string, unknown> = structuredPayload;
    if (rawJsonDraft !== null) {
      try {
        const parsed: unknown = JSON.parse(rawJsonDraft);
        if (isPlainObject(parsed)) payloadToSend = parsed;
      } catch {
        toast.error('Invalid JSON payload');
        return;
      }
    }

    await launchRun(selectedModelForMode.id, payloadToSend, mode, attachments);
  }, [
    isRunning,
    workspaceMode,
    handleRunBatchAll,
    selectedModelForMode,
    mode,
    sidecarOffline,
    rawJsonError,
    leftView,
    fields.length,
    stateText,
    attachments,
    startAutoDetect,
    hasValidationErrors,
    structuredPayload,
    rawJsonDraft,
    launchRun,
  ]);

  const handleCancel = useCallback(async (): Promise<void> => {
    if (cancelling) return;
    if (batchRunning) {
      batchCancelledRef.current = true;
      setCancelling(true);
      const activeId = batchActiveJobIdRef.current;
      if (activeId !== null) {
        try {
          await mutate<{ ok: true }>('POST', `/systemone/jobs/${encodeURIComponent(activeId)}/cancel`);
        } catch (err) {
          toast.error('Could not stop active case', { description: errMessage(err) });
          setCancelling(false);
        }
      }
      return;
    }
    if (activeJob === null) return;
    setCancelling(true);
    try {
      await mutate<{ ok: true }>('POST', `/systemone/jobs/${encodeURIComponent(activeJob.id)}/cancel`);
    } catch (err) {
      toast.error('Could not cancel run', { description: errMessage(err) });
      setCancelling(false);
    }
  }, [batchRunning, activeJob, cancelling]);

  const handleUnload = async (): Promise<void> => {
    if (unloading) return;
    setUnloading(true);
    try {
      const res = await mutate<{ unloaded: string[] }>('POST', '/systemone/unload');
      if (res.unloaded.length > 0) {
        toast.success('Unloaded System One models', { description: res.unloaded.join(', ') });
      } else {
        toast.info('No resident System One models to unload');
      }
      stateQuery.reload();
    } catch (err) {
      toast.error('Unload failed', { description: errMessage(err) });
    } finally {
      setUnloading(false);
    }
  };

  const handleClearHistory = async (): Promise<void> => {
    if (clearingHistory) return;
    setClearingHistory(true);
    try {
      await mutate<{ ok: true }>('DELETE', '/systemone/history');
      toast.success('History cleared');
      stateQuery.reload();
    } catch (err) {
      toast.error('Could not clear history', { description: errMessage(err) });
    } finally {
      setClearingHistory(false);
    }
  };

  const handleRestoreHistory = (entry: SystemOneHistoryEntry): void => {
    setRawJsonDraft(null);
    setRawJsonError(null);
    setFocusedSpanField(null);
    setMode(entry.mode);
    setWorkspaceMode('single');
    const matching = installedModels.find((m) => m.id === entry.model || m.name === entry.model);
    if (matching !== undefined) setSelectedModelId(matching.id);

    let omittedCount = 0;
    if (entry.mode === 'decide') {
      const parsed = parseClefPayload(entry.request);
      setDecideStateText(parsed.stateText);
      setClefQuestions(parsed.questions);
      setAttachments([]);
      setLastSubmittedAttachments([]);
    } else {
      const parsed = parseClairicPayload(entry.request);
      const restored = restoreHistoryMedia(entry.request);
      omittedCount = restored.omittedCount;
      setExtractStateText(parsed.stateText);
      setReferenceDate(parsed.referenceDate);
      setIncludeQuantiles(parsed.includeQuantiles);
      setReturnStateText(parsed.returnStateText);
      setFields(parsed.fields);
      setAttachments(restored.attachments);
      setLastSubmittedAttachments(restored.attachments);
    }

    setLastRequestPayload(entry.request);
    setRequestMode(entry.mode);
    setResult(entry.result);
    setResultMode(entry.mode);
    setResultRun(entry.result !== undefined ? { model: entry.model, at: entry.createdAt } : null);
    setLastElapsedMs(entry.elapsedMs);
    const restoredError = entry.error;
    setRunFailure((prev) =>
      restoredError !== undefined
        ? { seq: (prev?.seq ?? 0) + 1, kind: 'run', title: 'Run failed', text: restoredError, mode: entry.mode }
        : null,
    );

    if (omittedCount > 0) {
      toast.info('Restored run (media omitted)', {
        description: `${String(omittedCount)} large attachment${omittedCount === 1 ? '' : 's'} not saved in history — re-attach before running again.`,
      });
    } else {
      toast.success('Restored run from history');
    }
  };

  const handleDownloadJson = (): void => {
    if (resultMode !== mode || result === undefined) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `system-one-${selectedModelForMode?.name ?? 'result'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Hotkey: ⌘↵ to Run
  const hotkeys = useMemo<Hotkey[]>(
    () => [
      {
        key: 'Enter',
        meta: true,
        allowInInput: true,
        run: () => {
          if (!isRunning) void handleRun();
        },
      },
    ],
    [handleRun, isRunning],
  );
  useHotkeys(hotkeys);

  // Command palette actions
  const commands = useMemo<AppCommand[]>(
    () => [
      {
        id: 'systemone:run',
        label: workspaceMode === 'batch' ? 'Run System One batch' : 'Run System One',
        group: 'System One',
        shortcut: '⌘ ↵',
        run: () => void handleRun(),
      },
      {
        id: 'systemone:toggle-batch',
        label: workspaceMode === 'batch' ? 'Switch to Single mode' : 'Switch to Batch eval mode',
        group: 'System One',
        run: () => setWorkspaceMode((prev) => (prev === 'single' ? 'batch' : 'single')),
      },
      {
        id: 'systemone:api-guide',
        label: 'Open System One API guide',
        group: 'System One',
        run: () => setApiGuideOpen(true),
      },
      {
        id: 'systemone:models',
        label: 'Download System One models',
        group: 'System One',
        run: () => setModelsDrawerOpen(true),
      },
      {
        id: 'systemone:history',
        label: 'Open System One history & compare',
        group: 'System One',
        run: () => setHistoryDrawerOpen(true),
      },
      ...SYSTEM_ONE_PRESETS.map((preset) => ({
        id: `systemone:preset:${preset.id}`,
        label: `Example: ${preset.label}`,
        description: preset.description,
        group: 'System One',
        run: () => applyPreset(preset),
      })),
    ],
    [handleRun, applyPreset, workspaceMode],
  );
  useCommands(commands);

  // Derived result summaries, from the run of the mode on screen.
  const shownResult = resultMode === mode ? result : undefined;
  const shownRequestPayload = requestMode === mode ? lastRequestPayload : null;
  const summary = useMemo(() => summarizeResult(shownResult), [shownResult]);
  const routingSummary = useMemo(() => summarizeRouting(shownResult, policy), [shownResult, policy]);
  const latencyParts = formatDurationParts(lastElapsedMs);

  const requestFields = useMemo(() => {
    const src = shownRequestPayload ?? structuredPayload;
    return isPlainObject(src.fields) ? (src.fields as Record<string, unknown>) : {};
  }, [shownRequestPayload, structuredPayload]);

  /** The fields or questions of the request in flight: the Output lists them while their answers load. */
  const pendingNames = useMemo(() => {
    const src = shownRequestPayload ?? structuredPayload;
    if (isPlainObject(src.fields)) return Object.keys(src.fields);
    return isPlainObject(src.questions) ? Object.keys(src.questions) : [];
  }, [shownRequestPayload, structuredPayload]);

  const evidenceText = useMemo(() => {
    if (shownResult !== undefined && typeof shownResult.state_text === 'string') return shownResult.state_text;
    const req = shownRequestPayload ?? structuredPayload;
    if (typeof req.state === 'string') return req.state;
    return stateText;
  }, [shownResult, shownRequestPayload, structuredPayload, stateText]);

  const spanEntries = useMemo<HighlightSpanEntry[]>(() => {
    if (shownResult === undefined || !isPlainObject(shownResult.answers)) return [];
    return Object.entries(shownResult.answers).map(([field, raw], index) => ({
      field,
      color: fieldAccentColor(index),
      confidence: isPlainObject(raw) && typeof raw.confidence === 'number' ? raw.confidence : undefined,
      spans: extractAnswerSpans(raw),
    }));
  }, [shownResult]);

  const liveServeProcess = useMemo(
    () => live?.processes.find((p) => p.status === 'running' && p.host !== 'desktop') ?? null,
    [live],
  );
  const liveServePort = liveServeProcess !== null ? liveServeProcess.port : null;
  const apiEndpointUrl =
    liveServePort !== null ? `http://127.0.0.1:${String(liveServePort)}/v1/systemone` : DEFAULT_SYSTEMONE_ENDPOINT;

  const resultJson = useMemo(
    () => (shownResult === undefined ? '' : JSON.stringify(shownResult, null, 2)),
    [shownResult],
  );
  const activeModelSlug = selectedModelForMode?.id ?? (mode === 'decide' ? 'clef-flash' : 'clairic-flash');
  const activeModelPath =
    selectedModelForMode?.path ??
    (mode === 'decide'
      ? '~/.mlx-node/models/clef-flash'
      : '~/.mlx-node/models/clairic-flash-gguf/Clairic-Flash-UD-Q4_K_XL.gguf');
  const curlSnippet = useMemo(
    () => buildCurlCommand(activeModelSlug, structuredPayload, apiEndpointUrl),
    [activeModelSlug, structuredPayload, apiEndpointUrl],
  );
  const tsSnippet = useMemo(
    () =>
      buildTypeScriptSnippet({
        model: activeModelSlug,
        mode,
        payload: structuredPayload,
        policy,
        endpointUrl: apiEndpointUrl,
      }),
    [activeModelSlug, mode, structuredPayload, policy, apiEndpointUrl],
  );
  const pythonSnippet = useMemo(
    () =>
      buildPythonSnippet({
        model: activeModelSlug,
        mode,
        payload: structuredPayload,
        policy,
        endpointUrl: apiEndpointUrl,
      }),
    [activeModelSlug, mode, structuredPayload, policy, apiEndpointUrl],
  );
  const sdkSnippet = useMemo(
    () =>
      buildNodeSdkSnippet({
        modelPath: activeModelPath,
        mode,
        payload: structuredPayload,
        policy,
      }),
    [activeModelPath, mode, structuredPayload, policy],
  );

  const handleJumpToSpan = (fieldName: string): void => {
    setFocusedSpanField(fieldName);
    setRightTab('evidence');
  };

  const missingModelForMode = installedModels.length > 0 && modelsForMode.length === 0;
  const showEmptyInstallCard = stateQuery.data !== undefined && installedModels.length === 0;
  const runnableBatchCount = batchRows.filter(
    (r) => r.stateText.trim() !== '' || (mode === 'extract' && r.attachments.length > 0),
  ).length;
  const runDisabled =
    selectedModelForMode === undefined || sidecarOffline || (workspaceMode === 'batch' && runnableBatchCount === 0);
  const runTooltip = sidecarOffline
    ? 'Inference engine is stopped. Start it from the Engine page or menu bar.'
    : installedModels.length === 0
      ? 'Install a System One model first'
      : missingModelForMode
        ? `No ${mode === 'decide' ? 'CLEF decision' : 'Clairic extraction'} model installed`
        : workspaceMode === 'batch' && runnableBatchCount === 0
          ? 'Add at least one case to run batch'
          : 'Run (⌘↵)';

  // A run started in one mode keeps its own wording and stays out of the other mode's Output.
  const runMode = activeJob?.mode ?? mode;
  const runModelName = activeJob?.model ?? selectedModelForMode?.name ?? 'model';
  const engineLoading = live?.processes.some((p) => p.status === 'running' && p.loading.length > 0) ?? false;
  const runPending = isRunning && !batchRunning && !autoDetecting && runMode === mode;
  const shownFailure = runFailure?.mode === mode ? runFailure : null;
  const runLineRest =
    shownResult === undefined
      ? 'No run yet'
      : resultRun === null
        ? 'Last run'
        : `Last run · ${resultRun.model} · ${formatClock(resultRun.at)}`;
  // CLEF answers are probabilities with no text spans, so Decide has no Highlights tab.
  const outputTab = mode === 'decide' && rightTab === 'evidence' ? 'answers' : rightTab;

  const optionsMeta = [
    referenceDate.trim() !== '' ? referenceDate.trim() : null,
    includeQuantiles ? 'Uncertainty curve' : null,
    returnStateText ? 'Highlights' : null,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');

  return (
    <Page busy={stateQuery.refreshing}>
      <PageHeader
        title="System One"
        description={
          mode === 'decide'
            ? 'Score situations with calibrated Yes/No, choice, and rubric probabilities.'
            : 'Pull facts, numbers, and forecasts from any text, image, or PDF.'
        }
        controls={
          <div className={styles.headerControls}>
            <ModelPicker
              value={pickerDisplayModel?.id ?? null}
              bases={modelBases}
              side="bottom"
              align="end"
              onValueChange={(id) => {
                setSelectedModelId(id);
                const picked = installedModels.find((m) => m.id === id);
                if (picked !== undefined) {
                  setMode(picked.modelType === 'clef' ? 'decide' : 'extract');
                }
                setRawJsonDraft(null);
                setRawJsonError(null);
              }}
              disabled={installedModels.length === 0}
              working={isRunning}
              memory={
                residentModel !== undefined
                  ? {
                      text: `${residentModel.name} loaded`,
                      action: (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          loading={unloading}
                          onClick={() => void handleUnload()}
                        >
                          Unload
                        </Button>
                      ),
                    }
                  : undefined
              }
            />

            <SegmentedControl
              label="System One mode"
              options={[
                { value: 'extract', label: 'Extract' },
                { value: 'decide', label: 'Decide' },
              ]}
              value={mode}
              onValueChange={(next) => {
                setMode(next as SystemOneMode);
                setRawJsonDraft(null);
                setRawJsonError(null);
              }}
            />

            <SegmentedControl
              label="Workspace mode"
              options={[
                { value: 'single', label: 'Single' },
                { value: 'batch', label: 'Batch' },
              ]}
              value={workspaceMode}
              onValueChange={(next) => setWorkspaceMode(next as WorkspaceMode)}
            />

            <SpecsPopover
              open={specsOpen}
              onOpenChange={setSpecsOpen}
              mode={mode}
              presets={SYSTEM_ONE_PRESETS}
              savedSpecs={savedSpecs}
              canSaveCurrent={canSaveCurrentSpec}
              onPickPreset={applyPreset}
              onApplySavedSpec={handleApplySavedSpec}
              onSaveCurrent={handleSaveCurrentSpec}
              onDeleteSavedSpec={handleDeleteSavedSpec}
            />
          </div>
        }
        actions={
          <div className={styles.headerActions}>
            <Tooltip content="History & compare">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                aria-label={
                  history.length > 0 ? `History & compare, ${String(history.length)} runs` : 'History & compare'
                }
                onClick={() => setHistoryDrawerOpen(true)}
              >
                <History size={14} strokeWidth={1.75} aria-hidden="true" />
                {history.length > 0 ? (
                  <span className={styles.historyCount}>{history.length > 99 ? '99+' : history.length}</span>
                ) : null}
              </Button>
            </Tooltip>

            <Button type="button" variant="ghost" size="sm" onClick={() => setModelsDrawerOpen(true)}>
              <Download size={14} strokeWidth={1.75} aria-hidden="true" />
              Models
            </Button>

            {isRunning ? (
              <span className={styles.runSlot}>
                <Button
                  type="button"
                  variant="danger"
                  size="sm"
                  loading={cancelling}
                  onClick={() => void handleCancel()}
                >
                  <Square size={12} strokeWidth={2} aria-hidden="true" />
                  {batchRunning ? 'Stop' : 'Cancel'}
                </Button>
              </span>
            ) : (
              <Tooltip content={runTooltip}>
                <span className={styles.runSlot}>
                  <Button
                    type="button"
                    variant="primary"
                    size="sm"
                    disabled={runDisabled}
                    onClick={() => void handleRun()}
                  >
                    <Play size={12} strokeWidth={2} aria-hidden="true" />
                    {workspaceMode === 'batch' && runnableBatchCount > 0 ? `Run ${String(runnableBatchCount)}` : 'Run'}
                  </Button>
                </span>
              </Tooltip>
            )}
          </div>
        }
      />

      {stateQuery.error !== undefined ? (
        <Alert tone="warning" title="Could not reach System One runtime">
          <span className="grid justify-items-start gap-2">
            <span>{stateQuery.error.message}</span>
            <Button variant="secondary" size="sm" onClick={stateQuery.reload}>
              Try again
            </Button>
          </span>
        </Alert>
      ) : null}

      {sidecarOffline && installedModels.length > 0 ? (
        <div className={styles.offlineStrip} role="status">
          <span>Inference engine is stopped. Start it from the menu bar or the Engine page to run queries.</span>
          <Link className="link" to="/engine">
            Open Engine
          </Link>
        </div>
      ) : null}

      {missingModelForMode ? (
        <div className={styles.offlineStrip} role="status">
          <span>
            {mode === 'decide'
              ? 'No CLEF decision model installed yet. You can still design rubrics, edit thresholds, and import test batches — install a CLEF model to run.'
              : 'No Clairic extraction model installed yet. Install a Clairic model from Models to run extractions.'}
          </span>
          <Button type="button" variant="ghost" size="sm" onClick={() => setModelsDrawerOpen(true)}>
            Open Models
          </Button>
        </div>
      ) : null}

      {showEmptyInstallCard ? (
        <NoSystemOneModelsCard catalog={catalogQuery} host={hostQuery} downloads={downloads} />
      ) : null}

      <div className={styles.apiStrip} role="region" aria-label="Local System One API endpoint">
        <div className={styles.apiStripLeft}>
          <Badge size="sm" color="sky">
            POST
          </Badge>
          <code className={styles.apiStripUrl}>{apiEndpointUrl}</code>
          <span className={styles.apiStripMeta}>
            {liveServePort !== null ? (
              <>
                <span className={styles.apiLiveDot} aria-hidden="true" />
                <span>{`mlx serve live on :${String(liveServePort)}`}</span>
              </>
            ) : (
              <span>
                Run <code>mlx serve</code> in terminal to call from your app
              </span>
            )}
          </span>
        </div>
        <div className={styles.apiStripActions}>
          <CopyButton
            value={liveServePort !== null ? apiEndpointUrl : 'mlx serve --port 8080'}
            label={liveServePort !== null ? 'Copy URL' : 'Copy mlx serve'}
          />
          <Button type="button" variant="ghost" size="sm" onClick={() => setApiGuideOpen(true)}>
            <Code2 size={13} strokeWidth={1.75} aria-hidden="true" />
            API Guide
          </Button>
        </div>
      </div>

      {workspaceMode === 'batch' ? (
        <BatchView
          mode={mode}
          modelName={selectedModelForMode?.name ?? null}
          fields={fields}
          clefQuestions={clefQuestions}
          fieldWarnings={fieldWarnings}
          clefWarnings={clefWarnings}
          policy={policy}
          onPolicyChange={setPolicy}
          rows={batchRows}
          onRowsChange={(next) => setBatchRows(next, mode)}
          primaryColumn={primaryColumn}
          onPrimaryColumnChange={(col) => setPrimaryColumn(col, mode)}
          running={batchRunning}
          runningRowId={batchRunningRowId}
          canRun={!runDisabled && !isRunning}
          runTooltip={runTooltip}
          onRunAll={handleRunBatchAll}
          onRunRow={handleRunBatchRow}
          onCancel={() => void handleCancel()}
          onLoadSample={handleLoadBatchSample}
          onFieldsChange={(next) => {
            setRawJsonDraft(null);
            setFields(next);
          }}
          onClefQuestionsChange={(next) => {
            setRawJsonDraft(null);
            setClefQuestions(next);
          }}
          onSaveSpec={() => setSpecsOpen(true)}
        />
      ) : (
        <>
          {/* Always mounted: a row that comes and goes with a run moves the two cards below it. */}
          <StatGroup label="System One run metrics" maxColumns={4}>
            <StatTile
              label="Latency"
              value={shownResult === undefined ? null : latencyParts.value}
              unit={shownResult === undefined ? undefined : latencyParts.unit}
              countUp={false}
              held={runPending}
            />
            <StatTile
              label="Routing"
              value={
                shownResult === undefined
                  ? null
                  : routingSummary.totalCount > 0
                    ? `${formatCount(routingSummary.autoCount)} / ${formatCount(routingSummary.totalCount)}`
                    : formatCount(summary.extractedCount + summary.nullCount)
              }
              hint={
                shownResult === undefined
                  ? 'No answers yet'
                  : [
                      `${formatCount(routingSummary.autoCount)} auto`,
                      routingSummary.reviewCount > 0 ? `${formatCount(routingSummary.reviewCount)} review` : null,
                      routingSummary.missingCount > 0 ? `${formatCount(routingSummary.missingCount)} missing` : null,
                      summary.truncated ? 'truncated' : null,
                    ]
                      .filter((part): part is string => part !== null)
                      .join(' · ')
              }
              countUp={false}
              held={runPending}
            />
            <StatTile
              label="Tokens"
              value={
                summary.inputTokens === null && summary.outputTokens === null
                  ? null
                  : formatCount((summary.inputTokens ?? 0) + (summary.outputTokens ?? 0))
              }
              hint={[
                summary.inputTokens !== null ? `${formatCount(summary.inputTokens)} in` : null,
                summary.outputTokens !== null ? `${formatCount(summary.outputTokens)} out` : null,
                summary.imageTokens !== null && summary.imageTokens > 0
                  ? `${formatCount(summary.imageTokens)} img`
                  : null,
              ]
                .filter((part): part is string => part !== null)
                .join(' · ')}
              countUp={false}
              held={runPending}
            />
            <StatTile
              label="Confidence"
              value={summary.meanConfidence === null ? null : formatPercent(summary.meanConfidence)}
              footer={
                summary.meanConfidence !== null ? (
                  <Meter
                    label="Confidence"
                    value={summary.meanConfidence * 100}
                    max={100}
                    unit="%"
                    warnAt={1.01}
                    dangerAt={1.01}
                  />
                ) : undefined
              }
              countUp={false}
              held={runPending}
            />
          </StatGroup>

          <div className={styles.workspace}>
            <Panel
              title="Input"
              className={styles.panePanel}
              actions={
                <SegmentedControl
                  label="Input view"
                  options={[
                    { value: 'visual', label: 'Visual' },
                    { value: 'json', label: 'JSON' },
                  ]}
                  value={leftView}
                  onValueChange={(next) => {
                    if (next === 'visual' && rawJsonError === null) setRawJsonDraft(null);
                    setLeftView(next as 'visual' | 'json');
                  }}
                />
              }
            >
              {leftView === 'visual' ? (
                <div className={styles.visualStack}>
                  <div className={styles.sourceSection}>
                    <div className={styles.sourceHead}>
                      <div className={styles.sectionHeadingGroup}>
                        <label htmlFor="system-one-state" className={styles.sectionLabel}>
                          {mode === 'decide' ? '1. Context' : '1. Document'}
                        </label>
                        <span className={styles.sectionHint}>
                          {mode === 'decide'
                            ? 'Paste text or JSON describing the case to evaluate. Swap inputs anytime without losing your rubric.'
                            : 'Type text, drop or paste an image/PDF, or both — each is optional.'}
                        </span>
                      </div>
                    </div>

                    <MediaDropzone
                      textId="system-one-state"
                      text={stateText}
                      onTextChange={(next) => {
                        setRawJsonDraft(null);
                        setStateText(next);
                      }}
                      attachments={attachments}
                      allowMedia={mode === 'extract'}
                      disabled={isRunning}
                      placeholder={
                        mode === 'decide'
                          ? 'Paste JSON or plain text context to evaluate…'
                          : 'Paste text, or drop / paste an image or PDF here…'
                      }
                      onChange={(next) => {
                        setRawJsonDraft(null);
                        setAttachments(next);
                      }}
                      onClear={() => {
                        setRawJsonDraft(null);
                        setStateText('');
                        if (mode === 'extract') setAttachments([]);
                      }}
                    />

                    {mode === 'extract' ? (
                      <Disclosure
                        title="Options"
                        meta={optionsMeta || undefined}
                        defaultOpen={false}
                        className={styles.optionsDisclosure}
                      >
                        <div className={styles.optionsBody}>
                          <Input
                            label="Reference date"
                            placeholder="YYYY-MM-DD"
                            description="Anchor date used to resolve relative phrases like 'next Tuesday'."
                            value={referenceDate}
                            disabled={isRunning}
                            onChange={(event) => {
                              setRawJsonDraft(null);
                              setReferenceDate(event.target.value);
                            }}
                          />
                          <div className={styles.switchRow}>
                            <Switch
                              label="Include uncertainty curve (P10–P90)"
                              checked={includeQuantiles}
                              disabled={isRunning}
                              onCheckedChange={(checked) => {
                                setRawJsonDraft(null);
                                setIncludeQuantiles(checked);
                              }}
                            />
                            <Switch
                              label="Keep text for highlights"
                              checked={returnStateText}
                              disabled={isRunning}
                              onCheckedChange={(checked) => {
                                setRawJsonDraft(null);
                                setReturnStateText(checked);
                              }}
                            />
                          </div>
                        </div>
                      </Disclosure>
                    ) : null}
                  </div>

                  <div className={styles.sectionDivider} />

                  {mode === 'decide' ? (
                    <ClefBuilder
                      questions={clefQuestions}
                      warnings={clefWarnings}
                      disabled={isRunning}
                      focusInvalidNonce={focusInvalidNonce}
                      onSaveSpec={() => setSpecsOpen(true)}
                      onChange={(next) => {
                        setRawJsonDraft(null);
                        setClefQuestions(next);
                      }}
                    />
                  ) : (
                    <SchemaBuilder
                      fields={fields}
                      warnings={fieldWarnings}
                      disabled={isRunning}
                      focusInvalidNonce={focusInvalidNonce}
                      onSaveSpec={() => setSpecsOpen(true)}
                      onAutoDetect={() => void startAutoDetect({ thenRun: false })}
                      autoDetecting={autoDetecting}
                      canAutoDetect={
                        !isRunning &&
                        !sidecarOffline &&
                        selectedModelForMode?.modelType === 'clairic' &&
                        (stateText.trim() !== '' || attachments.length > 0)
                      }
                      onChange={(next) => {
                        setRawJsonDraft(null);
                        setFields(next);
                      }}
                    />
                  )}
                </div>
              ) : (
                <div className={styles.jsonStack}>
                  <div className={styles.subToolbar}>
                    <span className={styles.subToolbarHint}>Synced with visual editor</span>
                    <div className={styles.subToolbarActions}>
                      <CopyButton value={tsSnippet} label="Copy TypeScript" />
                      <CopyButton value={curlSnippet} label="Copy cURL" />
                    </div>
                  </div>
                  {rawJsonError !== null ? (
                    <div className={styles.rawErrorStrip} role="alert">
                      {`Invalid JSON: ${rawJsonError}`}
                    </div>
                  ) : null}
                  <textarea
                    aria-label="Raw JSON request payload"
                    className={styles.rawJsonFill}
                    value={rawJsonValue}
                    disabled={isRunning}
                    spellCheck={false}
                    onChange={(event) => applyRawJson(event.target.value)}
                  />
                </div>
              )}
            </Panel>

            <Panel
              title="Output"
              className={styles.panePanel}
              actions={
                <SegmentedControl
                  label="Output view"
                  options={
                    mode === 'decide'
                      ? [
                          { value: 'answers', label: 'Answers' },
                          { value: 'code', label: 'Code & JSON' },
                        ]
                      : [
                          { value: 'answers', label: 'Answers' },
                          { value: 'evidence', label: 'Highlights' },
                          { value: 'code', label: 'Code & JSON' },
                        ]
                  }
                  value={outputTab}
                  onValueChange={(next) => {
                    const tab = next as 'answers' | 'evidence' | 'code';
                    setRightTab(tab);
                    if (tab === 'code' && shownResult === undefined && codeFormat === 'json') {
                      setCodeFormat('ts');
                    }
                  }}
                />
              }
            >
              <div className={styles.paneContent}>
                {/* One box with three faces: rest, running, failed. It is always mounted, and the face of the last run
                    stays in the cell during a run, so the answers below do not move. */}
                <div ref={runLineRef} className={styles.runLine}>
                  {shownFailure !== null ? (
                    <div
                      key={shownFailure.seq}
                      className={styles.runLineFailed}
                      role={isRunning ? undefined : 'alert'}
                      aria-hidden={isRunning ? true : undefined}
                      data-leaving={isRunning ? '' : undefined}
                    >
                      <CircleX size={16} strokeWidth={1.75} aria-hidden="true" />
                      <p title={shownFailure.text}>
                        <strong>{shownFailure.title}</strong> {shownFailure.text}
                      </p>
                    </div>
                  ) : (
                    <p
                      className={styles.runLineRest}
                      aria-hidden={isRunning ? true : undefined}
                      data-leaving={isRunning ? '' : undefined}
                    >
                      {runLineRest}
                    </p>
                  )}
                  {isRunning ? (
                    <div className={styles.runLineWait} role="status" aria-live="polite">
                      <WaitLine
                        orb={engineLoading ? 'load' : 'think'}
                        text={
                          engineLoading
                            ? `Loading ${runModelName}…`
                            : autoDetecting
                              ? `Detecting fields with ${runModelName}…`
                              : `Running ${runMode === 'decide' ? 'decision' : 'extraction'} on ${runModelName}…`
                        }
                      />
                      {activeJob !== null ? <WaitClock since={activeJob.startedAt} /> : null}
                    </div>
                  ) : null}
                </div>

                {outputTab === 'answers' ? (
                  <ResultsView
                    result={shownResult}
                    pending={runPending}
                    failed={shownFailure?.kind === 'run'}
                    pendingNames={pendingNames}
                    requestFields={requestFields}
                    mode={mode}
                    presets={SYSTEM_ONE_PRESETS}
                    canRun={
                      !runDisabled &&
                      !isRunning &&
                      (stateText.trim() !== '' || (mode === 'extract' && attachments.length > 0))
                    }
                    policy={policy}
                    onPolicyChange={setPolicy}
                    onRun={() => void handleRun()}
                    onPickPreset={applyPreset}
                    onJumpToSpan={handleJumpToSpan}
                    onOpenApiGuide={() => setApiGuideOpen(true)}
                    apiEndpointUrl={apiEndpointUrl}
                  />
                ) : outputTab === 'evidence' ? (
                  <SpanHighlighter
                    text={evidenceText}
                    entries={spanEntries}
                    hasResult={shownResult !== undefined}
                    focusedField={focusedSpanField}
                    onFocusField={setFocusedSpanField}
                    attachments={requestMode === mode ? lastSubmittedAttachments : []}
                  />
                ) : (
                  <div className={styles.jsonStack}>
                    <div className={styles.subToolbar}>
                      <SegmentedControl
                        label="Code format"
                        options={[
                          { value: 'json', label: 'JSON' },
                          { value: 'ts', label: 'TypeScript' },
                          { value: 'python', label: 'Python' },
                          { value: 'curl', label: 'cURL' },
                          { value: 'sdk', label: 'Node SDK' },
                        ]}
                        value={codeFormat}
                        onValueChange={(next) => setCodeFormat(next as CodeFormat)}
                      />
                      <div className={styles.subToolbarActions}>
                        {codeFormat === 'json' ? (
                          shownResult !== undefined ? (
                            <>
                              <CopyButton value={resultJson} label="Copy JSON" />
                              <Button type="button" variant="secondary" size="sm" onClick={handleDownloadJson}>
                                <FileDown size={14} strokeWidth={1.75} aria-hidden="true" />
                                Download
                              </Button>
                            </>
                          ) : null
                        ) : codeFormat === 'ts' ? (
                          <CopyButton value={tsSnippet} label="Copy TypeScript" />
                        ) : codeFormat === 'python' ? (
                          <CopyButton value={pythonSnippet} label="Copy Python" />
                        ) : codeFormat === 'curl' ? (
                          <CopyButton value={curlSnippet} label="Copy cURL" />
                        ) : (
                          <CopyButton value={sdkSnippet} label="Copy SDK" />
                        )}
                      </div>
                    </div>

                    {codeFormat === 'json' ? (
                      shownResult === undefined ? (
                        <div className={styles.emptyCenter}>
                          <EmptyState
                            title="No response JSON yet"
                            description="Run a query to inspect the raw response, or switch to TypeScript, Python, cURL, or Node SDK above to copy threshold-ready API code."
                          />
                        </div>
                      ) : (
                        <CodePanel code={resultJson} label="System One JSON response" wrap />
                      )
                    ) : codeFormat === 'ts' ? (
                      <CodePanel code={tsSnippet} label="TypeScript threshold routing snippet" wrap />
                    ) : codeFormat === 'python' ? (
                      <CodePanel code={pythonSnippet} label="Python threshold routing snippet" wrap />
                    ) : codeFormat === 'curl' ? (
                      <CodePanel code={curlSnippet} label="cURL request snippet" wrap />
                    ) : (
                      <CodePanel code={sdkSnippet} label="@mlx-node/lm in-process SDK snippet" wrap />
                    )}
                  </div>
                )}
              </div>
            </Panel>
          </div>
        </>
      )}

      <SystemOneModelsDrawer
        open={modelsDrawerOpen}
        onOpenChange={setModelsDrawerOpen}
        catalog={catalogQuery}
        host={hostQuery}
        updates={updatesQuery}
        downloads={downloads}
      />

      <HistoryDrawer
        open={historyDrawerOpen}
        onOpenChange={setHistoryDrawerOpen}
        history={history}
        clearing={clearingHistory}
        onClear={() => void handleClearHistory()}
        onRestore={handleRestoreHistory}
      />

      <ApiGuideDrawer
        open={apiGuideOpen}
        onOpenChange={setApiGuideOpen}
        mode={mode}
        modelId={activeModelSlug}
        endpointUrl={apiEndpointUrl}
        liveServePort={liveServePort}
        tsSnippet={tsSnippet}
        pythonSnippet={pythonSnippet}
        curlSnippet={curlSnippet}
        sdkSnippet={sdkSnippet}
      />
    </Page>
  );
}

import { AlertTriangle, ArrowRight, ArrowUpRight, Code2, Play } from 'lucide-react';
import type { CSSProperties, ReactElement } from 'react';

import type { ChartPoint } from '../../components/arc-x/chart/types';
import { KeyValue, type KeyValueItem } from '../../components/arc-x/key-value/key-value';
import { RangeBar } from '../../components/arc-x/range-bar/range-bar';
import { Meter } from '../../components/arc-x/split-bar/split-bar';
import { TrendChart } from '../../components/arc-x/trend-chart/trend-chart';
import { Badge, type BadgeTone } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { Tooltip } from '../../components/arc/tooltip/tooltip';
import { formatPercent } from '../../lib/format';
import {
  type ClairicForecastAnswer,
  clampUnit,
  type ClefAnswer,
  type DecisionPolicy,
  evaluateClairicAnswerRoute,
  evaluateClefAnswerRoute,
  extractAnswerSpans,
  fieldAccentColor,
  isClefResult,
  isEstimateAnswer,
  isForecastAnswer,
  isListAnswer,
  isPlainObject,
  summarizeRouting,
  type SystemOneMode,
  type SystemOnePreset,
} from '../../lib/system-one';
import styles from './system-one.module.css';

export interface ResultsViewProps {
  result: Record<string, unknown> | undefined;
  /** Original request fields map so estimate/forecast can show configured min/max/unit/interval. */
  requestFields: Record<string, unknown>;
  mode: SystemOneMode;
  presets: readonly SystemOnePreset[];
  canRun: boolean;
  onRun: () => void;
  onPickPreset: (preset: SystemOnePreset) => void;
  onJumpToSpan?: (fieldName: string) => void;
  policy: DecisionPolicy;
  onPolicyChange: (next: DecisionPolicy) => void;
  hidePolicyBar?: boolean;
  /** Opens the API integration guide drawer. */
  onOpenApiGuide?: () => void;
  /** Local API endpoint URL shown in onboarding API tip. */
  apiEndpointUrl?: string;
}

function friendlyFieldType(rawType: string | undefined): string | undefined {
  if (rawType === undefined) return undefined;
  if (rawType === 'boolean') return 'yes / no';
  if (rawType === 'enum') return 'choice';
  if (rawType === 'datetime') return 'date & time';
  if (rawType === 'compute') return 'calculate';
  if (rawType === 'object') return 'nested';
  return rawType;
}

function confidenceTone(confidence: number): BadgeTone {
  if (confidence >= 0.9) return 'success';
  if (confidence >= 0.7) return 'warning';
  return 'danger';
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return JSON.stringify(value);
}

function toNumeric(value: number | string): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function ForecastDisplay({
  fieldName,
  answer,
  spec,
  color,
}: {
  fieldName: string;
  answer: ClairicForecastAnswer;
  spec: Record<string, unknown> | null;
  color: string;
}): ReactElement {
  const unit = typeof spec?.unit === 'string' ? spec.unit : '';
  const interval = typeof spec?.interval === 'string' ? spec.interval : 'step';
  const minVal = typeof spec?.min === 'number' || typeof spec?.min === 'string' ? spec.min : undefined;
  const maxVal = typeof spec?.max === 'number' || typeof spec?.max === 'string' ? spec.max : undefined;

  const chartData: ChartPoint[] = answer.median.map((med, index) => {
    const stepLabel = `${interval} ${String(index + 1)}`;
    const nMed = toNumeric(med);
    const nP10 = toNumeric(answer.p10[index] ?? med);
    const nP90 = toNumeric(answer.p90[index] ?? med);
    return {
      key: String(index),
      label: stepLabel,
      axisLabel: `+${String(index + 1)}`,
      values: {
        median: nMed,
        p10: nP10,
        p90: nP90,
      },
    };
  });
  const hasNumericChart = chartData.some((pt) => pt.values.median !== null);

  return (
    <div className={styles.forecastWrap}>
      {hasNumericChart ? (
        <TrendChart
          label={`${fieldName} forecast`}
          unit={unit || interval}
          height={148}
          data={chartData}
          series={[{ key: 'median', label: 'Median', color }]}
          band={{ lowKey: 'p10', highKey: 'p90' }}
          yMin="auto"
          curve="monotone"
        />
      ) : null}
      <div className={styles.flatTableWrap}>
        <table className={styles.flatTable}>
          <thead>
            <tr>
              <th>Step</th>
              {minVal !== undefined ? <th>Min</th> : null}
              <th>Low (P10)</th>
              <th>Expected</th>
              <th>High (P90)</th>
              {maxVal !== undefined ? <th>Max</th> : null}
            </tr>
          </thead>
          <tbody>
            {answer.median.map((med, index) => {
              const step = answer.steps[index];
              return (
                <tr key={index}>
                  <td>{`${interval} +${String(index + 1)}`}</td>
                  {minVal !== undefined ? <td className="text-muted">{formatCellValue(minVal)}</td> : null}
                  <td>{formatCellValue(step?.p10 ?? answer.p10[index])}</td>
                  <td className="font-medium text-foreground">{formatCellValue(step?.median ?? med)}</td>
                  <td>{formatCellValue(step?.p90 ?? answer.p90[index])}</td>
                  {maxVal !== undefined ? <td className="text-muted">{formatCellValue(maxVal)}</td> : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function rangeFillStyle(value: number, min: number, max: number): CSSProperties {
  const span = max - min;
  const pct = span > 0 ? Math.max(0, Math.min(100, ((value - min) / span) * 100)) : 0;
  return { '--range-pct': `${String(pct)}%` } as CSSProperties;
}

function PolicyStrip({
  result,
  policy,
  onPolicyChange,
}: {
  result: Record<string, unknown>;
  policy: DecisionPolicy;
  onPolicyChange: (next: DecisionPolicy) => void;
}): ReactElement {
  const routing = summarizeRouting(result, policy);
  const clef = isClefResult(result);
  const clefAnswers = clef && isPlainObject(result.answers) ? (Object.values(result.answers) as ClefAnswer[]) : [];
  const hasNoul = clefAnswers.some((ans) => ans.type === 'noul');
  const hasChoiceOrScore = clefAnswers.some((ans) => ans.type === 'choice' || ans.type === 'score');

  const noPct = Math.round(clampUnit(policy.noThreshold) * 100);
  const yesPct = Math.round(clampUnit(policy.yesThreshold) * 100);
  const minConfPct = Math.round(clampUnit(policy.minConfidence) * 100);

  return (
    <div className={styles.policyBar} role="region" aria-label="Decision routing policy">
      <div className={styles.policyTop}>
        <span className={styles.policyTitle}>Routing policy</span>
        <div className={styles.policyCounts}>
          <Badge size="sm" tone={routing.autoCount > 0 ? 'success' : 'neutral'}>
            {`${String(routing.autoCount)} Auto`}
          </Badge>
          <Badge size="sm" tone={routing.reviewCount > 0 ? 'warning' : 'neutral'}>
            {`${String(routing.reviewCount)} Review`}
          </Badge>
          {routing.missingCount > 0 ? (
            <Badge size="sm" tone="neutral" outline>
              {`${String(routing.missingCount)} Missing`}
            </Badge>
          ) : null}
        </div>
      </div>

      <div className={styles.policySliders}>
        {clef ? (
          <>
            {hasNoul ? (
              <>
                <label className={styles.policySliderGroup}>
                  <span className={styles.policySliderHead}>
                    <span>Auto-No</span>
                    <span className={styles.policySliderVal}>{`≤ ${String(noPct)}%`}</span>
                  </span>
                  <input
                    type="range"
                    className={styles.policyRange}
                    min={0}
                    max={95}
                    step={5}
                    value={noPct}
                    style={rangeFillStyle(noPct, 0, 95)}
                    aria-label="Auto-No threshold"
                    onChange={(event) => {
                      const nextNo = Math.min(Number(event.target.value) / 100, policy.yesThreshold);
                      onPolicyChange({ ...policy, noThreshold: nextNo });
                    }}
                  />
                </label>

                <label className={styles.policySliderGroup}>
                  <span className={styles.policySliderHead}>
                    <span>Auto-Yes</span>
                    <span className={styles.policySliderVal}>{`≥ ${String(yesPct)}%`}</span>
                  </span>
                  <input
                    type="range"
                    className={styles.policyRange}
                    min={5}
                    max={100}
                    step={5}
                    value={yesPct}
                    style={rangeFillStyle(yesPct, 5, 100)}
                    aria-label="Auto-Yes threshold"
                    onChange={(event) => {
                      const nextYes = Math.max(Number(event.target.value) / 100, policy.noThreshold);
                      onPolicyChange({ ...policy, yesThreshold: nextYes });
                    }}
                  />
                </label>
              </>
            ) : null}

            {hasChoiceOrScore ? (
              <label className={styles.policySliderGroup}>
                <span className={styles.policySliderHead}>
                  <span>Auto-route</span>
                  <span className={styles.policySliderVal}>{`≥ ${String(minConfPct)}%`}</span>
                </span>
                <input
                  type="range"
                  className={styles.policyRange}
                  min={50}
                  max={99}
                  step={1}
                  value={minConfPct}
                  style={rangeFillStyle(minConfPct, 50, 99)}
                  aria-label="Auto-route minimum confidence"
                  onChange={(event) => {
                    onPolicyChange({ ...policy, minConfidence: Number(event.target.value) / 100 });
                  }}
                />
              </label>
            ) : null}
          </>
        ) : (
          <label className={styles.policySliderGroup}>
            <span className={styles.policySliderHead}>
              <span>Auto-accept</span>
              <span className={styles.policySliderVal}>{`≥ ${String(minConfPct)}%`}</span>
            </span>
            <input
              type="range"
              className={styles.policyRange}
              min={50}
              max={99}
              step={1}
              value={minConfPct}
              style={rangeFillStyle(minConfPct, 50, 99)}
              aria-label="Auto-accept minimum confidence"
              onChange={(event) => {
                onPolicyChange({ ...policy, minConfidence: Number(event.target.value) / 100 });
              }}
            />
          </label>
        )}
      </div>
    </div>
  );
}

function ClefAnswerRow({
  name,
  answer,
  index,
  policy,
}: {
  name: string;
  answer: ClefAnswer;
  index: number;
  policy: DecisionPolicy;
}): ReactElement {
  const color = fieldAccentColor(index);
  const routed = evaluateClefAnswerRoute(answer, policy);
  const cutoffPct = clampUnit(policy.minConfidence) * 100;

  if (answer.type === 'noul') {
    const noPct = clampUnit(policy.noThreshold) * 100;
    const yesPct = clampUnit(policy.yesThreshold) * 100;
    const reviewPct = Math.max(0, yesPct - noPct);
    const yesZonePct = Math.max(0, 100 - yesPct);
    const pipPct = clampUnit(answer.noul) * 100;

    return (
      <article className={styles.answerRow} data-review={routed.route === 'review' ? 'true' : undefined}>
        <div className={styles.answerHead}>
          <div className={styles.answerTitleGroup}>
            <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
            <span className={styles.answerName}>{name}</span>
            <span className={styles.answerType}>yes / no</span>
          </div>
          <div className={styles.answerPills}>
            <Badge size="sm" tone={routed.tone}>
              {routed.label}
            </Badge>
            <Badge size="sm" tone="neutral" outline>
              {`Yes ${formatPercent(answer.noul)}`}
            </Badge>
          </div>
        </div>
        <div className={styles.answerBody}>
          <div className={styles.decisionTrackWrap}>
            <div
              className={styles.decisionTrack}
              role="img"
              aria-label={`${name}: Yes ${formatPercent(answer.noul)}, routed as ${routed.label}`}
            >
              <span className={styles.decisionZoneNo} style={{ width: `${String(noPct)}%` }} />
              <span className={styles.decisionZoneReview} style={{ width: `${String(reviewPct)}%` }} />
              <span className={styles.decisionZoneYes} style={{ width: `${String(yesZonePct)}%` }} />
              <span className={styles.decisionTick} style={{ left: `${String(noPct)}%` }} aria-hidden="true" />
              <span className={styles.decisionTick} style={{ left: `${String(yesPct)}%` }} aria-hidden="true" />
              <span
                className={styles.decisionPip}
                style={{ left: `${String(pipPct)}%` }}
                data-route={routed.route}
                aria-hidden="true"
              />
            </div>
            <div className={styles.decisionAxis}>
              <span>{`Auto-No ≤ ${String(Math.round(noPct))}%`}</span>
              <span>Review</span>
              <span>{`Auto-Yes ≥ ${String(Math.round(yesPct))}%`}</span>
            </div>
          </div>
        </div>
      </article>
    );
  }

  if (answer.type === 'choice') {
    const entries = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
    return (
      <article className={styles.answerRow} data-review={routed.route === 'review' ? 'true' : undefined}>
        <div className={styles.answerHead}>
          <div className={styles.answerTitleGroup}>
            <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
            <span className={styles.answerName}>{name}</span>
            <span className={styles.answerType}>choice</span>
          </div>
          <div className={styles.answerPills}>
            <Badge size="sm" tone="info">
              {answer.choice}
            </Badge>
            <Badge size="sm" tone={routed.tone} outline={routed.route !== 'review'}>
              {routed.label}
            </Badge>
            <Badge size="sm" tone={confidenceTone(answer.confidence)}>
              {formatPercent(answer.confidence)}
            </Badge>
          </div>
        </div>
        <div className={styles.probList}>
          {entries.map(([option, prob]) => (
            <div key={option} className={styles.probRow}>
              <div className={styles.probMeta}>
                <span className={option === answer.choice ? 'font-medium text-foreground' : 'text-secondary'}>
                  {option}
                </span>
                <span className="tabular-nums text-secondary">{formatPercent(prob)}</span>
              </div>
              <div className={styles.meterWithCutoff}>
                <Meter
                  label={`${option} probability`}
                  value={prob * 100}
                  max={100}
                  unit="%"
                  size="dense"
                  warnAt={1.01}
                  dangerAt={1.01}
                />
                <span className={styles.meterCutoffTick} style={{ left: `${String(cutoffPct)}%` }} aria-hidden="true" />
              </div>
            </div>
          ))}
        </div>
      </article>
    );
  }

  const scoreEntries = Object.entries(answer.probabilities);
  return (
    <article className={styles.answerRow} data-review={routed.route === 'review' ? 'true' : undefined}>
      <div className={styles.answerHead}>
        <div className={styles.answerTitleGroup}>
          <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
          <span className={styles.answerName}>{name}</span>
          <span className={styles.answerType}>rating</span>
        </div>
        <div className={styles.answerPills}>
          <Badge size="sm" tone="info">
            {`rating ${answer.score.toFixed(2)}`}
          </Badge>
          <Badge size="sm" tone={routed.tone} outline={routed.route !== 'review'}>
            {routed.label}
          </Badge>
          <Badge size="sm" tone={confidenceTone(answer.confidence)}>
            {formatPercent(answer.confidence)}
          </Badge>
        </div>
      </div>
      <div className={styles.probList}>
        {scoreEntries.map(([levelKey, prob]) => (
          <div key={levelKey} className={styles.probRow}>
            <div className={styles.probMeta}>
              <span className="text-secondary">{`${levelKey}: ${answer.legend[levelKey] ?? levelKey}`}</span>
              <span className="tabular-nums text-secondary">{formatPercent(prob)}</span>
            </div>
            <div className={styles.meterWithCutoff}>
              <Meter
                label={`Level ${levelKey}`}
                value={prob * 100}
                max={100}
                unit="%"
                size="dense"
                warnAt={1.01}
                dangerAt={1.01}
              />
              <span className={styles.meterCutoffTick} style={{ left: `${String(cutoffPct)}%` }} aria-hidden="true" />
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

/** Flat hairline-separated structured answers for Clairic and CLEF results. */
export function ResultsView({
  result,
  requestFields,
  mode,
  presets,
  canRun,
  onRun,
  onPickPreset,
  onJumpToSpan,
  policy,
  onPolicyChange,
  hidePolicyBar = false,
  onOpenApiGuide,
  apiEndpointUrl = 'http://127.0.0.1:8080/v1/systemone',
}: ResultsViewProps): ReactElement {
  if (result === undefined || !isPlainObject(result.answers)) {
    const orderedPresets = [...presets].sort((a, b) => Number(b.mode === mode) - Number(a.mode === mode));
    return (
      <div className={styles.onboarding}>
        <div className={styles.onboardingHead}>
          <h3 className={styles.onboardingTitle}>Prototype here, ship via API</h3>
          <p className={styles.onboardingDesc}>
            {mode === 'decide'
              ? 'Use this studio to design rubrics and tune probability cutoffs, then call your local model from any service over HTTP or Node.'
              : 'Use this studio to test schemas and confidence cutoffs, then call your local model from any pipeline over HTTP or Node.'}
          </p>
        </div>

        <ol className={styles.onboardingSteps}>
          <li>
            <span className={styles.stepNum}>1</span>
            <span>
              {mode === 'decide'
                ? 'Paste sample context & define questions on the left (or load a Batch)'
                : 'Paste text or attach an image / PDF and define fields on the left'}
            </span>
          </li>
          <li>
            <span className={styles.stepNum}>2</span>
            <span>Press Run (⌘↵) and tune routing / confidence cutoffs on the results</span>
          </li>
          <li>
            <span className={styles.stepNum}>3</span>
            <span>
              Open <strong>Code &amp; JSON</strong> or <strong>API Guide</strong> to copy ready-to-run TypeScript,
              Python, cURL, or SDK code
            </span>
          </li>
        </ol>

        <div className={styles.apiOnboardingTip}>
          <div className={styles.apiOnboardingTipText}>
            <div className={styles.apiOnboardingTipTitle}>
              <Code2 size={13} strokeWidth={1.75} aria-hidden="true" />
              <span>Calling from your app?</span>
            </div>
            <p className={styles.apiOnboardingTipDesc}>
              Run <code>mlx serve</code> and <code>POST</code> to <code>{apiEndpointUrl}</code> with the same JSON
              payload — drop-in compatible with Cloudflare Workers AI.
            </p>
          </div>
          {onOpenApiGuide !== undefined ? (
            <Button type="button" variant="secondary" size="sm" onClick={onOpenApiGuide}>
              API Guide
            </Button>
          ) : null}
        </div>

        {canRun ? (
          <div>
            <Button type="button" variant="primary" size="sm" onClick={onRun}>
              <Play size={12} strokeWidth={2} aria-hidden="true" />
              Run now
            </Button>
          </div>
        ) : null}

        <div className={styles.onboardingPresets}>
          <span className={styles.smallLabel}>Try an example</span>
          <div className={styles.onboardingList}>
            {orderedPresets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={styles.onboardingRow}
                onClick={() => onPickPreset(preset)}
              >
                <div className={styles.onboardingRowText}>
                  <div className={styles.onboardingRowTitle}>
                    <span>{preset.label}</span>
                    {preset.mode !== mode ? (
                      <Badge size="sm" color={preset.mode === 'decide' ? 'orchid' : 'sky'} outline>
                        {preset.mode === 'decide' ? 'Decide' : 'Extract'}
                      </Badge>
                    ) : null}
                  </div>
                  <span className={styles.onboardingRowDesc}>{preset.description}</span>
                </div>
                <ArrowRight size={14} strokeWidth={1.75} className={styles.onboardingArrow} aria-hidden="true" />
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const entries = Object.entries(result.answers);
  if (entries.length === 0) {
    return (
      <div className={styles.emptyCenter}>
        <EmptyState title="Empty answers" description="The model returned an empty answers object." />
      </div>
    );
  }

  if (isClefResult(result)) {
    return (
      <>
        {!hidePolicyBar ? <PolicyStrip result={result} policy={policy} onPolicyChange={onPolicyChange} /> : null}
        <div className={styles.answerList}>
          {entries.map(([name, ans], index) => (
            <ClefAnswerRow key={name} name={name} answer={ans as ClefAnswer} index={index} policy={policy} />
          ))}
        </div>
      </>
    );
  }

  return (
    <>
      {!hidePolicyBar ? <PolicyStrip result={result} policy={policy} onPolicyChange={onPolicyChange} /> : null}
      <div className={styles.answerList}>
        {entries.map(([name, rawAnswer], index) => {
          const spec = isPlainObject(requestFields[name]) ? (requestFields[name] as Record<string, unknown>) : null;
          const color = fieldAccentColor(index);
          const answer = isPlainObject(rawAnswer) ? rawAnswer : { value: rawAnswer };
          const spans = extractAnswerSpans(answer);
          const confidence = typeof answer.confidence === 'number' ? answer.confidence : undefined;
          const pNull = typeof answer.p_null === 'number' ? answer.p_null : undefined;
          const isNullValue =
            !isForecastAnswer(answer) &&
            !isEstimateAnswer(answer) &&
            !isListAnswer(answer) &&
            (answer.value === null || answer.value === undefined);
          const showPNull = pNull !== undefined && !isNullValue && pNull >= 0.05;
          const isInvalid = answer.valid === false;
          const isIncomplete = answer.complete === false;
          const fieldType = friendlyFieldType(typeof spec?.type === 'string' ? spec.type : undefined);
          const unit =
            typeof answer.unit === 'string' ? answer.unit : typeof spec?.unit === 'string' ? spec.unit : undefined;
          const routed = evaluateClairicAnswerRoute(rawAnswer, policy);

          return (
            <article
              key={name}
              className={styles.answerRow}
              data-review={routed.route === 'review' ? 'true' : undefined}
            >
              <div className={styles.answerHead}>
                <div className={styles.answerTitleGroup}>
                  <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
                  <span className={styles.answerName}>{name}</span>
                  {fieldType !== undefined ? <span className={styles.answerType}>{fieldType}</span> : null}
                </div>

                <div className={styles.answerPills}>
                  {routed.route === 'review' && !isInvalid && !isIncomplete ? (
                    <Badge size="sm" tone="warning" outline>
                      {routed.label}
                    </Badge>
                  ) : null}
                  {confidence !== undefined ? (
                    <Tooltip
                      content={
                        isNullValue
                          ? `${formatPercent(pNull ?? confidence)} sure this field is missing from the document`
                          : `${formatPercent(confidence)} model confidence`
                      }
                    >
                      <span>
                        <Badge size="sm" tone={confidenceTone(confidence)}>
                          {formatPercent(confidence)}
                        </Badge>
                      </span>
                    </Tooltip>
                  ) : null}
                  {showPNull ? (
                    <Badge size="sm" tone={pNull >= 0.5 ? 'warning' : 'neutral'} outline>
                      {`${formatPercent(pNull)} chance missing`}
                    </Badge>
                  ) : null}
                  {isInvalid ? (
                    <Badge size="sm" tone="danger" icon={<AlertTriangle size={11} aria-hidden="true" />}>
                      invalid
                    </Badge>
                  ) : null}
                  {isIncomplete ? (
                    <Badge size="sm" tone="warning" icon={<AlertTriangle size={11} aria-hidden="true" />}>
                      incomplete
                    </Badge>
                  ) : null}
                  {spans.length > 0 && onJumpToSpan !== undefined ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => onJumpToSpan(name)}
                      className={styles.spanBtn}
                    >
                      Highlight
                      <ArrowUpRight size={12} strokeWidth={1.75} aria-hidden="true" />
                    </Button>
                  ) : null}
                </div>
              </div>

              <div className={styles.answerBody}>
                {isForecastAnswer(answer) ? (
                  <ForecastDisplay fieldName={name} answer={answer} spec={spec} color={color} />
                ) : isEstimateAnswer(answer) ? (
                  <RangeBar
                    label={name}
                    min={typeof spec?.min === 'number' || typeof spec?.min === 'string' ? spec.min : undefined}
                    max={typeof spec?.max === 'number' || typeof spec?.max === 'string' ? spec.max : undefined}
                    p10={answer.p10}
                    median={answer.median}
                    p90={answer.p90}
                    mean={answer.mean}
                    unit={unit}
                    pBelow={answer.p_below}
                    pAbove={answer.p_above}
                    quantiles={answer.quantiles}
                    color={color}
                  />
                ) : isListAnswer(answer) ? (
                  <ListAnswerDisplay items={answer.value} />
                ) : isNullValue ? (
                  <span className={styles.nullWrap}>
                    <span className={styles.nullLabel}>Not found</span>
                    {answer.guess !== undefined && answer.guess !== null ? (
                      <span className="text-caption text-muted">{`best guess: ${formatCellValue(answer.guess)}`}</span>
                    ) : null}
                  </span>
                ) : isPlainObject(answer.value) ? (
                  <KeyValue
                    layout="rows"
                    ariaLabel={name}
                    items={Object.entries(answer.value).map(([k, v]): KeyValueItem => ({
                      key: k,
                      label: k,
                      value: formatCellValue(v),
                      mono: typeof v === 'string' && v.length > 0 && !v.includes(' '),
                      numeric: typeof v === 'number',
                    }))}
                  />
                ) : (
                  <span className={styles.scalarValue}>
                    {formatCellValue(answer.value)}
                    {unit !== undefined && unit !== '' ? <span className={styles.scalarUnit}>{unit}</span> : null}
                  </span>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </>
  );
}

function ListAnswerDisplay({ items }: { items: readonly unknown[] }): ReactElement {
  if (items.length === 0) {
    return <span className="text-caption text-muted">Empty list</span>;
  }
  const objectRows = items.filter((item): item is Record<string, unknown> => isPlainObject(item));
  if (objectRows.length === items.length) {
    const colSet = new Set<string>();
    for (const row of objectRows) {
      for (const key of Object.keys(row)) colSet.add(key);
    }
    const cols = [...colSet];
    return (
      <div className={styles.flatTableWrap}>
        <table className={styles.flatTable}>
          <thead>
            <tr>
              {cols.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {objectRows.map((row, rowIdx) => (
              <tr key={rowIdx}>
                {cols.map((col) => (
                  <td key={col}>{formatCellValue(row[col])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <div className={styles.chipList}>
      {items.map((item, idx) => (
        <Badge key={idx} size="sm" outline>
          {formatCellValue(item)}
        </Badge>
      ))}
    </div>
  );
}

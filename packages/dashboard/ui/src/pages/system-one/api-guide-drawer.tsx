import { Terminal } from 'lucide-react';
import { type ReactElement, useState } from 'react';

import { CodePanel } from '../../components/arc-x/code-panel/code-panel';
import { Badge } from '../../components/arc/badge/badge';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { Drawer, DrawerContent } from '../../components/arc/drawer/drawer';
import SegmentedControl from '../../components/arc/segmented-control/segmented-control';
import type { SystemOneMode } from '../../lib/system-one';
import styles from './system-one.module.css';

export type ApiGuideLanguage = 'ts' | 'python' | 'curl' | 'sdk';

export interface ApiGuideDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: SystemOneMode;
  modelId: string;
  endpointUrl: string;
  /** Non-null when an external `mlx serve` process is already live on localhost. */
  liveServePort: number | null;
  tsSnippet: string;
  pythonSnippet: string;
  curlSnippet: string;
  sdkSnippet: string;
}

interface ApiTipItem {
  title: string;
  body: string;
}

const DECIDE_TIPS: readonly ApiTipItem[] = [
  {
    title: 'Use 3-zone routing instead of a 50% split',
    body: 'Route high-confidence Yes/No answers automatically and send the middle band to human review or a larger LLM. The snippet above stays synced to your live sliders.',
  },
  {
    title: 'Ask all questions in a single request',
    body: 'CLEF evaluates every Yes/No, choice, and rating question over the same context in one pass — combine them into one `questions` object instead of separate HTTP calls.',
  },
  {
    title: 'Pass structured JSON in state',
    body: 'CLEF accepts plain text or structured JSON objects in `state`. Consistent JSON keys keep probabilities well-calibrated across production traffic.',
  },
  {
    title: 'Drop-in Cloudflare Workers AI schema',
    body: 'The request and response payload match `@cf/cloudflare/clef` — swap the endpoint URL to move seamlessly between local Apple Silicon and cloud.',
  },
];

const EXTRACT_TIPS: readonly ApiTipItem[] = [
  {
    title: 'Gate automation on confidence & p_null',
    body: 'Accept fields above your confidence cutoff automatically; flag fields below cutoff (or `value: null` with low `p_null`) for a quick human check.',
  },
  {
    title: 'Send images & PDFs as base64 or data URIs',
    body: 'Attach up to 4 images or PDF pages per call (`images: [{ content_type, base64 }]`, max 4 MiB/file, 16 MiB total body).',
  },
  {
    title: 'Set max_items and reference_date for speed & accuracy',
    body: 'Set `max_items` on `list` fields to cap decoding slots, and pass `reference_date` (`YYYY-MM-DD`) whenever inputs contain relative dates like "next Tuesday".',
  },
  {
    title: 'Drop-in Cloudflare Workers AI schema',
    body: 'The request and response payload match `@cf/cloudflare/clairic` — prototype locally here, then reuse the exact same JSON schema in production.',
  },
];

/** Slide-over API guide showing how to call System One via local HTTP (`mlx serve`) or in-process `@mlx-node/lm`. */
export function ApiGuideDrawer({
  open,
  onOpenChange,
  mode,
  modelId,
  endpointUrl,
  liveServePort,
  tsSnippet,
  pythonSnippet,
  curlSnippet,
  sdkSnippet,
}: ApiGuideDrawerProps): ReactElement {
  const [lang, setLang] = useState<ApiGuideLanguage>('ts');

  const serveCommand = `mlx serve --port ${String(liveServePort ?? 8080)}`;
  const activeCode =
    lang === 'ts' ? tsSnippet : lang === 'python' ? pythonSnippet : lang === 'curl' ? curlSnippet : sdkSnippet;
  const activeLabel =
    lang === 'ts'
      ? 'TypeScript fetch snippet'
      : lang === 'python'
        ? 'Python requests snippet'
        : lang === 'curl'
          ? 'cURL snippet'
          : '@mlx-node/lm in-process SDK snippet';
  const tips = mode === 'decide' ? DECIDE_TIPS : EXTRACT_TIPS;

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent
        side="right"
        title="Use System One via API"
        description={
          mode === 'decide'
            ? 'Prototype rubrics and thresholds here, then call your local CLEF model from any app or backend.'
            : 'Prototype schemas and cutoffs here, then call your local Clairic model from any app or pipeline.'
        }
      >
        <div className={styles.apiGuideStack}>
          <section className={styles.apiGuideSection} aria-label="Local HTTP endpoint">
            <div className={styles.apiGuideStepHead}>
              <span className={styles.stepNum}>1</span>
              <span className={styles.apiGuideStepTitle}>Local HTTP endpoint</span>
              {liveServePort !== null ? (
                <Badge size="sm" tone="success">
                  {`mlx serve live on :${String(liveServePort)}`}
                </Badge>
              ) : (
                <Badge size="sm" tone="neutral" outline>
                  Start via terminal
                </Badge>
              )}
            </div>

            <div className={styles.apiEndpointCard}>
              <div className={styles.apiEndpointRow}>
                <Badge size="sm" color="sky">
                  POST
                </Badge>
                <code className={styles.apiEndpointUrl}>{endpointUrl}</code>
                <CopyButton value={endpointUrl} label="Copy URL" />
              </div>

              <div className={styles.apiEndpointMeta}>
                <span>{`model: "${modelId}"`}</span>
                <span>·</span>
                <span>Single JSON response (non-streaming)</span>
                <span>·</span>
                <span>Up to 16 MiB body</span>
              </div>
            </div>

            {liveServePort === null ? (
              <div className={styles.apiCmdBox}>
                <div className={styles.apiCmdLine}>
                  <Terminal size={13} strokeWidth={1.75} aria-hidden="true" className="shrink-0 text-muted" />
                  <code className={styles.apiCmdText}>{serveCommand}</code>
                  <CopyButton value={serveCommand} label="Copy command" />
                </div>
                <p className={styles.apiCmdHint}>
                  The Desktop App&apos;s own inference process is private. Run <code>mlx serve</code> to open an HTTP
                  API for external scripts and apps — models load lazily on first request. Add{' '}
                  <code>--auth-token &lt;secret&gt;</code> if exposing beyond localhost.
                </p>
              </div>
            ) : null}
          </section>

          <section className={styles.apiGuideSection} aria-label="Live client code">
            <div className={styles.apiGuideStepHead}>
              <span className={styles.stepNum}>2</span>
              <span className={styles.apiGuideStepTitle}>Copy code synced to your current setup</span>
            </div>

            <div className={styles.apiCodeToolbar}>
              <SegmentedControl
                label="API snippet language"
                options={[
                  { value: 'ts', label: 'TypeScript' },
                  { value: 'python', label: 'Python' },
                  { value: 'curl', label: 'cURL' },
                  { value: 'sdk', label: 'Node SDK' },
                ]}
                value={lang}
                onValueChange={(next) => setLang(next as ApiGuideLanguage)}
              />
            </div>

            <CodePanel code={activeCode} label={activeLabel} wrap maxLines={22} />

            {lang === 'sdk' ? (
              <p className={styles.apiCmdHint}>
                In-process mode loads the model directly inside Node.js via <code>@mlx-node/lm</code> — no separate{' '}
                <code>mlx serve</code> process needed.
              </p>
            ) : null}
          </section>

          <section className={styles.apiGuideSection} aria-label="API best practices">
            <div className={styles.apiGuideStepHead}>
              <span className={styles.stepNum}>3</span>
              <span className={styles.apiGuideStepTitle}>
                {mode === 'decide'
                  ? 'Tips for Decide (CLEF) in production'
                  : 'Tips for Extract (Clairic) in production'}
              </span>
            </div>

            <div className={styles.apiTipsList}>
              {tips.map((tip) => (
                <div key={tip.title} className={styles.apiTipRow}>
                  <span className={styles.apiTipTitle}>{tip.title}</span>
                  <p className={styles.apiTipBody}>{tip.body}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </DrawerContent>
    </Drawer>
  );
}

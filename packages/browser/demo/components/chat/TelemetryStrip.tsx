import { type ProfileLikeStats, formatTelemetry } from '../../lib/display-helpers';

export type TelemetryStripProps = {
  stats: ProfileLikeStats | null;
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
  modelLine: string;
};

export function TelemetryStrip({
  stats,
  prefillTokensPerSecond,
  decodeTokensPerSecond,
  modelLine,
}: TelemetryStripProps) {
  const view = formatTelemetry(stats, prefillTokensPerSecond, decodeTokensPerSecond, modelLine);
  return (
    <div className="telemetry-strip" role="status" aria-label="Telemetry">
      <span className="pill-tok">{view.decodeTokPerSec}</span>
      <span className="pill-prefill">{view.prefillTokPerSec}</span>
      <span className="sep">•</span>
      <span>{view.gpuRpc}</span>
      <span className="sep">•</span>
      <span>{view.pool}</span>
      <span className="right">{view.modelLine}</span>
    </div>
  );
}

import { type ProfileLikeStats, formatTelemetry } from "../../lib/screen-state";

export type TelemetryStripProps = {
  stats: ProfileLikeStats | null;
  decodeTokensPerSecond: number | null;
  modelLine: string;
};

export function TelemetryStrip({ stats, decodeTokensPerSecond, modelLine }: TelemetryStripProps) {
  const view = formatTelemetry(stats, decodeTokensPerSecond, modelLine);
  return (
    <div className="telemetry-strip" role="status" aria-label="Telemetry">
      <span className="pill-tok">{view.tokPerSec}</span>
      <span className="sep">•</span>
      <span>{view.gpuRpc}</span>
      <span className="sep">•</span>
      <span>{view.pool}</span>
      <span className="right">{view.modelLine}</span>
    </div>
  );
}

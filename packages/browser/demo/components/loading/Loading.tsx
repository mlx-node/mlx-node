import { formatLoadingText } from "../../lib/screen-state";

export type LoadingProps = {
  status: string | null;
};

export function Loading({ status }: LoadingProps) {
  return (
    <div className="overlay-screen" style={{ background: "var(--bg)" }}>
      <div className="loading-stack">
        <div className="loader-ring" aria-hidden="true" />
        <div>
          <div className="loader-text">{formatLoadingText(status)}</div>
          <div className="loader-sub">
            Model weights are cached for future visits.
          </div>
        </div>
      </div>
    </div>
  );
}

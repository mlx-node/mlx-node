import { useState } from "react";

export type InlinePreviewCardProps = {
  title: string;
  /** The full HTML doc to render in the iframe. */
  srcdoc: string;
};

export function InlinePreviewCard({ title, srcdoc }: InlinePreviewCardProps) {
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <>
      <div className="inline-preview">
        <div className="inline-preview-head">
          <span>▶ App preview · {title}</span>
          <button type="button" onClick={() => setFullscreen(true)}>
            ↗ open full-screen
          </button>
        </div>
        <iframe
          className="inline-preview-frame"
          srcDoc={srcdoc}
          title={title}
          sandbox="allow-scripts allow-forms allow-popups allow-modals allow-downloads"
        />
      </div>

      {fullscreen && (
        <div
          className="preview-modal-backdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget) setFullscreen(false);
          }}
        >
          <div className="preview-modal">
            <div className="preview-modal-head">
              <span>▶ App preview · {title}</span>
              <button
                type="button"
                style={{
                  background: "none",
                  border: "1px solid var(--border)",
                  color: "var(--text-dim)",
                  borderRadius: 6,
                  padding: "4px 10px",
                  cursor: "pointer",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
                onClick={() => setFullscreen(false)}
              >
                ✕ close (Esc)
              </button>
            </div>
            <iframe
              className="preview-modal-frame"
              srcDoc={srcdoc}
              title={title}
              sandbox="allow-scripts allow-forms allow-popups allow-modals allow-downloads"
            />
          </div>
        </div>
      )}
    </>
  );
}

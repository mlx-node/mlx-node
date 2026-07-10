import { useModelLoader } from '../providers/model-loader';

export default function JSpaceApp() {
  const { status, kickoffLoad } = useModelLoader();
  const modelReady = status === 'ready';

  return (
    <main className="mx-auto max-w-[110rem] px-4 py-6">
      <header>
        <h1 className="text-xl font-semibold">J-Space</h1>
        <p className="text-sm text-[var(--muted-foreground)]">
          Every layer’s guess, at every position — computed on your device.
        </p>
      </header>

      {!modelReady && (
        <section aria-labelledby="jspace-consent">
          <h2 id="jspace-consent">Run the model on your device</h2>
          <p>
            This downloads about <strong>1.6 GB</strong> of model weights, and a further{' '}
            <strong>46 MB</strong> when you first switch to the Jacobian lens. Nothing is
            downloaded until you press the button.
          </p>
          <button type="button" onClick={() => kickoffLoad()}>
            Download and run
          </button>
        </section>
      )}

      {/* Tasks 7-9 add the panels; Task 10 wires the data path. */}
    </main>
  );
}

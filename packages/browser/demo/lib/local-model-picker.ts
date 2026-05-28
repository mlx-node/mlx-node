/**
 * Triggers the hidden #model-dir-input file picker so route components can
 * open the local-model directory chooser without duplicating this lookup.
 */
export function triggerLocalPicker(): void {
  const el = document.getElementById('model-dir-input') as HTMLInputElement | null;
  el?.click();
}

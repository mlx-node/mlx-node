import { canAutoLoadModel } from './device-capability';

/**
 * Triggers the hidden #model-dir-input file picker so route components can
 * open the local-model directory chooser without duplicating this lookup.
 */
export function triggerLocalPicker(): void {
  // A blocked device (iOS Safari, low-RAM, no WebGPU) can't run the model: the
  // load a pick would kick off is gated at the source (handleLocalModelInputChange),
  // so opening the directory chooser here would only lead to a silent no-op. Skip
  // it so the user isn't handed a file dialog that does nothing. Desktops (and any
  // capable device) return true and open the picker as before.
  if (!canAutoLoadModel()) return;
  const el = document.getElementById('model-dir-input') as HTMLInputElement | null;
  el?.click();
}

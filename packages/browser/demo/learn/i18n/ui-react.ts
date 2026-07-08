// ui-react.ts — the React side of the UI string dictionary: a hook that
// resolves the current locale's strings from context. Kept separate so ui.ts
// stays React-free (same split as lib/i18n.ts vs lib/i18n-react.tsx).

import { useLocale } from '../../lib/i18n-react';
import { UI_STRINGS, type UiStrings } from './ui';

export function useUiStrings(): UiStrings {
  return UI_STRINGS[useLocale()];
}

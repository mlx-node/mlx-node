import { createContext } from 'react';

export interface DrilldownContextValue {
  /** detailKey of the open tile, or null. */
  open: string | null;
  /** id of the panel, for aria-controls. */
  panelId: string;
  /** A trigger was pressed: `tile` is the tile's root, measured as the rectangle the panel grows from. */
  toggle: (key: string, tile: HTMLElement) => void;
  /** Tiles register their trigger so focus can return to it when the panel closes. */
  registerTrigger: (key: string, trigger: HTMLElement | null) => void;
}

/** Given by KpiDrilldown to the StatTiles of its stage. */
export const DrilldownContext = createContext<DrilldownContextValue | null>(null);

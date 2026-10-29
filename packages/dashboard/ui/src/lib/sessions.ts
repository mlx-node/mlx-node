import { toast } from 'sonner';

import { DashboardApiError } from './api';

/** Display title: the stored title, or the id when the session never got one. */
export function sessionTitle(session: { id: string; title: string }): string {
  const title = session.title.trim();
  return title === '' ? session.id : title;
}

/** ISO timestamp from `summary.json` → epoch ms; `NaN` for a blank or bad value. */
export function isoMs(iso: string): number {
  return iso === '' ? Number.NaN : Date.parse(iso);
}

/** A 409 from a write guard: the session is live, recently written, or changed underneath. */
export function isConflict(err: unknown): err is DashboardApiError {
  return err instanceof DashboardApiError && err.status === 409;
}

export function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const LIVE_HINT = 'In use by a running agent. Rename and delete are off until it stops.';

export function copyResumeCommand(command: string): void {
  void navigator.clipboard.writeText(command).then(
    () => toast.success('Resume command copied', { description: command }),
    () => toast.error('Failed to copy to clipboard'),
  );
}

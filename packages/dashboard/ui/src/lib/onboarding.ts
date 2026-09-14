// Scope first-run dismissal to the configured model library, including when a
// browser/desktop renderer refuses persistent storage.
const dismissed = new Set<string>();
const key = (dir: string): string => `mlx-node:onboarding:v1:${dir}`;

export function hasDismissedOnboarding(dir: string): boolean {
  if (dismissed.has(dir)) return true;
  try {
    return localStorage.getItem(key(dir)) === 'done';
  } catch {
    return false;
  }
}

export function dismissOnboarding(dir: string): void {
  dismissed.add(dir);
  try {
    localStorage.setItem(key(dir), 'done');
  } catch {
    // The in-memory dismissal still allows navigation in this session.
  }
}

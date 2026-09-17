import { describe, expect, it } from 'vite-plus/test';

import { viewMenuTemplate } from '../src/main/menu-policy.js';

describe('viewMenuTemplate', () => {
  it('gates the developer trio behind dev access', () => {
    const roles = viewMenuTemplate(true).submenu.map((item) => item.role);
    expect(roles).toContain('reload');
    expect(roles).toContain('forceReload');
    expect(roles).toContain('toggleDevTools');
    // The debugging trio leads the menu, before the always-on zoom controls.
    expect(roles.indexOf('toggleDevTools')).toBeLessThan(roles.indexOf('resetZoom'));
  });

  it('never ships devtools in a production menu', () => {
    const roles = viewMenuTemplate(false).submenu.map((item) => item.role);
    expect(roles).not.toContain('reload');
    expect(roles).not.toContain('forceReload');
    expect(roles).not.toContain('toggleDevTools');
    // Zoom and fullscreen are standard and harmless: always present.
    expect(roles).toEqual(['resetZoom', 'zoomIn', 'zoomOut', undefined, 'togglefullscreen']);
  });
});

/** Words of the Fonts page, its popover block and its palette items. `{x}` is filled by `fill()` in `chat-strings.ts`. */
export const FONTS = {
  title: 'Fonts',
  description: 'Choose the font and text size for answers, code and the interface.',
  reset: 'Reset fonts',
  resetDone: 'Fonts reset',
  groups: {
    reading: { title: 'Reading text', description: 'Answers, your messages, reasoning and the message box.' },
    code: { title: 'Code', description: 'Code blocks, diffs, tool input and output.' },
    ui: { title: 'Interface', description: 'Menus, lists, buttons, tables and titles.' },
  },
  rows: { font: 'Font', size: 'Size', sample: 'Sample' },
  controls: {
    reading: { font: 'Reading font', size: 'Reading size' },
    code: { font: 'Code font', size: 'Code size' },
    ui: { font: 'Interface font', size: 'Interface size' },
  },
  range: '{min} to {max} px',
  rangeUi: '{min} to {max} px. Text only. ⌘+ and ⌘− zoom the whole window.',
  sizeSuffix: ' px',
  options: {
    inter: 'Inter (default)',
    geist: 'Geist',
    geistMono: 'Geist Mono (default)',
    system: 'System font',
    notInstalled: '{family} (not installed)',
    loading: 'Loading installed fonts…',
  },
  placeholder: 'Search fonts',
  empty: 'No font with that name',
  unavailable: 'Installed fonts are not available.',
  missing: '{family} is not installed on this computer. Showing {fallback}.',
  fallbackName: { reading: 'Inter', code: 'Geist Mono', ui: 'Inter' },
  samples: {
    reading: 'Reading text sample',
    code: 'Code sample',
    ui: 'Interface sample',
  },
  readingSample:
    'Portato read **12 files** and found the bug in `scheduler.rs`.\n\n| Step | Time |\n| --- | ---: |\n| Prefill | 1.8 s |\n| Decode | 42 tok/s |',
  codeSample:
    "function fits(text: string, width = 80): boolean {\n  return text.length <= width && text !== 'Il1 O0';\n}",
  uiSample: { title: 'Sessions', line: '12 turns today · 3.2 s median', time: 'Updated 2 minutes ago' },
  popover: { label: 'Fonts', link: 'Open font settings' },
  goTo: {
    label: 'Fonts',
    description: 'Font and text size',
    keywords: ['settings', 'text size', 'typography', 'appearance'],
  },
  resetAction: {
    label: 'Reset fonts',
    description: 'Back to the default fonts and sizes',
    keywords: ['typography', 'text size'],
  },
} as const;

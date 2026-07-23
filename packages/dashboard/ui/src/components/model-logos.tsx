import { cn } from '@/lib/utils';
import { Bird } from 'lucide-react';

/**
 * Per-model brand marks for assistant messages. Each model family shows the logo
 * of the org behind its base model — so a reader can tell at a glance which model
 * produced a turn. Marks are simplified, self-contained inline SVG (single path,
 * no external refs) traced from each provider's official icon; unknown models get
 * a name monogram. Brand colour lives on the glyph only, an accent against the
 * otherwise-monochrome chrome.
 */

export type ModelFamily = 'qwen' | 'gemma' | 'liquid' | 'ornith' | 'internscience' | 'generic';

/**
 * Classify a model id into a logo family by substring. Order matters: more
 * specific brands are checked before the Qwen base family. `Agents-A1`
 * (InternScience, on a Qwen3.5-MoE base) gets its own InternScience mark;
 * `Qwen-AgentWorld` is a first-party Qwen release, so it rides the Qwen mark.
 */
export function classifyModel(model: string): ModelFamily {
  const m = model.toLowerCase();
  if (m.includes('gemma')) return 'gemma';
  if (m.includes('lfm') || m.includes('liquid')) return 'liquid';
  if (m.includes('ornith')) return 'ornith';
  if (m.includes('agents-a1') || m.includes('internscience')) return 'internscience';
  if (m.includes('qwen') || m.includes('agentworld')) return 'qwen';
  return 'generic';
}

const PACKAGING = new Set([
  'mlx',
  'unsloth',
  'nvidia',
  'mxfp',
  'mxfp4',
  'mxfp8',
  'q4',
  'q8',
  'gguf',
  'bf16',
  'fp8',
  'awq',
  'nvfp4',
  'it',
]);

/**
 * A short, readable model name from its id — drop the quant/packaging suffixes
 * (`-unsloth-mxfp4-mlx`, `-q4`, …) and title-case the rest, keeping size/expert
 * codes upper (`27b` → `27B`, `a3b` → `A3B`).
 */
export function prettyModelName(model: string): string {
  const parts = model.split(/[-_\s]+/).filter((p) => p !== '' && !PACKAGING.has(p.toLowerCase()));
  if (parts.length === 0) return model;
  return parts
    .map((p) => {
      if (/^\d/.test(p)) return p.toUpperCase(); // 27b -> 27B, 3.6 -> 3.6
      if (/\d/.test(p) && p.length <= 4) return p.toUpperCase(); // a3b -> A3B
      return p.charAt(0).toUpperCase() + p.slice(1); // qwen -> Qwen, Gemma -> Gemma
    })
    .join(' ');
}

interface LogoProps {
  className?: string;
}

/** Qwen / Alibaba — the interlocking six-point knot mark (indigo). */
function QwenLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="#6336E7" aria-hidden>
      <path
        transform="scale(0.6667)"
        fillRule="evenodd"
        d="M12.604 1.34c.393.69.784 1.382 1.174 2.075a.18.18 0 00.157.091h5.552c.174 0 .322.11.446.327l1.454 2.57c.19.337.24.478.024.837-.26.43-.513.864-.76 1.3l-.367.658c-.106.196-.223.28-.04.512l2.652 4.637c.172.301.111.494-.043.77-.437.785-.882 1.564-1.335 2.34-.159.272-.352.375-.68.37-.777-.016-1.552-.01-2.327.016a.099.099 0 00-.081.05 575.097 575.097 0 01-2.705 4.74c-.169.293-.38.363-.725.364-.997.003-2.002.004-3.017.002a.537.537 0 01-.465-.271l-1.335-2.323a.09.09 0 00-.083-.049H4.982c-.285.03-.553-.001-.805-.092l-1.603-2.77a.543.543 0 01-.002-.54l1.207-2.12a.198.198 0 000-.197 550.951 550.951 0 01-1.875-3.272l-.79-1.395c-.16-.31-.173-.496.095-.965.465-.813.927-1.625 1.387-2.436.132-.234.304-.334.584-.335a338.3 338.3 0 012.589-.001.124.124 0 00.107-.063l2.806-4.895a.488.488 0 01.422-.246c.524-.001 1.053 0 1.583-.006L11.704 1c.341-.003.724.032.9.34zm-3.432.403a.06.06 0 00-.052.03L6.254 6.788a.157.157 0 01-.135.078H3.253c-.056 0-.07.025-.041.074l5.81 10.156c.025.042.013.062-.034.063l-2.795.015a.218.218 0 00-.2.116l-1.32 2.31c-.044.078-.021.118.068.118l5.716.008c.046 0 .08.02.104.061l1.403 2.454c.046.081.092.082.139 0l5.006-8.76.783-1.382a.055.055 0 01.096 0l1.424 2.53a.122.122 0 00.107.062l2.763-.02a.04.04 0 00.035-.02.041.041 0 000-.04l-2.9-5.086a.108.108 0 010-.113l.293-.507 1.12-1.977c.024-.041.012-.062-.035-.062H9.2c-.059 0-.073-.026-.043-.077l1.434-2.505a.107.107 0 000-.114L9.225 1.774a.06.06 0 00-.053-.031z"
      />
    </svg>
  );
}

/** Google Gemma — the blue four-point gem sparkle. */
function GemmaLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="#446EFF" aria-hidden>
      <path d="M8 1C8.3 5.5 10.5 7.7 15 8 10.5 8.3 8.3 10.5 8 15 7.7 10.5 5.5 8.3 1 8 5.5 7.7 7.7 5.5 8 1Z" />
    </svg>
  );
}

/** Liquid AI (LFM / LFM2) — the liquid-droplet / "A" mark (monochrome). */
function LiquidLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="currentColor" aria-hidden>
      <path
        transform="scale(0.6667)"
        d="M12.028 8.546l-.008.005 3.03 5.25a3.94 3.94 0 01.643 2.162c0 .754-.212 1.46-.58 2.062l6.173-1.991L11.63 0 9.304 3.872l2.724 4.674zM6.837 24l4.85-4.053h-.013c-2.219 0-4.017-1.784-4.017-3.984 0-.794.235-1.534.64-2.156l2.865-4.976-2.381-4.087L2 16.034 6.83 24h.007zM13.737 19.382h-.001L8.222 24h8.182l4.148-6.769-6.815 2.151z"
      />
    </svg>
  );
}

/** InternScience (Shanghai AI Lab) — the white lab-flask emblem on its blue tile. */
function InternScienceLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <rect width="16" height="16" rx="3" fill="#16069F" />
      <g fill="#fff">
        <circle cx="8" cy="1.5" r=".95" />
        <path d="M6.6 3H9.4V6.1L13 13.4Q13.5 14 12.7 14H3.3Q2.5 14 3 13.4L6.6 6.1Z" />
      </g>
    </svg>
  );
}

/** Two leading alphanumerics of a name, for the unknown-model fallback tile. */
function initialsOf(name: string): string {
  const first = name.trim().split(/\s+/)[0] ?? '';
  const alnum = first.replace(/[^a-z0-9]/gi, '');
  return (alnum.slice(0, 2) || '?').toUpperCase();
}

/** Neutral monogram tile for a model with no known brand. */
function Monogram({ model, className }: { model: string; className?: string }) {
  return (
    <span
      className={cn(
        'bg-foreground/10 text-foreground grid place-items-center rounded-[3px] text-[8px] leading-none font-bold',
        className,
      )}
      aria-hidden
    >
      {initialsOf(prettyModelName(model))}
    </span>
  );
}

/** The brand logo for a model id, sized by `className` (e.g. `size-3.5`). */
export function ModelLogo({ model, className }: { model: string; className?: string }) {
  switch (classifyModel(model)) {
    case 'qwen':
      return <QwenLogo className={className} />;
    case 'gemma':
      return <GemmaLogo className={className} />;
    case 'liquid':
      return <LiquidLogo className={className} />;
    case 'ornith':
      return <Bird className={className} style={{ color: '#FD8E5B' }} aria-hidden />;
    case 'internscience':
      return <InternScienceLogo className={className} />;
    default:
      return <Monogram model={model} className={className} />;
  }
}

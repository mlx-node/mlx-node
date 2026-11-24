import { Bird } from 'lucide-react';

import { cn } from '../../../lib/cn';
import { isAsciiDigit, isWhitespace, splitWhitespace } from '../../../lib/text-scan';
import styles from './model-logos.module.css';

/**
 * Per-model brand marks for assistant messages. Each model family shows the logo
 * of the org behind its base model — so a reader can tell at a glance which model
 * produced a turn. The brand marks in this file are self-contained inline SVG on
 * a 16×16 view box (no ids, no external refs) taken from each provider's official
 * icon; a multi-colour mark sets the fill on each shape, so no `currentColor`
 * rule recolours it. Unknown models get a name monogram. Brand colour lives on
 * the glyph only, an accent against the otherwise-monochrome chrome.
 */

export type ModelFamily = 'qwen' | 'gemma' | 'liquid' | 'ornith' | 'internscience' | 'cloudflare' | 'hy' | 'generic';

/**
 * Classify a model id into a logo family by substring. Order matters: more
 * specific brands are checked before the Qwen base family. `Agents-A1`
 * (InternScience, on a Qwen3.5-MoE base) gets its own InternScience mark;
 * `Qwen-AgentWorld` is a first-party Qwen release, so it rides the Qwen mark.
 * Tencent Hy names match `hunyuan`, `hy4`, `hy3`, `hy_v4`, or `hy-` at the start
 * of the name or of its repo part: a bare `hy` would take `Hybrid …`, and
 * `tencent` also publishes other brands.
 */
export function classifyModel(model: string): ModelFamily {
  const m = model.toLowerCase();
  if (
    m.includes('clairic') ||
    m.includes('clef') ||
    m.includes('cloudflare') ||
    m.includes('systemone') ||
    m.includes('system-one')
  ) {
    return 'cloudflare';
  }
  if (
    m.includes('hunyuan') ||
    m.includes('hy4') ||
    m.includes('hy3') ||
    m.includes('hy_v4') ||
    m.startsWith('hy-') ||
    m.includes('/hy-')
  ) {
    return 'hy';
  }
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
  const parts = nameParts(model).filter((p) => !PACKAGING.has(p.toLowerCase()));
  if (parts.length === 0) return model;
  return parts
    .map((p) => {
      if (isAsciiDigit(p[0])) return p.toUpperCase(); // 27b -> 27B, 3.6 -> 3.6
      if (hasDigit(p) && p.length <= 4) return p.toUpperCase(); // a3b -> A3B
      return p.charAt(0).toUpperCase() + p.slice(1); // qwen -> Qwen, Gemma -> Gemma
    })
    .join(' ');
}

/** The pieces of a model id between '-', '_' and whitespace; no empty pieces. */
function nameParts(model: string): string[] {
  const parts: string[] = [];
  let current = '';
  for (const ch of model) {
    if (ch === '-' || ch === '_' || isWhitespace(ch)) {
      if (current !== '') parts.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  if (current !== '') parts.push(current);
  return parts;
}

function hasDigit(text: string): boolean {
  for (const ch of text) if (isAsciiDigit(ch)) return true;
  return false;
}

function isAsciiAlnum(ch: string): boolean {
  const lower = ch.toLowerCase();
  return isAsciiDigit(ch) || (lower >= 'a' && lower <= 'z' && lower.length === 1);
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

/**
 * Tencent Hy (Hy4-flash; Hunyuan is another Tencent Hy name): the swirl-circle
 * mark. The shapes of hy.tencent.com `logo.svg` (view box 31×30) without three
 * zero-area slivers, scaled to 16; its luminance mask is the right half of its
 * own circle, drawn here as a path.
 */
function HyLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <g transform="matrix(.53333 0 0 .53333 -.122 0)">
        <path
          fill="#B3DDF2"
          d="M15.2287 30C23.4603 30 30.1333 23.2843 30.1333 15C30.1333 6.71573 23.4603 0 15.2287 0C6.9972 0 0.324219 6.71573 0.324219 15C0.324219 23.2843 6.9972 30 15.2287 30Z"
        />
        <path
          fill="#0053E0"
          d="M15.2318 29.9966C19.3467 29.9966 22.6824 26.6395 22.6824 22.4983C22.6824 18.3571 19.3467 15 15.2318 15C11.117 15 7.78125 18.3571 7.78125 22.4983C7.78125 26.6395 11.117 29.9966 15.2318 29.9966Z"
        />
        <path
          fill="#0053E0"
          d="M15.2326 0C23.4642 0 30.1372 6.71573 30.1372 15C30.1372 23.2843 23.4642 30 15.2326 30Z"
        />
        <path
          fill="#B3DDF2"
          d="M15.2318 14.9966C19.3467 14.9966 22.6824 11.6395 22.6824 7.49831C22.6824 3.35711 19.3467 0 15.2318 0C11.117 0 7.78125 3.35711 7.78125 7.49831C7.78125 11.6395 11.117 14.9966 15.2318 14.9966Z"
        />
        <path
          fill="#2AB9FF"
          d="M15.3712 0.00674309C18.5069 0.0842886 21.0228 2.66015 21.0228 5.83277C21.0228 8.50977 19.2271 10.8328 16.6007 11.5037C16.6007 11.5037 16.6241 11.5003 16.6576 11.4936C14.4097 11.9757 12.7246 13.9852 12.7246 16.3924C12.7246 19.1605 14.9524 21.4026 17.7028 21.4026C18.4164 21.4026 19.3879 21.234 20.001 20.9609C23.9675 19.3493 26.47 15.4552 26.47 10.8867C26.47 4.87188 21.6258 0 15.6559 0C15.5588 0 15.465 0.00337154 15.3712 0.00674309Z"
        />
        <path
          fill="#0053E0"
          d="M4.20222 18.9028C4.20222 15.8718 5.4116 13.124 7.36805 11.1213V11.1145C7.87726 10.5549 8.19217 9.81313 8.19217 8.99384C8.19217 7.2575 6.79518 5.85156 5.06989 5.85156C3.87056 5.85156 2.82868 6.53599 2.30607 7.53733C1.04979 9.73558 0.326172 12.2811 0.326172 15.0019C0.326172 23.0936 6.69468 29.685 14.6645 29.9884C8.83873 29.6917 4.20222 24.8435 4.20222 18.9028Z"
        />
        <path
          fill="#ECECEE"
          d="M15.3756 0.0078125C18.5113 0.085358 21.0272 2.66122 21.0272 5.83384C21.0272 8.51084 19.2014 10.8372 16.6051 11.5048C14.6285 11.9667 13.1679 13.5344 12.8262 15.4023C12.8262 15.4023 13.8948 15.0044 15.2148 15.0044C15.2148 15.0044 15.2081 15.0044 15.2047 15.0044C15.2148 15.0044 15.2215 15.0044 15.2315 15.0044C19.3488 15.0044 22.6821 11.6464 22.6821 7.50612C22.6821 3.36587 19.4225 0.085358 15.3756 0.0078125Z"
        />
      </g>
    </svg>
  );
}

/** Two leading alphanumerics of a name, for the unknown-model fallback tile. */
function initialsOf(name: string): string {
  const first = splitWhitespace(name)[0] ?? '';
  let alnum = '';
  for (const ch of first) if (isAsciiAlnum(ch)) alnum += ch;
  return (alnum.slice(0, 2) || '?').toUpperCase();
}

/** Neutral monogram tile for a model with no known brand. */
function Monogram({ model, className }: { model: string; className?: string }) {
  return (
    <span className={cn(styles.monogram, className)} aria-hidden>
      {initialsOf(prettyModelName(model))}
    </span>
  );
}

/** Cloudflare (Clairic / CLEF) — official Cloudflare cloud mark (orange). */
function CloudflareLogo({ className }: LogoProps) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="#F38020" aria-hidden>
      <path
        transform="scale(0.6667)"
        d="M16.5088 16.8447c.1475-.5068.0908-.9707-.1553-1.3154-.2246-.3164-.6045-.499-1.0615-.5205l-8.6592-.1123a.1559.1559 0 0 1-.1333-.0713c-.0283-.042-.0351-.0986-.021-.1553.0278-.084.1123-.1484.2036-.1562l8.7359-.1123c1.0351-.0489 2.1601-.8868 2.5537-1.9136l.499-1.3013c.0215-.0561.0293-.1128.0147-.168-.5625-2.5463-2.835-4.4453-5.5499-4.4453-2.5039 0-4.6284 1.6177-5.3876 3.8614-.4927-.3658-1.1187-.5625-1.794-.499-1.2026.119-2.1665 1.083-2.2861 2.2856-.0283.31-.0069.6128.0635.894C1.5683 13.171 0 14.7754 0 16.752c0 .1748.0142.3515.0352.5273.0141.083.0844.1475.1689.1475h15.9814c.0909 0 .1758-.0645.2032-.1553l.12-.4268zm2.7568-5.5634c-.0771 0-.1611 0-.2383.0112-.0566 0-.1054.0415-.127.0976l-.3378 1.1744c-.1475.5068-.0918.9707.1543 1.3164.2256.3164.6055.498 1.0625.5195l1.8437.1133c.0557 0 .1055.0263.1329.0703.0283.043.0351.1074.0214.1562-.0283.084-.1132.1485-.204.1553l-1.921.1123c-1.041.0488-2.1582.8867-2.5527 1.914l-.1406.3585c-.0283.0713.0215.1416.0986.1416h6.5977c.0771 0 .1474-.0489.169-.126.1122-.4082.1757-.837.1757-1.2803 0-2.6025-2.125-4.727-4.7344-4.727"
      />
    </svg>
  );
}

/** The brand logo for a model id, sized by `className`. */
export function ModelLogo({ model, className }: { model: string; className?: string }) {
  switch (classifyModel(model)) {
    case 'cloudflare':
      return <CloudflareLogo className={className} />;
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
    case 'hy':
      return <HyLogo className={className} />;
    default:
      return <Monogram model={model} className={className} />;
  }
}

/**
 * The catalog card of the Models page in every state it can show, with the real model names and sizes. The harness
 * refuses downloads, so the download, update and blocker states can only be seen here.
 */

import { type ReactElement, useState } from 'react';

import {
  INITIAL_PROGRESS,
  type ProgressEvent,
  type ProgressState,
  reduceProgress,
} from '../../../ui/src/components/download-progress';
import type {
  AdoptPlanResponse,
  CatalogBlocker,
  CatalogBuildItem,
  CatalogItem,
  ModelsResponse,
} from '../../../ui/src/lib/types';
import {
  CatalogCardPlaceholder,
  CatalogCardView,
  type CatalogCardViewProps,
  CatalogGrid,
} from '../../../ui/src/pages/models/catalog';
import type { ActiveJob, CatalogDownloadProps } from '../../../ui/src/pages/models/use-model-downloads';

const GIB = 2 ** 30;
const MB = 1e6;

function build(
  fields: Pick<CatalogBuildItem, 'id' | 'hfRepo' | 'sizeGb' | 'source' | 'slug'> & Partial<CatalogBuildItem>,
) {
  return {
    installed: false,
    present: false,
    heldBy: null,
    blockedByForeignDir: false,
    blocker: null,
    localRevision: null,
    localAssetsRepo: null,
    localAssetsRevision: null,
    ...fields,
  } satisfies CatalogBuildItem;
}

const QWEN_XL = build({
  id: 'UD-Q4_K_XL',
  hfRepo: 'unsloth/Qwen3.8-27B-GGUF',
  sizeGb: 17.7,
  source: 'Unsloth GGUF',
  slug: 'qwen3.8-27b-gguf',
});
const QWEN_M = build({
  id: 'UD-Q4_K_M',
  hfRepo: 'unsloth/Qwen3.8-27B-GGUF',
  sizeGb: 16.6,
  source: 'Unsloth GGUF',
  note: '1.1 GB smaller',
  slug: 'qwen3.8-27b-gguf-ud-q4_k_m',
});
const DRAFT: NonNullable<CatalogItem['draft']> = {
  label: 'DFlash2',
  hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
  sizeGb: 3.85,
  slug: 'qwen3.8-27b-dflash2',
  installed: false,
  present: false,
  blockedByForeignDir: false,
  blocker: null,
  localRevision: null,
};
const INKLING_XL = build({
  id: 'UD-Q4_K_XL',
  hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
  sizeGb: 18.7,
  source: 'Unsloth GGUF',
  included: 'Vision and DSpark draft included',
  slug: 'inkling-2-mini-gguf',
});
const INKLING_M = build({
  id: 'UD-Q4_K_M',
  hfRepo: 'unsloth/Inkling-2-Mini-GGUF',
  sizeGb: 17.7,
  source: 'Unsloth GGUF',
  note: '1 GB smaller',
  included: 'Vision and DSpark draft included',
  slug: 'inkling-2-mini-gguf-ud-q4_k_m',
});
const INKLING_ORIGINAL = build({
  id: 'Original',
  hfRepo: 'thinkingmachines/Inkling-2-Mini',
  sizeGb: 20.2,
  source: 'Thinking Machines',
  note: 'NVFP4 + MXFP8',
  included: 'Vision and DSpark draft included',
  slug: 'inkling-2-mini',
});
const HY4 = build({
  id: 'Q4_K_M',
  hfRepo: 'AngelSlim/Hy4-flash-GGUF',
  sizeGb: 168.3,
  source: 'AngelSlim GGUF',
  included: 'Streams from SSD',
  slug: 'hy4-flash-gguf',
});
const CLAIRIC_FLASH = build({
  id: 'UD-Q4_K_XL',
  hfRepo: 'Brooooooklyn/Clairic-Flash-GGUF',
  sizeGb: 7.6,
  source: 'GGUF',
  included: 'Vision and Clair head included',
  slug: 'clairic-flash-gguf',
});

function qwen(builds: [CatalogBuildItem, CatalogBuildItem], draft = DRAFT): CatalogItem {
  return { label: 'Qwen3.8-27B', description: 'Best tool use — recommended default', isDefault: true, builds, draft };
}
function inkling(builds: [CatalogBuildItem, CatalogBuildItem, CatalogBuildItem]): CatalogItem {
  return { label: 'Inkling-2-Mini', description: 'Looped MoE with vision and DSpark drafts, 1M context', builds };
}
function hy4(only: CatalogBuildItem): CatalogItem {
  return {
    label: 'Hy4-flash',
    description: '512-expert MoE streamed from SSD, 256K context',
    minMemoryGb: 96,
    builds: [only],
  };
}
function clairicFlash(only: CatalogBuildItem): CatalogItem {
  return {
    label: 'Clairic Flash (9B)',
    description: 'Fast 9B extraction, vision, PDF & forecasting',
    category: 'systemone',
    builds: [only],
  };
}

const INSTALLED = { present: true, installed: true, localRevision: 'a'.repeat(40) } as const;

/** A folder the app did not create, holding one leftover file. */
const LEFTOVER: CatalogBlocker = {
  kind: 'directory',
  fileCount: 1,
  sizeBytes: 192_223_904,
  truncated: false,
  names: ['imatrix_unsloth.gguf_file'],
  listed: false,
  modelLike: false,
  holdsModel: false,
  linkTarget: null,
  dangling: false,
};
/** A full model folder the app did not create. */
const FOREIGN_MODEL: CatalogBlocker = {
  ...LEFTOVER,
  fileCount: 4,
  sizeBytes: 7_612_909_326,
  names: [],
  listed: true,
  modelLike: true,
  holdsModel: true,
};

const PLAN: AdoptPlanResponse = {
  repo: CLAIRIC_FLASH.hfRepo,
  slug: CLAIRIC_FLASH.slug,
  token: 'a1'.repeat(32),
  revision: 'b'.repeat(40),
  assetsRevision: 'c'.repeat(40),
  check: { count: 3, bytes: 7_611_143_561 },
  download: { count: 6, bytes: 12_800_000 },
  other: { count: 1, bytes: 1765, names: ['README.md'] },
};

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function job(fields: Partial<ActiveJob> = {}): ActiveJob {
  return { id: 'gallery', committing: false, cancelling: false, connection: 0, source: 'local', ...fields };
}

function download(
  item: CatalogBuildItem | NonNullable<CatalogItem['draft']>,
  fields: Partial<CatalogDownloadProps> = {},
): CatalogDownloadProps {
  return {
    item: { heldBy: null, localAssetsRepo: null, localAssetsRevision: null, ...item },
    updateAvailable: false,
    settling: false,
    updateSettling: false,
    job: undefined,
    onInstall: () => wait(800),
    onReinstall: () => wait(800),
    onReinstallStarted: () => {},
    onPlanUpdate: () => wait(600).then(() => PLAN),
    onUpdateInPlace: () => wait(800),
    onDone: () => {},
    onError: () => {},
    onCancelled: () => {},
    onCancel: () => {},
    ...fields,
  };
}

/** The state after a start of `total` bytes and two frames a second apart, the second at `received`. */
function running(total: number, received: number, rate: number, fileCount = 1): ProgressState {
  const frame = (bytes: number): ProgressEvent => ({
    type: 'progress',
    id: 'gallery',
    file: 'Qwen3.8-27B-UD-Q4_K_M.gguf',
    receivedBytes: bytes,
    jobReceivedBytes: bytes,
    totalBytes: total,
    fileIndex: 0,
    fileCount,
  });
  const events: ProgressEvent[] = [
    { type: 'start', id: 'gallery', repo: QWEN_M.hfRepo, totalBytes: total, fileCount },
    frame(Math.max(0, received - rate)),
    frame(received),
  ];
  return events.reduce<ProgressState>(
    (state, event, index) => reduceProgress(state, event, index * 1000),
    INITIAL_PROGRESS,
  );
}

const HOST: ModelsResponse['host'] = { totalMemBytes: 128 * GIB, diskFreeBytes: 900 * GIB };

type Case = Omit<CatalogCardViewProps, 'onChoose' | 'onStart' | 'host' | 'progress' | 'starting'> &
  Partial<Pick<CatalogCardViewProps, 'host' | 'progress' | 'starting'>> & { name: string };

const locate = (): void => {};

const QWEN_M_RUNNING = running(16.6 * 1e9, 7.4 * 1e9, 84 * MB);
const DRAFT_RUNNING = running(3.85 * 1e9, 1.6 * 1e9, 42 * MB);

const CASES: readonly Case[] = [
  {
    name: 'Another build chosen (Original)',
    entry: inkling([INKLING_XL, INKLING_M, INKLING_ORIGINAL]),
    chosen: INKLING_ORIGINAL.slug,
    downloads: [download(INKLING_XL), download(INKLING_M), download(INKLING_ORIGINAL)],
  },
  {
    name: 'Another build chosen (UD-Q4_K_M)',
    entry: qwen([QWEN_XL, QWEN_M]),
    chosen: QWEN_M.slug,
    downloads: [download(QWEN_XL), download(QWEN_M)],
    draft: { download: download(DRAFT), progress: INITIAL_PROGRESS },
  },
  {
    name: 'Starting, Locate is off',
    entry: qwen([QWEN_XL, QWEN_M]),
    chosen: QWEN_M.slug,
    downloads: [download(QWEN_XL), download(QWEN_M)],
    draft: { download: download(DRAFT), progress: INITIAL_PROGRESS },
    starting: [QWEN_M.slug],
  },
  {
    name: 'Downloading, the model and DFlash2',
    entry: qwen([QWEN_XL, QWEN_M]),
    chosen: QWEN_M.slug,
    downloads: [download(QWEN_XL), download(QWEN_M, { job: job() })],
    progress: QWEN_M_RUNNING,
    draft: { download: download(DRAFT, { job: job({ id: 'draft' }) }), progress: DRAFT_RUNNING },
  },
  {
    name: 'A build that is not chosen downloads',
    entry: qwen([{ ...QWEN_XL, ...INSTALLED }, QWEN_M], { ...DRAFT, present: true, installed: true }),
    chosen: QWEN_XL.slug,
    downloads: [download({ ...QWEN_XL, ...INSTALLED }), download(QWEN_M, { job: job() })],
    draft: { download: download({ ...DRAFT, present: true, installed: true }), progress: INITIAL_PROGRESS },
  },
  {
    name: 'Last step of a download, no Cancel',
    entry: qwen([QWEN_XL, QWEN_M]),
    chosen: QWEN_M.slug,
    downloads: [download(QWEN_XL), download(QWEN_M, { job: job({ committing: true }) })],
    progress: reduceProgress(
      QWEN_M_RUNNING,
      {
        type: 'progress',
        id: 'gallery',
        file: 'Qwen3.8-27B-UD-Q4_K_M.gguf',
        receivedBytes: 16.6e9,
        jobReceivedBytes: 16.6e9,
        totalBytes: 16.6e9,
        fileIndex: 0,
        fileCount: 1,
      },
      2000,
    ),
    draft: { download: download(DRAFT), progress: INITIAL_PROGRESS },
  },
  {
    name: 'Installed, two builds on disk',
    entry: inkling([{ ...INKLING_XL, ...INSTALLED }, INKLING_M, { ...INKLING_ORIGINAL, ...INSTALLED }]),
    chosen: INKLING_ORIGINAL.slug,
    downloads: [
      download({ ...INKLING_XL, ...INSTALLED }),
      download(INKLING_M),
      download({ ...INKLING_ORIGINAL, ...INSTALLED }),
    ],
    detailsTo: `/models?focus=${INKLING_ORIGINAL.slug}`,
  },
  {
    name: 'Update available',
    entry: qwen([{ ...QWEN_XL, ...INSTALLED }, QWEN_M], { ...DRAFT, present: true, installed: true }),
    chosen: QWEN_XL.slug,
    downloads: [download({ ...QWEN_XL, ...INSTALLED }, { updateAvailable: true }), download(QWEN_M)],
    draft: { download: download({ ...DRAFT, present: true, installed: true }), progress: INITIAL_PROGRESS },
    detailsTo: `/models?focus=${QWEN_XL.slug}`,
  },
  {
    name: 'Needs cleanup: Reinstall',
    entry: inkling([{ ...INKLING_XL, blockedByForeignDir: true, blocker: LEFTOVER }, INKLING_M, INKLING_ORIGINAL]),
    chosen: INKLING_XL.slug,
    downloads: [
      download({ ...INKLING_XL, blockedByForeignDir: true, blocker: LEFTOVER }),
      download(INKLING_M),
      download(INKLING_ORIGINAL),
    ],
  },
  {
    name: 'Needs update: a folder the app did not make',
    entry: clairicFlash({ ...CLAIRIC_FLASH, blockedByForeignDir: true, blocker: FOREIGN_MODEL }),
    chosen: CLAIRIC_FLASH.slug,
    downloads: [download({ ...CLAIRIC_FLASH, blockedByForeignDir: true, blocker: FOREIGN_MODEL })],
  },
  {
    name: 'Needs cleanup: the app cannot replace it, listed',
    entry: hy4({ ...HY4, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, truncated: true, fileCount: 2000 } }),
    chosen: HY4.slug,
    downloads: [
      download({ ...HY4, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, truncated: true, fileCount: 2000 } }),
    ],
    detailsTo: `/models?focus=${HY4.slug}`,
  },
  {
    name: 'Needs cleanup: the app cannot replace it, not listed',
    entry: hy4({ ...HY4, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, kind: 'symlink', listed: false } }),
    chosen: HY4.slug,
    downloads: [
      download({ ...HY4, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, kind: 'symlink', listed: false } }),
    ],
  },
  {
    name: 'The folder holds another build',
    entry: qwen([
      { ...QWEN_XL, heldBy: 'UD-Q4_K_M' },
      { ...QWEN_M, present: true },
    ]),
    chosen: QWEN_XL.slug,
    downloads: [download({ ...QWEN_XL, heldBy: 'UD-Q4_K_M' }), download({ ...QWEN_M, present: true })],
    draft: { download: download(DRAFT), progress: INITIAL_PROGRESS },
    detailsTo: `/models?focus=${QWEN_XL.slug}`,
  },
  {
    name: 'The folder holds another model',
    entry: inkling([{ ...INKLING_XL, heldBy: 'another model' }, INKLING_M, INKLING_ORIGINAL]),
    chosen: INKLING_XL.slug,
    downloads: [download({ ...INKLING_XL, heldBy: 'another model' }), download(INKLING_M), download(INKLING_ORIGINAL)],
  },
  {
    name: 'Warning: disk',
    entry: hy4(HY4),
    chosen: HY4.slug,
    downloads: [download(HY4)],
    host: { totalMemBytes: 128 * GIB, diskFreeBytes: 121 * GIB },
  },
  {
    name: 'Warning: memory',
    entry: inkling([INKLING_XL, INKLING_M, INKLING_ORIGINAL]),
    chosen: INKLING_ORIGINAL.slug,
    downloads: [download(INKLING_XL), download(INKLING_M), download(INKLING_ORIGINAL)],
    host: { totalMemBytes: 16 * GIB, diskFreeBytes: 900 * GIB },
  },
  {
    name: 'A description longer than two lines',
    entry: {
      ...clairicFlash(CLAIRIC_FLASH),
      description:
        'Fast 9B extraction with vision, PDF and forecasting, tuned for long scanned contracts, invoices, bank statements and wide tables, with a Clair head that reads charts, stamps and handwriting in the margins of every page, and keeps the layout of each page when it writes the text out again.',
    },
    chosen: CLAIRIC_FLASH.slug,
    downloads: [download(CLAIRIC_FLASH)],
  },
  {
    name: 'System One card, installed',
    entry: clairicFlash({ ...CLAIRIC_FLASH, ...INSTALLED }),
    chosen: CLAIRIC_FLASH.slug,
    downloads: [download({ ...CLAIRIC_FLASH, ...INSTALLED })],
    detailsTo: `/models?focus=${CLAIRIC_FLASH.slug}`,
    openTo: `/system-one?model=${CLAIRIC_FLASH.slug}`,
  },
];

/** The DFlash2 line in each state, on a Qwen3.8-27B card with UD-Q4_K_XL chosen. */
const DRAFT_CASES: ReadonlyArray<{
  name: string;
  draft: NonNullable<CatalogItem['draft']>;
  fields?: Partial<CatalogDownloadProps>;
  progress?: ProgressState;
  starting?: boolean;
  modelOnDisk?: boolean;
}> = [
  { name: 'DFlash2: click sent', draft: DRAFT, starting: true },
  { name: 'DFlash2: last step', draft: DRAFT, fields: { job: job({ committing: true }) }, progress: DRAFT_RUNNING },
  { name: 'DFlash2: added, waits for the model', draft: { ...DRAFT, present: true, installed: true } },
  {
    name: 'DFlash2: update available',
    draft: { ...DRAFT, present: true, installed: true },
    fields: { updateAvailable: true },
    modelOnDisk: true,
  },
  {
    name: 'DFlash2: needs update',
    draft: { ...DRAFT, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, sizeBytes: 3_850_000_000 } },
  },
  {
    name: 'DFlash2: needs cleanup, Reinstall',
    draft: { ...DRAFT, blockedByForeignDir: true, blocker: { ...LEFTOVER, fileCount: 0, sizeBytes: 0, names: [] } },
  },
  {
    name: 'DFlash2: needs cleanup, the app cannot replace it',
    draft: { ...DRAFT, blockedByForeignDir: true, blocker: { ...FOREIGN_MODEL, kind: 'symlink', listed: false } },
  },
];

function GalleryCard({ name, chosen: first, ...props }: Case): ReactElement {
  const [chosen, setChosen] = useState(first);
  return (
    <div data-gallery-card={name} style={{ display: 'grid', gridTemplateRows: 'auto 1fr', gap: 4, minWidth: 0 }}>
      <span className="gallery-note">{name}</span>
      <CatalogCardView
        {...props}
        chosen={chosen}
        onChoose={setChosen}
        progress={props.progress ?? INITIAL_PROGRESS}
        starting={props.starting ?? []}
        onStart={() => {}}
        host={props.host ?? HOST}
        locate={locate}
      />
    </div>
  );
}

export function CatalogCardSection(): ReactElement {
  const draftCases: Case[] = DRAFT_CASES.map((item) => {
    const xl = item.modelOnDisk === true ? { ...QWEN_XL, ...INSTALLED } : QWEN_XL;
    return {
      name: item.name,
      entry: qwen([xl, QWEN_M], item.draft),
      chosen: QWEN_XL.slug,
      downloads: [download(xl), download(QWEN_M)],
      draft: { download: download(item.draft, item.fields), progress: item.progress ?? INITIAL_PROGRESS },
      starting: item.starting === true ? [DRAFT.slug] : [],
    };
  });
  return (
    <CatalogGrid>
      {[...CASES, ...draftCases].map((item) => (
        <GalleryCard key={item.name} {...item} />
      ))}
      <div data-gallery-card="Loading" style={{ display: 'grid', gridTemplateRows: 'auto 1fr', gap: 4, minWidth: 0 }}>
        <span className="gallery-note">Loading</span>
        <CatalogCardPlaceholder />
      </div>
    </CatalogGrid>
  );
}

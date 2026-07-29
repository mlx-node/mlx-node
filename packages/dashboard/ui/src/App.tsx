import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import Models from '@/pages/models';
import Overview from '@/pages/overview';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { Boxes, HardDrive, LayoutDashboard, type LucideIcon, MessagesSquare, TrendingUp } from 'lucide-react';
import { type CSSProperties, useLayoutEffect, useRef, useState } from 'react';
import { BrowserRouter, NavLink, Outlet, Route, Routes, useLocation } from 'react-router-dom';

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/sessions', label: 'Sessions', icon: MessagesSquare },
  { to: '/metrics', label: 'Metrics', icon: TrendingUp },
  { to: '/cache', label: 'Cache', icon: HardDrive },
];

function Sidebar() {
  const { pathname } = useLocation();
  const navRef = useRef<HTMLElement>(null);
  // `null` until measured. Rendering the pill at a guessed position and
  // correcting it a frame later is a visible jump on first paint.
  const [pill, setPill] = useState<{ top: number; height: number } | null>(null);

  useLayoutEffect(() => {
    // Read the DOM rather than deriving from `pathname`: NavLink decides what
    // counts as current (`end`, and prefix matching that keeps Sessions lit on
    // /sessions/:id), so asking the element it marked keeps one rule instead of
    // reimplementing that matching here and letting the two disagree.
    const active = navRef.current?.querySelector<HTMLElement>('[aria-current="page"]') ?? null;
    setPill(active === null ? null : { top: active.offsetTop, height: active.offsetHeight });
  }, [pathname]);

  return (
    // `h-full`, not `sticky top-0 h-screen`: the shell no longer scrolls, so
    // there is nothing to stick to. Height comes from the shell rather than from
    // `100vh` so the two cannot disagree.
    <aside className="bg-sidebar/75 text-sidebar-foreground flex h-full w-60 shrink-0 flex-col border-r backdrop-blur-xl">
      {/*
        The window is `titleBarStyle: 'hiddenInset'`, so macOS paints the close/
        minimise/zoom buttons OVER the content at the top-left — which is exactly
        where this sidebar puts its logo. Without this strip the traffic lights
        sit on top of the mark.

        It doubles as the drag handle. `hiddenInset` removes the title bar you
        would otherwise grab, so something has to be declared draggable or the
        window can only be moved by its edges. Children must opt back out with
        `no-drag` or they stop receiving clicks — which is why the strip is empty
        rather than wrapping the header.

        Harmless outside Electron (`-webkit-app-region` is ignored in a plain
        browser), so no platform branch is needed for `vp dev`.
      */}
      <div className="h-9 shrink-0" style={{ WebkitAppRegion: 'drag' } as CSSProperties} aria-hidden />

      <div className="flex h-16 items-center gap-2.5 px-5">
        <span className="bg-brand-gradient text-primary-foreground shadow-soft flex size-9 items-center justify-center rounded-xl">
          <Boxes className="size-5" aria-hidden />
        </span>
        <div className="leading-tight">
          <div className="text-foreground text-sm font-semibold">mlx-node</div>
          <div className="text-muted-foreground text-[11px] font-medium tracking-[0.14em] uppercase">
            Control Panel
          </div>
        </div>
      </div>

      <nav ref={navRef} className="relative flex flex-1 flex-col gap-0.5 px-3 py-2">
        <p className="text-muted-foreground/70 px-3 pt-2 pb-1.5 text-[11px] font-semibold tracking-[0.12em] uppercase">
          Workspace
        </p>

        {/*
          ONE highlight that travels, rather than a background on each item.

          Every item owning its own `aria-[current=page]:bg-accent` meant a switch
          was two independent fades — one out, one in — with nothing moving
          between them. No amount of easing fixes that, because there is no
          motion to ease: the eye is asked to re-find the highlight rather than
          follow it.

          Position is MEASURED off the active link instead of computed from an
          index and a row height. Padding, font size and the section label all
          feed the real offset, so arithmetic here would be a second source of
          truth that silently drifts the first time any of them changes.

          Transitioning `transform` rather than `top` keeps this on the
          compositor — no layout per frame. The initial render sets the transform
          before paint (useLayoutEffect), and CSS transitions do not fire on an
          element's first style, so the pill appears in place instead of flying
          in from the top on mount.
        */}
        {pill !== null && (
          <span
            aria-hidden
            className="bg-accent pointer-events-none absolute top-0 right-3 left-3 rounded-lg transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none"
            style={{ height: pill.height, transform: `translateY(${pill.top}px)` }}
          >
            <span className="bg-primary absolute top-1/2 left-0 h-4 w-1 -translate-y-1/2 rounded-r-full" />
          </span>
        )}

        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            // `relative` keeps the label above the pill, which is painted first.
            // Only colour transitions here — the background belongs to the pill.
            className="group text-muted-foreground hover:text-foreground aria-[current=page]:text-accent-foreground relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors"
          >
            <item.icon className="text-muted-foreground group-hover:text-foreground size-4 shrink-0 transition-colors group-aria-[current=page]:text-primary" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t px-5 py-3.5">
        <div className="text-muted-foreground flex items-center gap-2 text-xs font-medium">
          <span className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-500/60" />
            <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
          </span>
          Running locally
        </div>
      </div>
    </aside>
  );
}

function Layout() {
  return (
    // The shell is exactly the viewport and never scrolls; `main` is the only
    // scroll container in the app.
    //
    // It used to be `min-h-screen` — which lets the DOCUMENT scroll — with
    // `main` ALSO declared `overflow-auto`. Two nested scroll containers for one
    // scrolling region, and on the Metrics page both drew a scrollbar: recharts
    // positions a legend `absolute`, so an 18-model legend added 117px to
    // `main`'s scrollHeight without ever growing its box. `main` became
    // scrollable inside a document that was already scrolling.
    //
    // Sizing the shell to the viewport also gets the behaviour a desktop window
    // should have: the sidebar and the drag strip above it are structurally
    // fixed rather than merely sticky, and only the content moves.
    <div className="app-shell text-foreground flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-8 lg:px-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="models" element={<Models />} />
            <Route path="sessions" element={<Sessions />} />
            <Route path="sessions/:id" element={<SessionDetail />} />
            <Route path="metrics" element={<Metrics />} />
            <Route path="cache" element={<Cache />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster />
    </TooltipProvider>
  );
}

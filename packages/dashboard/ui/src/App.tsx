import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import Cache from '@/pages/cache';
import Metrics from '@/pages/metrics';
import Models from '@/pages/models';
import Overview from '@/pages/overview';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { Boxes, HardDrive, LayoutDashboard, type LucideIcon, MessagesSquare, TrendingUp } from 'lucide-react';
import { BrowserRouter, NavLink, Outlet, Route, Routes } from 'react-router-dom';

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
  return (
    <aside className="bg-sidebar/75 text-sidebar-foreground sticky top-0 flex h-screen w-60 shrink-0 flex-col border-r backdrop-blur-xl">
      <div className="flex h-16 items-center gap-2.5 px-5">
        <span className="bg-brand-gradient text-primary-foreground shadow-soft flex size-9 items-center justify-center rounded-xl">
          <Boxes className="size-5" aria-hidden />
        </span>
        <div className="leading-tight">
          <div className="text-foreground text-sm font-semibold">mlx-node</div>
          <div className="text-muted-foreground text-[11px] font-medium tracking-[0.14em] uppercase">Agent</div>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-0.5 px-3 py-2">
        <p className="text-muted-foreground/70 px-3 pt-2 pb-1.5 text-[11px] font-semibold tracking-[0.12em] uppercase">
          Workspace
        </p>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className="group text-muted-foreground hover:bg-accent/50 hover:text-foreground aria-[current=page]:bg-accent aria-[current=page]:text-accent-foreground relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-all"
          >
            <span className="bg-primary absolute top-1/2 left-0 h-4 w-1 -translate-y-1/2 rounded-r-full opacity-0 transition-opacity group-aria-[current=page]:opacity-100" />
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
    <div className="app-shell text-foreground flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">
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

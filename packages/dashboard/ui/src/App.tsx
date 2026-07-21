import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
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
    <aside className="bg-sidebar text-sidebar-foreground flex w-56 shrink-0 flex-col border-r">
      <div className="flex h-14 items-center gap-2 border-b px-4">
        <LayoutDashboard className="size-5" />
        <span className="font-semibold">mlx dashboard</span>
      </div>
      <nav className="flex flex-col gap-1 p-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                  : 'text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground',
              )
            }
          >
            <item.icon className="size-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

function Layout() {
  return (
    <div className="bg-background text-foreground flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}

function Placeholder({ title, description }: { title: string; description: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="text-muted-foreground text-sm">This page is under construction.</CardContent>
    </Card>
  );
}

export default function App() {
  return (
    <TooltipProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route
              index
              element={<Placeholder title="Overview" description="Models, sessions, tokens, and cache at a glance." />}
            />
            <Route
              path="models"
              element={
                <Placeholder title="Models" description="Local models and downloads from the recommended list." />
              }
            />
            <Route
              path="sessions"
              element={<Placeholder title="Sessions" description="Browse, rename, and resume agent sessions." />}
            />
            <Route
              path="sessions/:id"
              element={
                <Placeholder title="Session detail" description="Transcript and per-turn metrics for a session." />
              }
            />
            <Route
              path="metrics"
              element={<Placeholder title="Metrics" description="Throughput, token, and acceptance trends." />}
            />
            <Route
              path="cache"
              element={<Placeholder title="Cache" description="PagedAttention cold-tier usage and management." />}
            />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster />
    </TooltipProvider>
  );
}

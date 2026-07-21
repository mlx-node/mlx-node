import { cn } from '@/lib/utils';
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';

/**
 * Minimal shadcn-shaped collapsible (`Collapsible` / `CollapsibleTrigger` /
 * `CollapsibleContent`) built on React state rather than `@radix-ui/react-collapsible`,
 * so it adds no dependency. Supports controlled (`open` + `onOpenChange`) and
 * uncontrolled (`defaultOpen`) use; the trigger is a real `<button>` carrying
 * `aria-expanded`, and content unmounts while closed.
 */

interface CollapsibleContextValue {
  open: boolean;
  toggle: () => void;
}

const CollapsibleContext = createContext<CollapsibleContextValue | null>(null);

function useCollapsible(): CollapsibleContextValue {
  const ctx = useContext(CollapsibleContext);
  if (ctx === null) throw new Error('Collapsible parts must be used within <Collapsible>');
  return ctx;
}

export interface CollapsibleProps {
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
  children: ReactNode;
}

export function Collapsible({
  defaultOpen = false,
  open: controlled,
  onOpenChange,
  className,
  children,
}: CollapsibleProps) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen);
  const open = controlled ?? uncontrolled;
  const toggle = useCallback(() => {
    const next = !open;
    if (controlled === undefined) setUncontrolled(next);
    onOpenChange?.(next);
  }, [open, controlled, onOpenChange]);
  return (
    <CollapsibleContext.Provider value={{ open, toggle }}>
      <div data-slot="collapsible" data-state={open ? 'open' : 'closed'} className={className}>
        {children}
      </div>
    </CollapsibleContext.Provider>
  );
}

export function CollapsibleTrigger({ className, children }: { className?: string; children: ReactNode }) {
  const { open, toggle } = useCollapsible();
  return (
    <button
      type="button"
      data-slot="collapsible-trigger"
      data-state={open ? 'open' : 'closed'}
      aria-expanded={open}
      onClick={toggle}
      className={cn('flex w-full items-center gap-2 text-left', className)}
    >
      {children}
    </button>
  );
}

export function CollapsibleContent({ className, children }: { className?: string; children: ReactNode }) {
  const { open } = useCollapsible();
  if (!open) return null;
  return (
    <div data-slot="collapsible-content" className={className}>
      {children}
    </div>
  );
}

/**
 * Toasts from anywhere, including code outside React (`lib/sessions.ts`).
 *
 * `ToastBridge`, mounted once inside Arc's `ToastStackProvider`, hands the stack to
 * this module. Calls made before it mounts are queued and replayed in order.
 */

import { useEffect } from 'react';

import { useToastStack, type ToastOptions, type ToastStackApi } from '../components/arc/toast-stack/toast-stack';

export interface ToastInput {
  description?: string;
  id?: string;
  duration?: number;
  action?: { label: string; onClick(id: string): void };
}

type Call = (api: ToastStackApi) => void;

let api: ToastStackApi | null = null;
const queue: Call[] = [];
let nextId = 0;

function run(call: Call): void {
  if (api === null) queue.push(call);
  else call(api);
}

function show(type: NonNullable<ToastOptions['type']>, title: string, input: ToastInput = {}): string {
  nextId += 1;
  const id = input.id ?? `toast-${nextId}`;
  const options: ToastOptions = { id, type, title };
  if (input.description !== undefined) options.description = input.description;
  if (input.duration !== undefined) options.duration = input.duration;
  if (input.action !== undefined) options.action = input.action;
  run((target) => {
    target.toast(options);
  });
  return id;
}

export const toast = {
  success: (title: string, input?: ToastInput): string => show('success', title, input),
  error: (title: string, input?: ToastInput): string => show('error', title, input),
  info: (title: string, input?: ToastInput): string => show('info', title, input),
  warning: (title: string, input?: ToastInput): string => show('warning', title, input),
  loading: (title: string, input?: ToastInput): string => show('loading', title, input),
  update(id: string, patch: Partial<ToastOptions>): void {
    const rest = { ...patch };
    delete rest.id;
    run((target) => target.update(id, rest));
  },
  dismiss(id?: string): void {
    run((target) => target.dismiss(id));
  },
};

/** Mounted once inside the provider; hands useToastStack() to the module. Calls made before mount are queued. */
export function ToastBridge(): null {
  const stack = useToastStack();
  const { toast: show, update, dismiss } = stack;
  useEffect(() => {
    const bound: ToastStackApi = { toast: show, update, dismiss };
    api = bound;
    for (const call of queue.splice(0)) call(bound);
    return () => {
      if (api === bound) api = null;
    };
  }, [show, update, dismiss]);
  return null;
}

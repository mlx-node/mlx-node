import {
  useLayoutEffect,
  useRef,
  useState,
  type AnimationEvent,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from 'react';

import { cn } from '../../../lib/cn';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './placeholder.module.css';

export type PlaceholderRadius = 'text' | 'control' | 'panel' | 'pill';

export interface PlaceholderProps {
  /** Default '100%'. */
  width?: string | number;
  /** Default '1lh'. */
  height?: string | number;
  /** 'text' paints a text bar at mid height inside the box. Default 'text'. */
  radius?: PlaceholderRadius;
  inline?: boolean;
  /** The value this slot holds once loaded. Without children the bone is rendered alone. */
  children?: ReactNode;
  /** Only read with children. Default true. */
  loading?: boolean;
  /** Place in the top-to-bottom order of its card (0-7); only read with children. */
  step?: number;
  className?: string;
}

type SlotState = 'loading' | 'settling' | 'ready';

const MAX_STEP = 7;

function size(value: string | number | undefined, fallback: string): string {
  if (value === undefined) return fallback;
  return typeof value === 'number' ? `${value}px` : value;
}

function clampStep(step: number | undefined): number {
  if (step === undefined || !Number.isFinite(step)) return 0;
  return Math.min(MAX_STEP, Math.max(0, Math.floor(step)));
}

interface BoneProps {
  width: string;
  height: string;
  radius: PlaceholderRadius;
  inline: boolean;
  settling?: boolean;
  step?: number;
  className?: string;
  boneRef?: (node: HTMLElement | null) => void;
  onAnimationEnd?: (event: AnimationEvent<HTMLElement>) => void;
}

function Bone(props: BoneProps): ReactElement {
  const Tag = props.inline ? 'span' : 'div';
  const style = {
    width: props.width,
    height: props.height,
    '--step': props.step ?? 0,
  } as CSSProperties;
  return (
    <Tag
      ref={props.boneRef}
      className={cn(styles.bone, props.className)}
      style={style}
      data-radius={props.radius}
      data-placeholder={props.settling === true ? undefined : ''}
      data-settling={props.settling === true ? '' : undefined}
      aria-hidden="true"
      onAnimationEnd={props.onAnimationEnd}
    />
  );
}

/**
 * A loading block. With children it is the slot of a value: the bone while loading, then the children, with
 * the bone shrinking to the content box and fading under it (skeleton -> content).
 */
export function Placeholder(props: PlaceholderProps): ReactElement {
  const width = size(props.width, '100%');
  const height = size(props.height, '1lh');
  const radius = props.radius ?? 'text';
  const inline = props.inline === true;
  const hasChildren = props.children !== undefined;
  const loading = props.loading !== false;
  const staticMotion = useStaticMotion();

  const [slot, setSlot] = useState<SlotState>(loading ? 'loading' : 'ready');
  const [seenLoading, setSeenLoading] = useState(loading);
  if (seenLoading !== loading) {
    setSeenLoading(loading);
    setSlot(loading ? 'loading' : 'settling');
  }
  const state: SlotState = staticMotion ? (loading ? 'loading' : 'ready') : slot;

  const slotRef = useRef<HTMLElement | null>(null);
  const boneRef = useRef<HTMLElement | null>(null);
  const lastBone = useRef<{ width: number; height: number } | null>(null);

  useLayoutEffect(() => {
    if (state === 'loading') {
      const bone = boneRef.current;
      if (bone !== null) {
        const rect = bone.getBoundingClientRect();
        lastBone.current = { width: rect.width, height: rect.height };
      }
      return;
    }
    if (state !== 'settling') return;
    const bone = boneRef.current;
    const box = slotRef.current?.getBoundingClientRect();
    const from = lastBone.current;
    if (bone === null || box === undefined || from === null || box.width <= 0 || box.height <= 0) {
      setSlot('ready');
      return;
    }
    bone.style.setProperty('--from-x', String(from.width / box.width));
    bone.style.setProperty('--from-y', String(from.height / box.height));
  }, [state]);

  if (!hasChildren) {
    return <Bone width={width} height={height} radius={radius} inline={inline} className={props.className} />;
  }

  const Tag = inline ? 'span' : 'div';
  const onBoneEnd = (event: AnimationEvent<HTMLElement>): void => {
    if (event.target === event.currentTarget) setSlot('ready');
  };
  const setBone = (node: HTMLElement | null): void => {
    boneRef.current = node;
  };
  const setSlotNode = (node: HTMLElement | null): void => {
    slotRef.current = node;
  };

  return (
    <Tag ref={setSlotNode} className={cn(styles.slot, props.className)} data-state={state}>
      {state === 'loading' ? (
        <Bone width={width} height={height} radius={radius} inline={inline} boneRef={setBone} />
      ) : null}
      {state === 'settling' ? (
        <Bone
          width="auto"
          height="auto"
          radius={radius}
          inline={inline}
          settling
          step={clampStep(props.step)}
          boneRef={setBone}
          onAnimationEnd={onBoneEnd}
        />
      ) : null}
      {state === 'loading' ? null : props.children}
    </Tag>
  );
}

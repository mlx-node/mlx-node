import type { CSSProperties, ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import { hoverTitle } from '../../../lib/hover-title';
import { truncateMiddle } from '../../../lib/labels';
import { modelColor } from '../../../lib/model-colors';
import { Badge } from '../../arc/badge/badge';
import { MiddleText } from '../middle-text/middle-text';
import styles from './model-badge.module.css';
import { ModelLogo } from './model-logos';

export interface ModelBadgeProps {
  name: string;
  /** Color registry key when it is not the printed name: a checkpoint FOLDER is colored by the id it is served and traced under. Default: `name`. */
  colorKey?: string;
  /** Small outline badge after the name. */
  quant?: string | null;
  /** 11 px / 13 px text. Default 'md'. */
  size?: 'sm' | 'md';
  /** Family mark, 14 px. */
  showLogo?: boolean;
  to?: string;
  /** Middle-truncate past this many characters. Default 28. Ignored with `fit="width"`. */
  maxChars?: number;
  /** 'chars' (default): cut at `maxChars`. 'width': the whole name when it fits, else cut in the middle by the width. */
  fit?: 'chars' | 'width';
  className?: string;
}

function titleWhenClipped(node: HTMLElement): void {
  node.removeAttribute('title');
  const title = hoverTitle(node);
  if (title !== null) node.setAttribute('title', title);
}

/** The one way a page prints a model name next to data: the dot always matches the model's color in every chart. */
export function ModelBadge({
  name,
  colorKey,
  quant,
  size = 'md',
  showLogo = false,
  to,
  maxChars = 28,
  fit = 'chars',
  className,
}: ModelBadgeProps): ReactElement {
  const color = modelColor(colorKey ?? name);
  const shown = truncateMiddle(name, maxChars);
  const dotStyle = { '--model-color': color.color } as CSSProperties;
  const cut = shown !== name;
  const label =
    fit === 'width' ? (
      <MiddleText text={name} />
    ) : (
      <span
        className={styles.name}
        title={cut ? name : undefined}
        data-clip-title={cut ? undefined : name}
        onPointerEnter={cut ? undefined : (event) => titleWhenClipped(event.currentTarget)}
      >
        {shown}
      </span>
    );
  return (
    <span className={cn(styles.root, className)} data-size={size} data-model={name}>
      <span className={styles.dot} data-pattern={color.pattern} style={dotStyle} aria-hidden="true" />
      {showLogo ? <ModelLogo model={name} className={styles.logo} /> : null}
      {to !== undefined ? (
        <Link className={styles.link} to={to}>
          {label}
        </Link>
      ) : (
        label
      )}
      {quant !== undefined && quant !== null && quant.length > 0 ? (
        <Badge tone="neutral" size="sm" outline className={styles.quant}>
          {quant}
        </Badge>
      ) : null}
    </span>
  );
}

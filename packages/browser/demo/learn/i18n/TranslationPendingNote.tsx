import * as React from 'react';

/**
 * Shown at the top of a /zh chapter page whose body has no Chinese translation
 * yet (the English body renders below it). Removed automatically once the
 * chapter is registered in bodies.zh.ts.
 */
export function TranslationPendingNote() {
  return (
    <div
      lang="zh-CN"
      className="not-prose my-5 rounded-md border border-border bg-muted/40 px-3 py-2 text-[13px] text-muted-foreground"
    >
      本章的中文翻译正在进行中，以下内容暂以英文显示。
      <span lang="en"> · This chapter has not been translated yet; the English version is shown below.</span>
    </div>
  );
}

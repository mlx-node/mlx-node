import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vite-plus/test';

import { PREFILL_TPS_SQL } from '../src/api/sql.js';
import { PREFILL_MIN_TOKENS, prefillRate } from '../src/prefill-rule.js';

describe('prefill rule', () => {
  it('rates a prefill from the floor up', () => {
    expect(PREFILL_MIN_TOKENS).toBe(256);
    expect(prefillRate(255, 100)).toBeNull();
    expect(prefillRate(256, 100)).toBe(2560);
  });

  it('applies the floor per call for a sum over calls', () => {
    expect(prefillRate(600, 300, 3)).toBeNull();
    expect(prefillRate(600, 300, 2)).toBe(2000);
  });

  it('gives no rate for a missing or empty window', () => {
    expect(prefillRate(null, 100)).toBeNull();
    expect(prefillRate(400, null)).toBeNull();
    expect(prefillRate(400, 0)).toBeNull();
    expect(prefillRate(400, 100, 0)).toBeNull();
  });

  it('agrees with the SQL fragment at the floor', () => {
    const db = new DatabaseSync(':memory:');
    db.exec('CREATE TABLE traces (prefill_ms REAL, prefill_tokens INTEGER)');
    const insert = db.prepare('INSERT INTO traces (prefill_ms, prefill_tokens) VALUES (?, ?)');
    insert.run(100, 255);
    insert.run(100, 256);
    const rates = (
      db.prepare(`SELECT ${PREFILL_TPS_SQL} AS r FROM traces ORDER BY prefill_tokens`).all() as Array<{
        r: number | null;
      }>
    ).map((row) => row.r);
    expect(rates).toEqual([prefillRate(255, 100), prefillRate(256, 100)]);
    expect(rates).toEqual([null, 2560]);
    db.close();
  });
});

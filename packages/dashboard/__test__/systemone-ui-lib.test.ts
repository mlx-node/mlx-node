import { describe, expect, it } from 'vite-plus/test';

import {
  buildAutoDetectPayload,
  buildCurlCommand,
  buildNodeSdkSnippet,
  buildPythonSnippet,
  buildSuggestedFields,
  buildTypeScriptSnippet,
  compareExpectedAnswer,
  createBatchRow,
  DEFAULT_DECISION_POLICY,
  evaluateClairicAnswerRoute,
  evaluateClefAnswerRoute,
  exportBatchToCsv,
  normalizePolicy,
  parseBatchImportText,
  scanNumbersInText,
  summarizeRouting,
  type ClefAnswer,
  type DecisionPolicy,
  type MediaAttachment,
} from '../ui/src/lib/system-one.js';

describe('scanNumbersInText', () => {
  it('parses integers, negative values, decimals, and comma-grouped numbers', () => {
    const text = 'Revenue: $1,440.50, delta -12.5, units 250, total 1,234,567';
    expect(scanNumbersInText(text)).toEqual([1440.5, -12.5, 250, 1234567]);
  });

  it('skips ISO dates and clock times so years and timestamps do not skew bounds', () => {
    const text = 'Measured on 2026-07-01 at 13:34:00 (and 09:15): Mon 118, Tue 124, Wed 131.5. Next check 2026-07-08.';
    expect(scanNumbersInText(text)).toEqual([118, 124, 131.5]);
  });

  it('skips slash-formatted dates and letter-prefixed identifiers (including hyphenated codes like INV-2026)', () => {
    const text = 'INVOICE #INV-2026-0842 Batch v2 on 07/01/2026 had item42 quantity 88 and 95';
    expect(scanNumbersInText(text)).toEqual([88, 95]);
  });

  it('returns an empty list when no standalone numbers exist', () => {
    expect(scanNumbersInText('Date 2026-07-01 at 08:30:00 no metrics')).toEqual([]);
  });
});

describe('buildAutoDetectPayload & buildSuggestedFields', () => {
  it('builds a 4-field auto-detect payload without observed_min or observed_max', () => {
    const img: MediaAttachment = {
      id: 'img-1',
      name: 'receipt.png',
      contentType: 'image/png',
      sizeBytes: 1200,
      base64: 'AAA',
      dataUrl: 'data:image/png;base64,AAA',
      rotation: 90,
      kind: 'image',
    };
    const pdf: MediaAttachment = {
      id: 'pdf-1',
      name: 'invoice.pdf',
      contentType: 'application/pdf',
      sizeBytes: 2400,
      base64: 'BBB',
      dataUrl: 'data:application/pdf;base64,BBB',
      rotation: 0,
      kind: 'pdf',
    };

    const payload = buildAutoDetectPayload({ stateText: '', attachments: [img, pdf] });
    expect(payload.state).toBe('Classify the attached document or image.');
    const fields = payload.fields as Record<string, unknown>;
    expect(Object.keys(fields).sort()).toEqual(['document_type', 'has_dates', 'has_list_items', 'has_money']);
    expect(fields).not.toHaveProperty('observed_min');
    expect(fields).not.toHaveProperty('observed_max');
    expect(payload.images).toEqual([{ content_type: 'image/png', base64: 'AAA', rotation: 90 }]);
    expect(payload.documents).toEqual([{ content_type: 'application/pdf', base64: 'BBB' }]);
  });

  it('computes tight forecast bounds from sourceText while ignoring ISO dates and times', () => {
    const sourceText = ['Recorded 2026-07-01 13:34:00', 'Mon: 100', 'Tue: 120', 'Wed: 140', 'Thu: 150'].join('\n');

    const suggested = buildSuggestedFields(
      {
        answers: {
          document_type: { value: 'time_series_forecast' },
          has_dates: { value: true },
          has_money: { value: false },
          has_list_items: { value: false },
        },
      },
      sourceText,
    );

    expect(suggested.categoryLabel).toBe('time series');
    const estimate = suggested.fields.find((f) => f.type === 'estimate');
    const forecast = suggested.fields.find((f) => f.type === 'forecast');
    expect(estimate).toBeDefined();
    expect(forecast).toBeDefined();
    // min=100, max=150, span=50 -> low=floor(100-20)=80, high=ceil(150+25)=175
    expect(estimate?.min).toBe('80');
    expect(estimate?.max).toBe('175');
    expect(forecast?.min).toBe('80');
    expect(forecast?.max).toBe('175');
    // Should not append a stray date field for time_series_forecast
    expect(suggested.fields.some((f) => f.type === 'date')).toBe(false);
  });

  it('augments non-forecast categories with detected dates, money, and list items when absent', () => {
    const suggested = buildSuggestedFields({
      answers: {
        document_type: { value: 'article_or_notes' },
        has_dates: { value: true },
        has_money: { value: true },
        has_list_items: { value: true },
      },
    });
    expect(suggested.categoryLabel).toBe('article / notes');
    const names = suggested.fields.map((f) => f.name);
    expect(names).toContain('date');
    expect(names).toContain('total_amount');
    // article_or_notes already includes list fields (key_points, entities), so no duplicate generic items field
    expect(names).not.toContain('items');
  });
});

describe('decision policy & confidence routing', () => {
  const policy: DecisionPolicy = {
    minConfidence: 0.85,
    yesThreshold: 0.75,
    noThreshold: 0.25,
  };

  it('normalizes out-of-range and inverted thresholds', () => {
    expect(normalizePolicy({ minConfidence: 1.5, noThreshold: 0.8, yesThreshold: 0.4 })).toEqual({
      minConfidence: 1,
      noThreshold: 0.6000000000000001,
      yesThreshold: 0.6000000000000001,
    });
    expect(normalizePolicy({ minConfidence: -0.2, noThreshold: -1, yesThreshold: 2 })).toEqual({
      minConfidence: 0,
      noThreshold: 0,
      yesThreshold: 1,
    });
  });

  it('routes CLEF noul answers across all 3 zones (auto-yes / review / auto-no)', () => {
    const yes = evaluateClefAnswerRoute({ type: 'noul', noul: 0.82 }, policy);
    expect(yes.route).toBe('auto-yes');
    expect(yes.label).toBe('Auto-Yes');
    expect(yes.tone).toBe('success');
    expect(yes.confidence).toBeCloseTo(0.82);

    const exactYes = evaluateClefAnswerRoute({ type: 'noul', noul: 0.75 }, policy);
    expect(exactYes.route).toBe('auto-yes');

    const review = evaluateClefAnswerRoute({ type: 'noul', noul: 0.55 }, policy);
    expect(review.route).toBe('review');
    expect(review.label).toBe('Needs review');
    expect(review.tone).toBe('warning');
    expect(review.confidence).toBeCloseTo(0.55);

    const no = evaluateClefAnswerRoute({ type: 'noul', noul: 0.12 }, policy);
    expect(no.route).toBe('auto-no');
    expect(no.label).toBe('Auto-No');
    expect(no.tone).toBe('neutral');
    expect(no.confidence).toBeCloseTo(0.88);

    const exactNo = evaluateClefAnswerRoute({ type: 'noul', noul: 0.25 }, policy);
    expect(exactNo.route).toBe('auto-no');
  });

  it('gates CLEF choice and score answers on minConfidence', () => {
    const choiceAuto: ClefAnswer = {
      type: 'choice',
      choice: 'gpu_kernels',
      probabilities: { gpu_kernels: 0.91, product_backend: 0.09 },
      confidence: 0.91,
    };
    const choiceReview: ClefAnswer = {
      type: 'choice',
      choice: 'gpu_kernels',
      probabilities: { gpu_kernels: 0.62, product_backend: 0.38 },
      confidence: 0.62,
    };
    const scoreAuto: ClefAnswer = {
      type: 'score',
      score: 3.6,
      probabilities: { '3': 0.4, '4': 0.6 },
      legend: { '3': 'Strong', '4': 'Exceptional' },
      confidence: 0.88,
    };
    const scoreReview: ClefAnswer = {
      ...scoreAuto,
      confidence: 0.7,
    };

    expect(evaluateClefAnswerRoute(choiceAuto, policy).route).toBe('auto');
    expect(evaluateClefAnswerRoute(choiceReview, policy).route).toBe('review');
    expect(evaluateClefAnswerRoute(scoreAuto, policy).route).toBe('auto');
    expect(evaluateClefAnswerRoute(scoreReview, policy).route).toBe('review');
  });

  it('routes Clairic answers by confidence, validity, and p_null verification', () => {
    expect(evaluateClairicAnswerRoute({ value: 'Northstar', confidence: 0.96 }, policy)).toMatchObject({
      route: 'auto',
      label: 'Auto-accept',
      tone: 'success',
      isNull: false,
    });

    expect(evaluateClairicAnswerRoute({ value: 'Northstar', confidence: 0.62 }, policy)).toMatchObject({
      route: 'review',
      label: 'Needs review',
      tone: 'warning',
    });

    expect(evaluateClairicAnswerRoute({ value: 'bad-date', valid: false, confidence: 0.99 }, policy)).toMatchObject({
      route: 'review',
      label: 'Needs review',
    });

    expect(evaluateClairicAnswerRoute({ value: 'partial', complete: false, confidence: 0.99 }, policy)).toMatchObject({
      route: 'review',
      label: 'Needs review',
    });

    expect(evaluateClairicAnswerRoute({ value: null, p_null: 0.93, confidence: 0.4 }, policy)).toMatchObject({
      route: 'missing',
      label: 'Missing',
      tone: 'neutral',
      isNull: true,
    });

    expect(evaluateClairicAnswerRoute({ value: null, p_null: 0.55, confidence: 0.9 }, policy)).toMatchObject({
      route: 'review',
      label: 'Verify missing',
      tone: 'warning',
      isNull: true,
    });
  });

  it('summarizes routing counts across CLEF and Clairic results', () => {
    const clefResult = {
      answers: {
        advance: { type: 'noul', noul: 0.9 },
        borderline: { type: 'noul', noul: 0.52 },
        reject: { type: 'noul', noul: 0.1 },
        track: { type: 'choice', choice: 'gpu', probabilities: { gpu: 0.7 }, confidence: 0.7 },
      },
    };
    expect(summarizeRouting(clefResult, policy)).toEqual({
      autoCount: 2,
      reviewCount: 2,
      missingCount: 0,
      totalCount: 4,
    });

    const clairicResult = {
      answers: {
        vendor: { value: 'Acme', confidence: 0.95 },
        total: { value: 42.5, confidence: 0.6 },
        po: { value: null, p_null: 0.92 },
        notes: { value: null, p_null: 0.4 },
      },
    };
    expect(summarizeRouting(clairicResult, policy)).toEqual({
      autoCount: 1,
      reviewCount: 2,
      missingCount: 1,
      totalCount: 4,
    });

    expect(summarizeRouting(undefined, DEFAULT_DECISION_POLICY)).toEqual({
      autoCount: 0,
      reviewCount: 0,
      missingCount: 0,
      totalCount: 0,
    });
  });
});

describe('parseBatchImportText', () => {
  it('parses a JSON array of objects and plain strings', () => {
    const json = JSON.stringify([
      { label: 'Ticket A', input: 'Webhooks down', expected: 'critical' },
      { name: 'Ticket B', state: { customer: 'Acme', tier: 'pro' }, ground_truth: 'high' },
      'Plain string case',
    ]);
    const parsed = parseBatchImportText(json);
    expect(parsed).toHaveLength(3);
    expect(parsed[0]).toEqual({ label: 'Ticket A', stateText: 'Webhooks down', expected: 'critical' });
    expect(parsed[1]?.label).toBe('Ticket B');
    expect(parsed[1]?.expected).toBe('high');
    expect(JSON.parse(parsed[1]!.stateText)).toEqual({ customer: 'Acme', tier: 'pro' });
    expect(parsed[2]).toEqual({ label: 'Case 3', stateText: 'Plain string case', expected: '' });
  });

  it('parses JSONL with one JSON object per line', () => {
    const jsonl = [
      '{"id":"c1","text":"First document","target":"yes"}',
      '{"id":"c2","document":"Second document","expected":"no"}',
    ].join('\r\n');
    expect(parseBatchImportText(jsonl)).toEqual([
      { label: 'c1', stateText: 'First document', expected: 'yes' },
      { label: 'c2', stateText: 'Second document', expected: 'no' },
    ]);
  });

  it('parses RFC-4180 CSV with quoted commas, escaped quotes, and embedded newlines', () => {
    const csv = [
      'label,input,expected',
      '"Case, 1","Line one, with ""quotes""\nand line two","critical"',
      'Case 2,Simple ticket text,low',
    ].join('\n');
    expect(parseBatchImportText(csv)).toEqual([
      { label: 'Case, 1', stateText: 'Line one, with "quotes"\nand line two', expected: 'critical' },
      { label: 'Case 2', stateText: 'Simple ticket text', expected: 'low' },
    ]);
  });

  it('falls back to plain newline-separated entries', () => {
    const plain = 'First unstructured item\n\nSecond unstructured item\n';
    expect(parseBatchImportText(plain)).toEqual([
      { label: 'Case 1', stateText: 'First unstructured item', expected: '' },
      { label: 'Case 2', stateText: 'Second unstructured item', expected: '' },
    ]);
  });
});

describe('compareExpectedAnswer', () => {
  const policy: DecisionPolicy = {
    minConfidence: 0.85,
    yesThreshold: 0.75,
    noThreshold: 0.25,
  };

  it('returns null when expected is empty or whitespace', () => {
    expect(compareExpectedAnswer({ value: 'foo' }, '   ', policy)).toBeNull();
  });

  it('compares noul answers against yes, no, and review using active policy thresholds', () => {
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.9 }, 'yes', policy)).toBe(true);
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.6 }, 'yes', policy)).toBe(false);
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.6 }, 'review', policy)).toBe(true);
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.1 }, 'no', policy)).toBe(true);
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.1 }, 'false', policy)).toBe(true);
    expect(compareExpectedAnswer({ type: 'noul', noul: 0.1 }, 'review', policy)).toBe(false);
  });

  it('compares choice and score answers cleanly', () => {
    const choice: ClefAnswer = {
      type: 'choice',
      choice: 'data_pipelines',
      probabilities: { data_pipelines: 0.92 },
      confidence: 0.92,
    };
    expect(compareExpectedAnswer(choice, 'DATA_PIPELINES', policy)).toBe(true);
    expect(compareExpectedAnswer(choice, 'billing', policy)).toBe(false);

    const score: ClefAnswer = {
      type: 'score',
      score: 3.12,
      probabilities: { '3': 0.88 },
      legend: { '3': 'Strong fit' },
      confidence: 0.88,
    };
    expect(compareExpectedAnswer(score, '3', policy)).toBe(true);
    expect(compareExpectedAnswer(score, '3.12', policy)).toBe(true);
    expect(compareExpectedAnswer(score, 'strong fit', policy)).toBe(true);
    expect(compareExpectedAnswer(score, '4', policy)).toBe(false);
  });

  it('compares scalar numbers, strings, and null/missing values', () => {
    expect(compareExpectedAnswer({ value: 258.18 }, '258.180', policy)).toBe(true);
    expect(compareExpectedAnswer({ value: 258.18 }, '259.00', policy)).toBe(false);
    expect(compareExpectedAnswer({ value: 'Critical' }, 'critical', policy)).toBe(true);
    expect(compareExpectedAnswer({ value: null }, 'missing', policy)).toBe(true);
    expect(compareExpectedAnswer({ value: null }, 'null', policy)).toBe(true);
    expect(compareExpectedAnswer({ value: null }, 'PO-123', policy)).toBe(false);
  });
});

describe('exportBatchToCsv', () => {
  it('escapes RFC-4180 special characters and includes live routing columns', () => {
    const rows = [
      createBatchRow({
        label: 'Invoice, "A"',
        stateText: 'Vendor: Acme\nTotal: $12.50',
        expected: 'Acme',
        status: 'done',
        elapsedMs: 142.6,
        result: {
          answers: {
            vendor: { value: 'Acme, Inc.', confidence: 0.95 },
            po: { value: null, p_null: 0.5 },
          },
        },
      }),
      createBatchRow({
        label: 'Pending case',
        stateText: 'Not run yet',
        expected: '',
        status: 'idle',
        elapsedMs: null,
      }),
    ];

    const csv = exportBatchToCsv({
      rows,
      columns: ['vendor', 'po'],
      mode: 'extract',
      policy: { minConfidence: 0.85, yesThreshold: 0.75, noThreshold: 0.25 },
    });

    const expectedLines = [
      'case,input,expected,status,latency_ms,vendor,vendor_route,po,po_route',
      '"Invoice, ""A""","Vendor: Acme\nTotal: $12.50",Acme,done,143,"Acme, Inc.",Auto-accept,null,Verify missing',
      'Pending case,Not run yet,,idle,,,,,',
    ];
    expect(csv).toBe(expectedLines.join('\n'));
  });

  it('exports CLEF routing labels in decide mode', () => {
    const rows = [
      createBatchRow({
        label: 'Candidate 1',
        stateText: '{"applicant":"Avery"}',
        expected: 'yes',
        status: 'done',
        elapsedMs: 88,
        result: {
          answers: {
            advance: { type: 'noul', noul: 0.6 },
          },
        },
      }),
    ];

    const csv = exportBatchToCsv({
      rows,
      columns: ['advance'],
      mode: 'decide',
      policy: { minConfidence: 0.85, yesThreshold: 0.75, noThreshold: 0.25 },
    });

    expect(csv).toContain('true (60%),Needs review');
  });
});

describe('buildTypeScriptSnippet', () => {
  const customPolicy: DecisionPolicy = {
    minConfidence: 0.91,
    yesThreshold: 0.82,
    noThreshold: 0.18,
  };

  it('embeds live thresholds and 3-zone routing constants in Decide mode', () => {
    const code = buildTypeScriptSnippet({
      model: 'clef-small',
      mode: 'decide',
      policy: customPolicy,
      payload: {
        state: { applicant: 'Avery' },
        questions: {
          advance: { type: 'noul', instructions: 'Advance candidate?' },
          track: { type: 'choice', criteria: { gpu: 'GPU' } },
        },
      },
    });

    expect(code).toContain('const YES_THRESHOLD = 0.82;');
    expect(code).toContain('const NO_THRESHOLD = 0.18;');
    expect(code).toContain('const MIN_CONFIDENCE = 0.91;');
    expect(code).toContain('answers["advance"].noul');
    expect(code).toContain('answers["track"]');
    expect(code).toContain("prob >= YES_THRESHOLD ? 'auto_yes' : prob <= NO_THRESHOLD ? 'auto_no' : 'human_review'");
  });

  it('embeds live MIN_CONFIDENCE threshold and review filter in Extract mode', () => {
    const longBase64 = 'A'.repeat(120);
    const code = buildTypeScriptSnippet({
      model: 'clairic-v1',
      mode: 'extract',
      policy: customPolicy,
      payload: {
        state: 'Invoice #100',
        images: [{ content_type: 'image/png', base64: longBase64 }],
        fields: { total: { type: 'number' } },
      },
    });

    expect(code).toContain('const MIN_CONFIDENCE = 0.91;');
    expect(code).toContain('ans.p_null ?? ans.confidence ?? 1');
    expect(code).toContain('...<base64>');
    expect(code).not.toContain(longBase64);
  });

  it('escapes newlines in question IDs so comments cannot inject executable code', () => {
    const code = buildTypeScriptSnippet({
      model: 'clef-small',
      mode: 'decide',
      policy: customPolicy,
      payload: {
        state: 'hello',
        questions: {
          'ok\nprocess.exit(1)': { type: 'noul' },
        },
      },
    });
    expect(code).not.toContain('\nprocess.exit(1)');
    expect(code).toContain('// Calibrated 3-zone decision routing for "ok\\nprocess.exit(1)"');
  });
});

describe('edge-case hardening (number scan, negative forecasts, CSV safety & estimates)', () => {
  it('splits tight comma-separated series like 100,110,120 while preserving $1,440.50 and 1,234,567', () => {
    expect(scanNumbersInText('Series: 100,110,120')).toEqual([100, 110, 120]);
    expect(scanNumbersInText('Total $1,440.50 and 1,234,567')).toEqual([1440.5, 1234567]);
  });

  it('computes valid negative forecast bounds (min < max) for negative series', () => {
    const suggested = buildSuggestedFields(
      { answers: { document_type: { value: 'time_series_forecast' } } },
      'Temps: -30, -28, -25',
    );
    const est = suggested.fields.find((f) => f.type === 'estimate');
    expect(est).toBeDefined();
    expect(Number(est!.min)).toBeLessThan(Number(est!.max));
    expect(Number(est!.min)).toBeLessThanOrEqual(-30);
    expect(Number(est!.max)).toBeGreaterThanOrEqual(-25);
  });

  it('does not drop plain comma-containing lines as a CSV header and preserves JSON expected: null', () => {
    const plain = parseBatchImportText('first case, urgent\nsecond case');
    expect(plain).toHaveLength(2);
    expect(plain[0]!.stateText).toBe('first case, urgent');

    const jsonNull = parseBatchImportText(JSON.stringify([{ state: 'No PO', expected: null }]));
    expect(jsonNull).toHaveLength(1);
    expect(jsonNull[0]!.expected).toBe('null');
  });

  it('matches estimate answers against numeric value/median and neutralizes CSV formula injection', () => {
    expect(
      compareExpectedAnswer({ value: 20, mean: 20, median: 20, p10: 10, p90: 30 }, '20', DEFAULT_DECISION_POLICY),
    ).toBe(true);

    const row = createBatchRow({
      label: '=CMD()',
      stateText: '+1+1',
      expected: '@SUM(A1)',
      status: 'done',
      result: { answers: { total: { value: -12.5, confidence: 0.99 } } },
    });
    const csv = exportBatchToCsv({
      rows: [row],
      columns: ['total'],
      mode: 'extract',
      policy: DEFAULT_DECISION_POLICY,
    });
    expect(csv).toContain("'=CMD()");
    expect(csv).toContain("'+1+1");
    expect(csv).toContain("'@SUM(A1)");
    // Plain negative numeric value remains numeric without quote prefix
    expect(csv).toContain(',-12.5,Auto-accept');
  });
});

describe('API guide snippets (Python, Node SDK & custom endpointUrl)', () => {
  const policy: DecisionPolicy = { minConfidence: 0.88, yesThreshold: 0.8, noThreshold: 0.2 };

  it('uses custom live mlx serve port across cURL, TypeScript, and Python snippets', () => {
    const endpointUrl = 'http://127.0.0.1:9191/v1/systemone';
    const payload = { state: 'test', fields: { total: { type: 'number' } } };
    expect(buildCurlCommand('clairic-flash', payload, endpointUrl)).toContain(endpointUrl);
    expect(buildTypeScriptSnippet({ model: 'clairic-flash', mode: 'extract', payload, policy, endpointUrl })).toContain(
      endpointUrl,
    );
    const py = buildPythonSnippet({ model: 'clairic-flash', mode: 'extract', payload, policy, endpointUrl });
    expect(py).toContain(endpointUrl);
    expect(py).toContain('MIN_CONFIDENCE = 0.88');
    expect(py).toContain('json.loads(');
  });

  it('generates Decide Python & Node SDK snippets with live thresholds and exact model class', () => {
    const payload = {
      state: { applicant: 'Avery' },
      questions: { advance: { type: 'noul' }, track: { type: 'choice', criteria: { gpu: 'GPU' } } },
    };
    const py = buildPythonSnippet({ model: 'clef-flash', mode: 'decide', payload, policy });
    expect(py).toContain('YES_THRESHOLD = 0.80');
    expect(py).toContain('NO_THRESHOLD = 0.20');
    expect(py).toContain('answers["advance"]["noul"]');
    expect(py).toContain('answers["track"]["confidence"]');

    const sdkDecide = buildNodeSdkSnippet({
      modelPath: '/models/clef-flash',
      mode: 'decide',
      payload,
      policy,
    });
    expect(sdkDecide).toContain("import { ClefModel } from '@mlx-node/lm';");
    expect(sdkDecide).toContain('ClefModel.load("/models/clef-flash")');
    expect(sdkDecide).toContain('model.decide(');

    const sdkExtract = buildNodeSdkSnippet({
      modelPath: '/models/clairic-flash',
      mode: 'extract',
      payload: { state: 'Invoice', fields: { total: { type: 'number' } } },
      policy,
    });
    expect(sdkExtract).toContain("import { ClairicModel } from '@mlx-node/lm';");
    expect(sdkExtract).toContain('model.extract(');
  });
});

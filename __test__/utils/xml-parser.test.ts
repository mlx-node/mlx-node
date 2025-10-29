import { describe, expect, it } from 'vitest';
import { extractHashAnswer, extractXmlAnswer, parseXmlCot } from '../../src/index';

describe('XML chain-of-thought parser', () => {
  it('parses MLX-style strict XML format with surrounding whitespace', () => {
    const sample = `
<reasoning>
Step 1: compute 1 + 2 = 3.
Step 2: multiply by 4 to get 12.
</reasoning>
<answer>
12
</answer>
`.trim();

    const result = parseXmlCot(sample);

    expect(result.isStrictMatch).toBe(true);
    expect(result.isSoftMatch).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.reasoning).toBe('Step 1: compute 1 + 2 = 3.\nStep 2: multiply by 4 to get 12.');
    expect(result.answer).toBe('12');
  });

  it('handles TRL reward-pattern format with embedded newlines', () => {
    const sample = `<reasoning>
We think carefully.
</reasoning>
<answer>
42
</answer>`;

    const result = parseXmlCot(sample);
    expect(result.isStrictMatch).toBe(true);
    expect(result.isSoftMatch).toBe(true);
    expect(result.reasoning).toBe('We think carefully.');
    expect(result.answer).toBe('42');
  });

  it('reports soft match when extra characters wrap the XML sections', () => {
    const sample = `Preamble text
<reasoning>Reasoning here</reasoning>
<answer>24</answer>
Postscript.`;

    const result = parseXmlCot(sample);
    expect(result.isStrictMatch).toBe(false);
    expect(result.isSoftMatch).toBe(true);
    expect(result.errors).toEqual(
      expect.arrayContaining([
        'XML format contains extra characters before <reasoning> section.',
        'XML format contains extra characters after </answer> section.',
      ]),
    );
    expect(result.reasoning).toBe('Reasoning here');
    expect(result.answer).toBe('24');
  });

  it('extracts reasoning even when the answer section is missing', () => {
    const sample = `<reasoning>We tried.</reasoning>`;
    const result = parseXmlCot(sample);
    expect(result.isSoftMatch).toBe(false);
    expect(result.reasoning).toBe('We tried.');
    expect(result.answer).toBeNull();
    expect(result.errors).toContain('Missing <answer>...</answer> section.');
  });

  it('flags non-whitespace text between reasoning and answer sections', () => {
    const sample = `<reasoning>Plan</reasoning>---<answer>Result</answer>`;
    const result = parseXmlCot(sample);
    expect(result.isStrictMatch).toBe(false);
    expect(result.isSoftMatch).toBe(true);
    expect(result.errors).toContain('XML format contains extra characters between reasoning and answer sections.');
  });

  it('extracts answer even when reasoning section is missing', () => {
    const sample = `<answer>7</answer>`;
    const result = parseXmlCot(sample);
    expect(result.isSoftMatch).toBe(false);
    expect(result.answer).toBe('7');
    expect(result.reasoning).toBeNull();
    expect(result.errors).toContain('Missing <reasoning>...</reasoning> section.');
  });

  it('flags unterminated sections', () => {
    const sample = `<reasoning>half open<answer>42</answer>`;
    const result = parseXmlCot(sample);
    expect(result.isSoftMatch).toBe(false);
    expect(result.errors).toContain('Unterminated <reasoning>...</reasoning> section.');
  });

  it('extracts hash-separated GSM8K answers', () => {
    expect(extractHashAnswer('Work #### 17')).toBe('17');
    expect(extractHashAnswer('No separator here')).toBeNull();
    expect(extractHashAnswer('Trailing ####   ')).toBeNull();
  });

  it('extracts XML answer via helper', () => {
    const sample = `<reasoning>Something</reasoning><answer>Value</answer>`;
    expect(extractXmlAnswer(sample)).toBe('Value');
  });
});

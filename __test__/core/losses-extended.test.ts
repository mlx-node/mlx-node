/**
 * Extended tests for implemented loss functions.
 * Mirrors the critical scenarios covered in MLX's test_losses.py
 * for the subset of losses currently exposed through the bindings.
 */

import { describe, it, expect } from 'vitest';
import { MxArray, Losses } from '@mlx-node/core';
import { createFloat32Array } from '../test-utils';

const logSumExp = (row: number[]): number => {
  const max = Math.max(...row);
  const sum = row.reduce((acc, value) => acc + Math.exp(value - max), 0);
  return max + Math.log(sum);
};

const mean = (row: number[]): number => row.reduce((acc, value) => acc + value, 0) / row.length;

const logSoftmax = (row: number[]): number[] => {
  const lse = logSumExp(row);
  return row.map((value) => value - lse);
};

describe('Cross Entropy - Extended', () => {
  it('matches manual computation for class indices', () => {
    const logitsData = [2.0, -1.0, -1.0, 2.0];
    const logits = createFloat32Array(logitsData, [2, 2]);
    const targets = MxArray.fromInt32(new Int32Array([0, 1]), BigInt64Array.from([2n]));

    const loss = Losses.crossEntropy(logits, targets, 2);
    const value = loss.toFloat32()[0];

    const logitsRows = [
      [2.0, -1.0],
      [-1.0, 2.0],
    ];
    const targetIdx = [0, 1];

    const manual =
      targetIdx.map((idx, i) => logSumExp(logitsRows[i]) - logitsRows[i][idx]).reduce((acc, x) => acc + x, 0) /
      targetIdx.length;

    expect(value).toBeCloseTo(manual, 5);
  });

  it('zeros ignored targets but keeps batch mean', () => {
    const logits = createFloat32Array([3.0, 0.0, 0.0, 3.0], [2, 2]);
    const targets = MxArray.fromInt32(new Int32Array([0, -1]), BigInt64Array.from([2n]));

    const loss = Losses.crossEntropy(logits, targets, 2, -1);
    const value = loss.toFloat32()[0];

    const logitsRows = [
      [3.0, 0.0],
      [0.0, 3.0],
    ];
    const manualLosses = [logSumExp(logitsRows[0]) - logitsRows[0][0], 0];
    const manual = manualLosses.reduce((acc, x) => acc + x, 0) / manualLosses.length;

    expect(value).toBeCloseTo(manual, 5);
  });

  it('supports probability targets', () => {
    const logitsRows = [
      [2.0, -1.0, 0.0],
      [-1.0, 2.0, 0.0],
    ];
    const logits = createFloat32Array([2.0, -1.0, 0.0, -1.0, 2.0, 0.0], [2, 3]);
    const probs = createFloat32Array([0.7, 0.2, 0.1, 0.1, 0.8, 0.1], [2, 3]);

    const loss = Losses.crossEntropy(logits, probs, 3);
    const value = loss.toFloat32()[0];

    const manual =
      logitsRows
        .map((row, i) => {
          const logProbs = logSoftmax(row);
          const targets = i === 0 ? [0.7, 0.2, 0.1] : [0.1, 0.8, 0.1];
          return -targets.reduce((acc, p, j) => acc + p * logProbs[j], 0);
        })
        .reduce((acc, x) => acc + x, 0) / logitsRows.length;

    expect(value).toBeCloseTo(manual, 5);
  });

  it('applies label smoothing like the reference implementation', () => {
    const smoothing = 0.2;
    const logitsRows = [
      [2.0, -1.0],
      [-1.0, 2.0],
    ];
    const logits = createFloat32Array([2.0, -1.0, -1.0, 2.0], [2, 2]);
    const targets = MxArray.fromInt32(new Int32Array([0, 1]), BigInt64Array.from([2n]));

    const loss = Losses.crossEntropy(logits, targets, 2, undefined, smoothing);
    const value = loss.toFloat32()[0];

    const targetIndices = targets.toInt32();
    const manual =
      logitsRows
        .map((row, i) => {
          const targetIdx = targetIndices[i];
          const logsumexp = logSumExp(row);
          const adjustedScore = (1 - smoothing) * row[targetIdx];
          const smoothTerm = smoothing * mean(row);
          return logsumexp - adjustedScore - smoothTerm;
        })
        .reduce((acc, x) => acc + x, 0) / logitsRows.length;

    expect(value).toBeCloseTo(manual, 5);
  });
});

describe('KL Divergence', () => {
  it('matches manual KL(P || Q) for log probability inputs', () => {
    const pLogits = [[2.0, 0.5, -1.0]];
    const qLogits = [[1.0, 0.0, -0.5]];

    const logP = createFloat32Array(logSoftmax(pLogits[0]), [1, 3]);
    const logQ = createFloat32Array(logSoftmax(qLogits[0]), [1, 3]);

    const kl = Losses.klDivergence(logP, logQ);
    const value = kl.toFloat32()[0];

    const manual = logSoftmax(pLogits[0])
      .map((lp, i) => {
        const lq = logSoftmax(qLogits[0])[i];
        const prob = Math.exp(lp);
        return prob * (lp - lq);
      })
      .reduce((acc, x) => acc + x, 0);

    expect(value).toBeCloseTo(manual, 5);
  });
});

describe('Mean Squared Error', () => {
  it('matches manual MSE across arbitrary shapes', () => {
    const predictions = createFloat32Array([0.5, 0.2, 0.9, 0.0, 0.3, 0.6], [2, 3]);
    const targets = createFloat32Array([0.7, 0.1, 0.8, 0.2, 0.4, 0.5], [2, 3]);

    const loss = Losses.mse(predictions, targets);
    const value = loss.toFloat32()[0];

    const predValues = predictions.toFloat32();
    const targetValues = targets.toFloat32();
    const manual =
      predValues
        .map((v, i) => {
          const diff = v - targetValues[i];
          return diff * diff;
        })
        .reduce((acc, x) => acc + x, 0) / predValues.length;

    expect(value).toBeCloseTo(manual, 5);
  });
});

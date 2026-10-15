# @mlx-node/core changelog

## Unreleased

### Breaking changes

Removed `MxArray` NAPI bindings that nothing in the repo used. Each one had no caller in `@mlx-node/lm`, `@mlx-node/trl`, `@mlx-node/vlm`, or any other workspace package:

- `MxArray#getBatchSeqLen()` and `MxArray#getBatchSeqHidden()`. Use `shape()` or `shapeAt(axis)` instead.
- `MxArray#evalAsync()`. Use `eval()` instead.
- `MxArray#tan()`, `sinh()`, `cosh()`, `arcsin()`, `arccos()`, `arctan()`.
- `MxArray.randomBernoulli(shape, prob)`. Use `MxArray.randomUniform(shape, 0, 1).less(MxArray.scalarFloat(prob))` instead.

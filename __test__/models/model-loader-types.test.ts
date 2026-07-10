import {
  GrpoTrainingEngine,
  Qwen3Model as NativeQwen3Model,
  Qwen35Model as NativeQwen35Model,
  Qwen35MoeModel as NativeQwen35MoeModel,
  SftTrainingEngine,
  type GrpoEngineConfig,
  type SftEngineConfig,
} from '@mlx-node/core';
import {
  loadModel,
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
  type LoadableModel,
  type TrainableModel,
} from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

type ExpectedTrainableModel =
  | Qwen3Model
  | Qwen35Model
  | Qwen35MoeModel
  | NativeQwen3Model
  | NativeQwen35Model
  | NativeQwen35MoeModel;

function registryTrainableToExpected(model: TrainableModel): ExpectedTrainableModel {
  return model;
}

function expectedTrainableToRegistry(model: ExpectedTrainableModel): TrainableModel {
  return model;
}

/**
 * Compile-time public-boundary probe. The function never runs; typechecking its
 * body proves that loadModel's wrapper union narrows to each trainable wrapper,
 * retains required native cache capabilities, and crosses the real native
 * trainer boundary after the same native instanceof guard used by @mlx-node/trl.
 */
function assertLoadModelTrainerBoundaries(
  model: Awaited<ReturnType<typeof loadModel>>,
  grpoConfig: GrpoEngineConfig,
  sftConfig: SftEngineConfig,
): void {
  const loadable: LoadableModel = model;
  void loadable;

  if (model instanceof Qwen3Model) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    void trainable;
    void paged;

    if (model instanceof NativeQwen3Model) {
      const native: NativeQwen3Model = model;
      const grpo = new GrpoTrainingEngine(model, grpoConfig);
      const sft = new SftTrainingEngine(model, sftConfig);
      void native;
      void grpo;
      void sft;
    }
    return;
  }

  if (model instanceof Qwen35Model) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    void trainable;
    void paged;

    if (model instanceof NativeQwen35Model) {
      const native: NativeQwen35Model = model;
      const grpo = GrpoTrainingEngine.fromQwen35(model, grpoConfig);
      const sft = SftTrainingEngine.fromQwen35(model, sftConfig);
      void native;
      void grpo;
      void sft;
    }
    return;
  }

  if (model instanceof Qwen35MoeModel) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    void trainable;
    void paged;

    if (model instanceof NativeQwen35MoeModel) {
      const native: NativeQwen35MoeModel = model;
      const grpo = GrpoTrainingEngine.fromQwen35Moe(model, grpoConfig);
      const sft = SftTrainingEngine.fromQwen35Moe(model, sftConfig);
      void native;
      void grpo;
      void sft;
    }
  }
}

describe('model loader compile-time contracts', () => {
  it('keeps registry-derived trainable and trainer-boundary probes in the typecheck graph', () => {
    expect(registryTrainableToExpected).toBeTypeOf('function');
    expect(expectedTrainableToRegistry).toBeTypeOf('function');
    expect(assertLoadModelTrainerBoundaries).toBeTypeOf('function');
  });
});

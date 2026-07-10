import {
  GrpoTrainingEngine,
  Qwen3Model as NativeQwen3Model,
  Qwen35Model as NativeQwen35Model,
  Qwen35MoeModel as NativeQwen35MoeModel,
  SftTrainingEngine,
  type Gemma4Model as NativeGemma4Model,
  type GrpoEngineConfig,
  type HarrierModel as NativeHarrierModel,
  type Lfm2Model as NativeLfm2Model,
  type QianfanOCRModel as NativeQianfanOCRModel,
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
 * Compile-time restored-contract probes (never run): every native
 * `@mlx-node/core` model instance stays assignable to the public
 * `LoadableModel` union, and each native trainable instance to
 * `TrainableModel`, so downstream code (e.g. @mlx-node/trl) can pass native
 * instances directly to the Rust engine factory methods without type
 * conflicts.
 */
type NativeCoreLoadable =
  | NativeQwen3Model
  | NativeQwen35Model
  | NativeQwen35MoeModel
  | NativeGemma4Model
  | NativeLfm2Model
  | NativeHarrierModel
  | NativeQianfanOCRModel;

function nativeCoreToLoadable(model: NativeCoreLoadable): LoadableModel {
  return model;
}

function nativeCoreToTrainable(model: NativeQwen3Model | NativeQwen35Model | NativeQwen35MoeModel): TrainableModel {
  return model;
}

/**
 * Compile-time public-boundary probe. The function never runs; typechecking
 * its body proves that loadModel's native-class union narrows under the same
 * native instanceof guards used by @mlx-node/trl, retains required native
 * cache capabilities, and crosses the real Rust trainer-engine boundary.
 */
function assertLoadModelTrainerBoundaries(
  model: Awaited<ReturnType<typeof loadModel>>,
  grpoConfig: GrpoEngineConfig,
  sftConfig: SftEngineConfig,
): void {
  const loadable: LoadableModel = model;
  void loadable;

  if (model instanceof NativeQwen3Model) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    const grpo = new GrpoTrainingEngine(model, grpoConfig);
    const sft = new SftTrainingEngine(model, sftConfig);
    void trainable;
    void paged;
    void grpo;
    void sft;
    return;
  }

  if (model instanceof NativeQwen35Model) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    const grpo = GrpoTrainingEngine.fromQwen35(model, grpoConfig);
    const sft = SftTrainingEngine.fromQwen35(model, sftConfig);
    void trainable;
    void paged;
    void grpo;
    void sft;
    return;
  }

  if (model instanceof NativeQwen35MoeModel) {
    const trainable: TrainableModel = model;
    const paged: boolean = model.hasBlockPagedCache();
    const grpo = GrpoTrainingEngine.fromQwen35Moe(model, grpoConfig);
    const sft = SftTrainingEngine.fromQwen35Moe(model, sftConfig);
    void trainable;
    void paged;
    void grpo;
    void sft;
  }
}

describe('model loader compile-time contracts', () => {
  it('keeps registry-derived trainable and trainer-boundary probes in the typecheck graph', () => {
    expect(registryTrainableToExpected).toBeTypeOf('function');
    expect(expectedTrainableToRegistry).toBeTypeOf('function');
    expect(nativeCoreToLoadable).toBeTypeOf('function');
    expect(nativeCoreToTrainable).toBeTypeOf('function');
    expect(assertLoadModelTrainerBoundaries).toBeTypeOf('function');
  });
});

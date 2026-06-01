import {
  createOnMessage as __wasmCreateOnMessageForFsProxy,
  getDefaultContext as __emnapiGetDefaultContext,
  instantiateNapiModule as __emnapiInstantiateNapiModule,
  WASI as __WASI,
} from '@napi-rs/wasm-runtime';

const __wasi = new __WASI({
  version: 'preview1',
});

const __wasmUrl = new URL('./mlx-core.wasm32-wasi.wasm', import.meta.url).href;
const __emnapiContext = __emnapiGetDefaultContext();

const __sharedMemory = new WebAssembly.Memory({
  initial: 16000,
  maximum: 65536,
  shared: true,
});

const __wasmFile = await fetch(__wasmUrl).then((res) => res.arrayBuffer());

const {
  instance: __napiInstance,
  module: __wasiModule,
  napiModule: __napiModule,
} = await __emnapiInstantiateNapiModule(__wasmFile, {
  context: __emnapiContext,
  asyncWorkPoolSize: 4,
  wasi: __wasi,
  onCreateWorker() {
    const worker = new Worker(new URL('./wasi-worker-browser.mjs', import.meta.url), {
      type: 'module',
    });

    return worker;
  },
  overwriteImports(importObject) {
    importObject.env = {
      ...importObject.env,
      ...importObject.napi,
      ...importObject.emnapi,
      memory: __sharedMemory,
    };
    return importObject;
  },
  beforeInit({ instance }) {
    for (const name of Object.keys(instance.exports)) {
      if (name.startsWith('__napi_register__')) {
        instance.exports[name]();
      }
    }
  },
});
export default __napiModule.exports;
export const BatchGenerationResult = __napiModule.exports.BatchGenerationResult;
export const ChatStreamHandle = __napiModule.exports.ChatStreamHandle;
export const DocLayoutModel = __napiModule.exports.DocLayoutModel;
export const PPDocLayoutV3Model = __napiModule.exports.PPDocLayoutV3Model;
export const DocOrientationModel = __napiModule.exports.DocOrientationModel;
export const DocUnwarpModel = __napiModule.exports.DocUnwarpModel;
export const Gemma4Model = __napiModule.exports.Gemma4Model;
export const GenerationResult = __napiModule.exports.GenerationResult;
export const HarrierModel = __napiModule.exports.HarrierModel;
export const Lfm2Model = __napiModule.exports.Lfm2Model;
export const MxArray = __napiModule.exports.MxArray;
export const QianfanOCRModel = __napiModule.exports.QianfanOCRModel;
export const Qwen35Model = __napiModule.exports.Qwen35Model;
export const Qwen3_5Model = __napiModule.exports.Qwen3_5Model;
export const Qwen35MoeModel = __napiModule.exports.Qwen35MoeModel;
export const Qwen3_5MoeModel = __napiModule.exports.Qwen3_5MoeModel;
export const Qwen3Model = __napiModule.exports.Qwen3Model;
export const Qwen3Tokenizer = __napiModule.exports.Qwen3Tokenizer;
export const Tensor = __napiModule.exports.Tensor;
export const TextDetModel = __napiModule.exports.TextDetModel;
export const TextRecModel = __napiModule.exports.TextRecModel;
export const TinyTrainer = __napiModule.exports.TinyTrainer;
export const VlmChatResult = __napiModule.exports.VlmChatResult;
export const VLMChatResult = __napiModule.exports.VLMChatResult;
export const VLModel = __napiModule.exports.VLModel;
export const buildRewardOutputs = __napiModule.exports.buildRewardOutputs;
export const ChatRole = __napiModule.exports.ChatRole;
export const convertForeignWeights = __napiModule.exports.convertForeignWeights;
export const convertModel = __napiModule.exports.convertModel;
export const createPaddleocrVlConfig = __napiModule.exports.createPaddleocrVlConfig;
export const createQianfanOcrConfig = __napiModule.exports.createQianfanOcrConfig;
export const createRandomQwen35Checkpoint = __napiModule.exports.createRandomQwen35Checkpoint;
export const createRandomQwen35MoeCheckpoint = __napiModule.exports.createRandomQwen35MoeCheckpoint;
export const createRandomQwen3Checkpoint = __napiModule.exports.createRandomQwen3Checkpoint;
export const DType = __napiModule.exports.DType;
export const ElementType = __napiModule.exports.ElementType;
export const formatDocument = __napiModule.exports.formatDocument;
export const memoryStats = __napiModule.exports.memoryStats;
export const OutputFormat = __napiModule.exports.OutputFormat;
export const parsePaddleResponse = __napiModule.exports.parsePaddleResponse;
export const parseToolCallsFromText = __napiModule.exports.parseToolCallsFromText;
export const parseVlmOutput = __napiModule.exports.parseVlmOutput;
export const webgpuScatterAddSelftest = __napiModule.exports.webgpuScatterAddSelftest;
export const wgpuGetGdnGCompileEnabled = __napiModule.exports.wgpuGetGdnGCompileEnabled;
export const wgpuGetGdnPostCompileEnabled = __napiModule.exports.wgpuGetGdnPostCompileEnabled;
export const wgpuGetGdnPreCompileEnabled = __napiModule.exports.wgpuGetGdnPreCompileEnabled;
export const wgpuGetLastTestError = __napiModule.exports.wgpuGetLastTestError;
export const wgpuGetSwigluCompileEnabled = __napiModule.exports.wgpuGetSwigluCompileEnabled;
export const wgpuSetGdnGCompileEnabled = __napiModule.exports.wgpuSetGdnGCompileEnabled;
export const wgpuSetGdnPostCompileEnabled = __napiModule.exports.wgpuSetGdnPostCompileEnabled;
export const wgpuSetGdnPreCompileEnabled = __napiModule.exports.wgpuSetGdnPreCompileEnabled;
export const wgpuSetPackedBf16Enabled = __napiModule.exports.wgpuSetPackedBf16Enabled;
export const wgpuSetSdpaFallbackForced = __napiModule.exports.wgpuSetSdpaFallbackForced;
export const wgpuSetSwigluCompileEnabled = __napiModule.exports.wgpuSetSwigluCompileEnabled;
export const wgpuTestCompileGdnGDispatchDelta = __napiModule.exports.wgpuTestCompileGdnGDispatchDelta;
export const wgpuTestCompileGdnPostDispatchDelta = __napiModule.exports.wgpuTestCompileGdnPostDispatchDelta;
export const wgpuTestCompileGdnPreDispatchDelta = __napiModule.exports.wgpuTestCompileGdnPreDispatchDelta;
export const wgpuTestCompileMlpDispatchDelta = __napiModule.exports.wgpuTestCompileMlpDispatchDelta;
export const __internal__ = __napiModule.exports.__internal__;

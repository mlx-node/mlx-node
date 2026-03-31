/**
 * MLX Web Worker with WebGPU bridge
 *
 * Each worker creates its own GPUDevice so spawn_blocking can run
 * GPU inference without blocking the main thread.
 */
import { instantiateNapiModuleSync, MessageHandler, WASI } from '@napi-rs/wasm-runtime';
import { createWebGPUBridge } from './webgpu-bridge.js';

// Pre-initialize WebGPU on this worker
let bridge = null;
const webgpuReady = (async () => {
  if (!navigator.gpu) return;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
      },
    });
    bridge = createWebGPUBridge(adapter, device);
    console.log('[Worker] WebGPU bridge ready');
  } catch (e) {
    console.warn('[Worker] WebGPU init failed:', e);
  }
})();

// No-op stubs for when WebGPU is unavailable
const WEBGPU_NOOP = {
  wgpuCreateInstance: () => 0, wgpuInstanceRequestAdapter: () => {},
  wgpuInstanceRelease: () => {}, wgpuAdapterRequestDevice: () => {},
  wgpuAdapterRelease: () => {}, wgpuDeviceSetUncapturedErrorCallback: () => {},
  wgpuDeviceSetDeviceLostCallback: () => {}, wgpuDeviceGetQueue: () => 0,
  mlx_webgpu_poll: () => {}, wgpuDeviceCreateComputePipeline: () => 0,
  wgpuComputePipelineGetBindGroupLayout: () => 0,
  wgpuDeviceCreateShaderModule: () => 0, wgpuQueueOnSubmittedWorkDone: () => {},
  wgpuAdapterGetProperties: () => {}, wgpuDeviceGetLimits: () => 0,
  wgpuCommandEncoderRelease: () => {}, wgpuComputePassEncoderEnd: () => {},
  wgpuComputePassEncoderRelease: () => {},
  wgpuDeviceCreateCommandEncoder: () => 0,
  wgpuCommandEncoderBeginComputePass: () => 0,
  wgpuComputePassEncoderSetPipeline: () => {},
  wgpuComputePassEncoderSetBindGroup: () => {},
  wgpuComputePassEncoderDispatchWorkgroups: () => {},
  wgpuCommandEncoderFinish: () => 0, wgpuQueueSubmit: () => {},
  wgpuCommandBufferRelease: () => {}, wgpuDeviceCreateBuffer: () => 0,
  wgpuBufferDestroy: () => {}, wgpuBufferRelease: () => {},
  wgpuCommandEncoderCopyBufferToBuffer: () => {},
  wgpuBufferMapAsync: () => {}, wgpuBufferGetConstMappedRange: () => 0,
  wgpuBufferUnmap: () => {}, wgpuBufferGetSize: () => 0n,
  wgpuBindGroupRelease: () => {}, wgpuDeviceCreateBindGroup: () => 0,
  wgpuBufferGetMappedRange: () => 0,
};

const handler = new MessageHandler({
  async onLoad({ wasmModule, wasmMemory }) {
    // Wait for WebGPU device to be ready
    await webgpuReady;

    const wasi = new WASI({
      print: function () { console.log.apply(console, arguments); },
      printErr: function () { console.error.apply(console, arguments); },
    });

    return instantiateNapiModuleSync(wasmModule, {
      childThread: true,
      wasi,
      overwriteImports(importObject) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: wasmMemory,
          __cxa_allocate_exception: () => 0,
          __cxa_throw: () => { throw new Error('C++ exception in worker'); },
          __cxa_init_primary_exception: (ptr) => ptr,
          _ZN3mlx4core3gpu4initEv: () => {},
          ...(bridge ? bridge.imports : WEBGPU_NOOP),
        };
      },
      beforeInit({ instance }) {
        if (bridge) bridge.setInstance(instance);
      },
    });
  },
});

globalThis.onmessage = function (e) {
  handler.handle(e);
};

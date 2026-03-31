import { instantiateNapiModuleSync, MessageHandler, WASI } from '@napi-rs/wasm-runtime';

const errorOutputs = [];

const handler = new MessageHandler({
  onLoad({ wasmModule, wasmMemory }) {
    const wasi = new WASI({
      print: function () {
        // eslint-disable-next-line no-console
        console.log.apply(console, arguments);
      },
      printErr: function () {
        // eslint-disable-next-line no-console
        console.error.apply(console, arguments);
      },
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
          __cxa_allocate_exception: (s) => 1, __cxa_throw: () => { throw new Error('[child worker] C++ exception'); }, __cxa_init_primary_exception: (ptr) => ptr, _ZN3mlx4core3gpu4initEv: () => {}, wgpuCreateInstance: () => 0, wgpuInstanceRequestAdapter: () => {}, wgpuInstanceRelease: () => {}, wgpuAdapterRequestDevice: () => {}, wgpuAdapterRelease: () => {}, wgpuDeviceSetUncapturedErrorCallback: () => {}, wgpuDeviceSetDeviceLostCallback: () => {}, wgpuDeviceGetQueue: () => 0, mlx_webgpu_poll: () => {}, wgpuDeviceCreateComputePipeline: () => 0, wgpuComputePipelineGetBindGroupLayout: () => 0, wgpuDeviceCreateShaderModule: () => 0, wgpuQueueOnSubmittedWorkDone: () => {}, wgpuAdapterGetProperties: () => {}, wgpuDeviceGetLimits: () => 0, wgpuCommandEncoderRelease: () => {}, wgpuComputePassEncoderEnd: () => {}, wgpuComputePassEncoderRelease: () => {}, wgpuDeviceCreateCommandEncoder: () => 0, wgpuCommandEncoderBeginComputePass: () => 0, wgpuComputePassEncoderSetPipeline: () => {}, wgpuComputePassEncoderSetBindGroup: () => {}, wgpuComputePassEncoderDispatchWorkgroups: () => {}, wgpuCommandEncoderFinish: () => 0, wgpuQueueSubmit: () => {}, wgpuCommandBufferRelease: () => {}, wgpuDeviceCreateBuffer: () => 0, wgpuBufferDestroy: () => {}, wgpuBufferRelease: () => {}, wgpuCommandEncoderCopyBufferToBuffer: () => {}, wgpuBufferMapAsync: () => {}, wgpuBufferGetConstMappedRange: () => 0, wgpuBufferUnmap: () => {}, wgpuBufferGetSize: () => 0n, wgpuBindGroupRelease: () => {}, wgpuDeviceCreateBindGroup: () => 0, wgpuBufferGetMappedRange: () => 0,
        };
      },
    });
  },
});

globalThis.onmessage = function (e) {
  handler.handle(e);
};

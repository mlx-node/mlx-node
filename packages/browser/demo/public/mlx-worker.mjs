import { instantiateNapiModuleSync, MessageHandler, WASI } from '@napi-rs/wasm-runtime';

const handler = new MessageHandler({
  onLoad({ wasmModule, wasmMemory }) {
    const wasi = new WASI({
      print: function () {
        console.log.apply(console, arguments);
      },
      printErr: function () {
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
          // WASM exception tag for C++ exceptions (used by -fwasm-exceptions)
          __cpp_exception: new WebAssembly.Tag({ parameters: ['i32'] }),
          // MLX GPU init — no-op (GPU pre-initialized on main thread)
          _ZN3mlx4core3gpu4initEv: () => {},
          // WebGPU stubs for worker threads (GPU ops dispatch from main thread)
          wgpuCreateInstance: () => 0,
          wgpuInstanceRequestAdapter: () => {},
          wgpuInstanceRelease: () => {},
          wgpuAdapterRequestDevice: () => {},
          wgpuAdapterRelease: () => {},
          wgpuDeviceSetUncapturedErrorCallback: () => {},
          wgpuDeviceSetDeviceLostCallback: () => {},
          wgpuDeviceGetQueue: () => 0,
          mlx_webgpu_poll: () => {},
          wgpuDeviceCreateComputePipeline: () => 0,
          wgpuComputePipelineGetBindGroupLayout: () => 0,
          wgpuDeviceCreateShaderModule: () => 0,
          wgpuQueueOnSubmittedWorkDone: () => {},
          wgpuAdapterGetProperties: () => {},
          wgpuDeviceGetLimits: () => 0,
          wgpuCommandEncoderRelease: () => {},
          wgpuComputePassEncoderEnd: () => {},
          wgpuComputePassEncoderRelease: () => {},
          wgpuDeviceCreateCommandEncoder: () => 0,
          wgpuCommandEncoderBeginComputePass: () => 0,
          wgpuComputePassEncoderSetPipeline: () => {},
          wgpuComputePassEncoderSetBindGroup: () => {},
          wgpuComputePassEncoderDispatchWorkgroups: () => {},
          wgpuCommandEncoderFinish: () => 0,
          wgpuQueueSubmit: () => {},
          wgpuCommandBufferRelease: () => {},
          wgpuDeviceCreateBuffer: () => 0,
          wgpuBufferDestroy: () => {},
          wgpuBufferRelease: () => {},
          wgpuCommandEncoderCopyBufferToBuffer: () => {},
          wgpuBufferMapAsync: () => {},
          wgpuBufferGetConstMappedRange: () => 0,
          wgpuBufferUnmap: () => {},
          wgpuBufferGetSize: () => 0n,
          wgpuBindGroupRelease: () => {},
          wgpuDeviceCreateBindGroup: () => 0,
          wgpuBufferGetMappedRange: () => 0,
        };
      },
    });
  },
});

globalThis.onmessage = function (e) {
  handler.handle(e);
};

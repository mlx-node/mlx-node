import { instantiateNapiModuleSync, MessageHandler, WASI } from '@napi-rs/wasm-runtime'

const handler = new MessageHandler({
  onLoad({ wasmModule, wasmMemory }) {
    const wasi = new WASI({
      print: function () { console.log.apply(console, arguments) },
      printErr: function () { console.error.apply(console, arguments) },
    })
    return instantiateNapiModuleSync(wasmModule, {
      childThread: true,
      wasi,
      overwriteImports(importObject) {
        importObject.env = {
          ...importObject.env,
          ...importObject.napi,
          ...importObject.emnapi,
          memory: wasmMemory,
          // C++ exception tag (required by -fwasm-exceptions)
          __cpp_exception: new WebAssembly.Tag({ parameters: ['i32'] }),
          // MLX GPU init — no-op (GPU initialized lazily via WebGPU bridge)
          _ZN3mlx4core3gpu4initEv: () => {},
          // WebGPU stubs — no-ops so WASM links. Real bridge injected by consumer.
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
          wgpuBufferUnmap: () => {}, wgpuBufferGetSize: () => 0,
          wgpuBindGroupRelease: () => {}, wgpuDeviceCreateBindGroup: () => 0,
          wgpuBufferGetMappedRange: () => 0,
        }
      },
    })
  },
})

globalThis.onmessage = function (e) {
  handler.handle(e)
}

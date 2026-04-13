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
          // Task 4's SabSink declares __wasm_i32_atomic_wait /
          // __wasm_atomic_notify as extern "C" — wasm-ld emits them as host
          // imports, so provide JS stubs wrapping Atomics.wait / notify.
          // Re-read wasmMemory.buffer each call because memory.grow() replaces
          // the buffer object; a cached view would point at the old, smaller
          // range and throw RangeError on indices beyond the old length.
          __wasm_i32_atomic_wait: (ptr, expected, timeoutNs) => {
            const view = new Int32Array(wasmMemory.buffer)
            const timeoutMs = timeoutNs === -1n ? Infinity : Number(timeoutNs / 1_000_000n)
            const result = Atomics.wait(view, ptr >>> 2, expected, timeoutMs)
            return result === 'ok' ? 0 : result === 'not-equal' ? 1 : 2
          },
          __wasm_atomic_notify: (ptr, count) => {
            const view = new Int32Array(wasmMemory.buffer)
            return Atomics.notify(view, ptr >>> 2, count)
          },
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

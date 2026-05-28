/**
 * SafeTensors Parser — creates GPU buffers directly from ArrayBuffer
 *
 * Parses the SafeTensors header to get tensor metadata (name, dtype, shape, offsets),
 * then creates WebGPU buffers directly from the raw data slices — zero WASM copy.
 *
 * SafeTensors format:
 *   [8 bytes: header_length (uint64 LE)]
 *   [header_length bytes: JSON header]
 *   [remaining bytes: raw tensor data]
 *
 * JSON header maps tensor name → { dtype, shape, data_offsets: [start, end] }
 */

export interface TensorInfo {
  name: string;
  dtype: string; // "F32", "F16", "BF16", "I32", "U32", "I8", "U8", "I16", "U16", "I64", "BOOL"
  shape: number[];
  byteOffset: number; // offset into the data region
  byteSize: number;
}

export interface ParsedSafeTensors {
  tensors: TensorInfo[];
  metadata: Record<string, string>;
  dataOffset: number; // offset where raw data starts in the ArrayBuffer
}

function parseSafeTensorsHeaderJson(headerBytes: Uint8Array, dataOffset: number): ParsedSafeTensors {
  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson);

  const tensors: TensorInfo[] = [];
  let metadata: Record<string, string> = {};

  for (const [key, value] of Object.entries(header)) {
    if (key === '__metadata__') {
      metadata = value as Record<string, string>;
      continue;
    }
    const info = value as { dtype: string; shape: number[]; data_offsets: [number, number] };
    tensors.push({
      name: key,
      dtype: info.dtype,
      shape: info.shape,
      byteOffset: info.data_offsets[0],
      byteSize: info.data_offsets[1] - info.data_offsets[0],
    });
  }

  return { tensors, metadata, dataOffset };
}

/** Parse a safetensors header JSON blob read separately from the tensor data. */
export function parseSafeTensorsHeaderBytes(headerBytes: Uint8Array, dataOffset: number): ParsedSafeTensors {
  return parseSafeTensorsHeaderJson(headerBytes, dataOffset);
}

/** Parse SafeTensors header without copying any tensor data */
export function parseSafeTensorsHeader(buffer: ArrayBufferLike): ParsedSafeTensors {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  return parseSafeTensorsHeaderJson(headerBytes, 8 + headerLen);
}

/**
 * Map SafeTensors dtype string to the integer code our C++ FFI expects.
 * Must match the DType enum in mlx-core/src/array/mod.rs:
 *   Float32=0, Int32=1, Float16=2, BFloat16=3, Uint32=4, Uint8=5
 */
export function dtypeToCode(dtype: string): number {
  switch (dtype) {
    case 'F32':
      return 0; // DType::Float32
    case 'I32':
      return 1; // DType::Int32
    case 'F16':
      return 2; // DType::Float16
    case 'BF16':
      return 3; // DType::BFloat16
    case 'U32':
      return 4; // DType::Uint32
    case 'U8':
      return 5; // DType::Uint8
    default:
      return 0; // fallback to float32
  }
}

/**
 * Create GPU buffers for all tensors in a SafeTensors file.
 * Data goes directly from the fetch ArrayBuffer → GPU — no WASM copy.
 *
 * Returns a map of tensor name → GPU buffer handle (integer from the bridge).
 */
export function createGPUBuffersFromSafeTensors(
  buffer: ArrayBufferLike,
  device: GPUDevice,
  addHandle: (obj: GPUBuffer) => number,
): Map<string, { handle: number; dtype: string; shape: number[]; byteSize: number }> {
  const { tensors, dataOffset } = parseSafeTensorsHeader(buffer);
  const result = new Map<string, { handle: number; dtype: string; shape: number[]; byteSize: number }>();

  for (const tensor of tensors) {
    // Round up size to 4 bytes (WebGPU buffer alignment requirement)
    const alignedSize = Math.max(4, Math.ceil(tensor.byteSize / 4) * 4);

    const gpuBuffer = device.createBuffer({
      size: alignedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    // Copy tensor data directly from the ArrayBuffer into the GPU buffer
    const mapped = new Uint8Array(gpuBuffer.getMappedRange());
    const src = new Uint8Array(buffer, dataOffset + tensor.byteOffset, tensor.byteSize);
    mapped.set(src);
    gpuBuffer.unmap();

    const handle = addHandle(gpuBuffer);
    result.set(tensor.name, {
      handle,
      dtype: tensor.dtype,
      shape: tensor.shape,
      byteSize: tensor.byteSize,
    });
  }

  return result;
}

/**
 * WebGPU limits to request when creating the renderer, so VirtualGeometry can use the GPU's full buffer sizes:
 *
 *   const renderer = new THREE.WebGPURenderer({ requiredLimits: await virtualGeometryLimits() });
 *
 * Only needed for single meshes whose cluster hierarchy is larger than 128 MB, the size every WebGPU device supports
 * by default: scans and models of roughly 5 million triangles or more. Smaller meshes, however many, never need it.
 * Pass the same `powerPreference` you give the renderer, so the limits come from the same GPU. Returns `{}` (no extra
 * limits) when WebGPU is not available.
 */
export async function virtualGeometryLimits(options: { powerPreference?: GPUPowerPreference } = {}): Promise<Record<string, number>> {
  const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined;
  const adapter = gpu ? await gpu.requestAdapter({ powerPreference: options.powerPreference }).catch(() => null) : null;
  if (!adapter) return {};
  return {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  };
}

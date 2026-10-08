/** Max triangles per meshlet. The renderer always issues `MAX_MESHLET_TRIANGLES * 3` vertices per drawn meshlet. */
export const MAX_MESHLET_TRIANGLES = 128;

/** Max unique vertices per meshlet (meshoptimizer allows up to 256). */
export const MAX_MESHLET_VERTICES = 128;

/** Stand-in for `Infinity` on the GPU: the parent error of root meshlets. */
export const ERROR_INFINITY = 1e30;

/**
 * Instances whose LOD range is at most this many meshlets, and where every meshlet is guaranteed to be
 * selected, emit their meshlets directly from the instance pass (no meshlet pass for them).
 */
export const FAST_PATH_MAX_MESHLETS = 4;

/**
 * Relative margin the fast path keeps from the LOD threshold, so float rounding between the bound and the
 * per-meshlet test can never make the fast path select a meshlet the full test would reject.
 */
export const FAST_PATH_MARGIN = 1e-3;

/** Floats per meshlet in `VirtualMeshData.meshletBounds` (4 x vec4). */
export const MESHLET_BOUNDS_STRIDE = 16;

/** Uints per meshlet in `VirtualMeshData.meshletInfo` (1 x uvec4). */
export const MESHLET_INFO_STRIDE = 4;



/** Debug views for `VirtualGeometry.debugMode`. */
export const VG_DEBUG_MODES = {
  shaded: 0,
  meshlets: 1,
  lodLevel: 2,
  instances: 3,
} as const;

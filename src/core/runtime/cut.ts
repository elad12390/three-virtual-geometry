import { ERROR_INFINITY, FAST_PATH_MARGIN, FAST_PATH_MAX_MESHLETS, MESHLET_BOUNDS_STRIDE } from '../constants.js';
import type { VirtualMeshData } from '../preprocess/buildVirtualMesh.js';

/** Camera parameters the LOD test needs. Same values the GPU pass uses. */
export interface CutView {
  /** Column-major world-to-view matrix (camera.matrixWorldInverse.elements). */
  viewMatrix: ArrayLike<number>;
  /** projection[1][1] * viewportHeight / 2: converts view-space error to pixels. */
  projScale: number;
  near: number;
  /**
   * Uniform scale of the instance, applied to the sphere radii and errors as the GPU does. `viewMatrix`
   * must then include the instance's scale in its centers too (view * model). Default 1.
   */
  scale?: number;
}

/**
 * CPU reference for the GPU LOD selection (no frustum culling, identity model matrix).
 * A meshlet is in the cut when its own projected error is within `threshold` pixels and its
 * parent's projected error is not. Returns one flag per meshlet.
 *
 * Use it to test the DAG (see `verifyCutCoverage`) or to pick LODs on the CPU, e.g. for physics or AI.
 */
export function selectCut(data: VirtualMeshData, view: CutView, threshold: number): Uint8Array {
  const v = view.viewMatrix;
  const B = data.meshletBounds;
  const selected = new Uint8Array(data.meshletCount);

  const s = view.scale ?? 1;
  const projectedError = (cx: number, cy: number, cz: number, r: number, error: number) => {
    if (error >= ERROR_INFINITY * 0.5) return Infinity;
    const x = v[0] * cx + v[4] * cy + v[8] * cz + v[12];
    const y = v[1] * cx + v[5] * cy + v[9] * cz + v[13];
    const z = v[2] * cx + v[6] * cy + v[10] * cz + v[14];
    const distance = Math.max(Math.hypot(x, y, z) - r * s, view.near);
    return (error * s * view.projScale) / distance;
  };

  for (let i = 0; i < data.meshletCount; i++) {
    const b = i * MESHLET_BOUNDS_STRIDE;
    const own = projectedError(B[b], B[b + 1], B[b + 2], B[b + 3], B[b + 8]);
    const parent = projectedError(B[b + 4], B[b + 5], B[b + 6], B[b + 7], B[b + 9]);
    selected[i] = own <= threshold && parent > threshold ? 1 : 0;
  }
  return selected;
}

/**
 * Checks that a cut has no holes: every leaf meshlet (LOD 0) is covered by a selected meshlet on its
 * path to the coarsest level (itself or one of its replacements, recursively).
 * Returns the indices of uncovered leaves (empty array = valid cut).
 */
export function verifyCutCoverage(data: VirtualMeshData, selected: Uint8Array): number[] {
  const covered = new Int8Array(data.meshletCount).fill(-1); // -1 unknown, 0 no, 1 yes
  const { replacementGroup: groupOf, replacementStart: start, replacementIndices: list } = data;

  const isCovered = (i: number): boolean => {
    if (covered[i] !== -1) return covered[i] === 1;
    let result = selected[i] === 1;
    const g = groupOf[i];
    if (g >= 0) for (let k = start[g]; !result && k < start[g + 1]; k++) result = isCovered(list[k]);
    covered[i] = result ? 1 : 0;
    return result;
  };

  const holes: number[] = [];
  for (let i = 0; i < data.meshletCount; i++) {
    if (data.meshletInfo[i * 4 + 2] === 0 && !isCovered(i)) holes.push(i);
  }
  return holes;
}

/** Distance from the camera to the instance's LOD sphere, and the band [dMin, dMax] every LOD test distance falls in. */
function lodDistanceBounds(data: VirtualMeshData, view: CutView): [number, number] {
  const v = view.viewMatrix;
  const [cx, cy, cz, r] = data.lodBoundsSphere;
  const x = v[0] * cx + v[4] * cy + v[8] * cz + v[12];
  const y = v[1] * cx + v[5] * cy + v[9] * cz + v[13];
  const z = v[2] * cx + v[6] * cy + v[10] * cz + v[14];
  const d = Math.hypot(x, y, z);
  const R = r * (view.scale ?? 1);
  return [Math.max(d - R, view.near), Math.max(d + R, view.near)];
}

/**
 * CPU reference for the GPU instance pass: the contiguous range of meshlets (sorted by LOD level)
 * that can possibly pass the LOD test for an instance, from distance bounds alone.
 *
 * Every LOD and parent sphere lies inside `lodBoundsSphere`, so the distance to any of them is within
 * [D - R, D + R]. A level can contribute only if its smallest own error could be within the threshold
 * at the farthest distance and its largest parent error could exceed it at the nearest.
 * Returns [start, end) or [0, 0] when nothing can be selected.
 */
export function instanceMeshletRange(data: VirtualMeshData, view: CutView, threshold: number): [number, number] {
  const [dMin, dMax] = lodDistanceBounds(data, view);

  const k = (view.scale ?? 1) * view.projScale;
  let lo = -1;
  let hi = -1;
  const levels = data.levelRanges.length / 2;
  for (let l = 0; l < levels; l++) {
    const minOwn = data.levelErrors[l * 2];
    const maxParent = data.levelErrors[l * 2 + 1];
    const ownPossible = (minOwn * k) / dMax <= threshold;
    const parentPossible = maxParent >= ERROR_INFINITY * 0.5 || (maxParent * k) / dMin > threshold;
    if (ownPossible && parentPossible) {
      if (lo < 0) lo = l;
      hi = l;
    }
  }
  if (lo < 0) return [0, 0];
  return [data.levelRanges[lo * 2], data.levelRanges[hi * 2] + data.levelRanges[hi * 2 + 1]];
}

/**
 * CPU reference for the GPU fast path of the instance pass. Returns the range [start, end) when every
 * meshlet in it is guaranteed to be selected by the LOD test, for any distance the instance can have, and
 * the range is at most FAST_PATH_MAX_MESHLETS long. Otherwise returns null (use the per-meshlet pass).
 *
 * Guarantee per level, over the distance band [dMin, dMax] of `instanceMeshletRange`:
 *  - largest own error projects to <= threshold at dMin (its closest possible distance), and
 *  - smallest parent error projects to > threshold at dMax (its farthest possible distance).
 * A small relative margin keeps float rounding on the GPU from flipping a borderline meshlet.
 * The caller must also check that the instance is fully inside the frustum (see `cullBoundsSphere`).
 */
export function instanceGuaranteedRange(data: VirtualMeshData, view: CutView, threshold: number): [number, number] | null {
  const [dMin, dMax] = lodDistanceBounds(data, view);
  const margin = FAST_PATH_MARGIN;
  const k = (view.scale ?? 1) * view.projScale;

  let lo = -1;
  let hi = -1;
  let possibleMeshlets = 0;
  let guaranteedMeshlets = 0;
  const levels = data.levelRanges.length / 2;
  for (let l = 0; l < levels; l++) {
    const minOwn = data.levelErrors[l * 2];
    const maxParent = data.levelErrors[l * 2 + 1];
    const ownPossible = (minOwn * k) / dMax <= threshold;
    const parentPossible = maxParent >= ERROR_INFINITY * 0.5 || (maxParent * k) / dMin > threshold;
    if (!ownPossible || !parentPossible) continue;
    if (lo < 0) lo = l;
    hi = l;
    const size = data.levelRanges[l * 2 + 1];
    possibleMeshlets += size;

    const maxOwn = data.levelGuardErrors[l * 2];
    const minParent = data.levelGuardErrors[l * 2 + 1];
    const ownSure = (maxOwn * k) / dMin <= threshold * (1 - margin);
    const parentSure = minParent >= ERROR_INFINITY * 0.5 || (minParent * k) / dMax > threshold * (1 + margin);
    if (ownSure && parentSure) guaranteedMeshlets += size;
  }
  if (lo < 0) return null;
  const start = data.levelRanges[lo * 2];
  const end = data.levelRanges[hi * 2] + data.levelRanges[hi * 2 + 1];
  // Possible levels must be contiguous (no impossible level inside the range) and all guaranteed.
  if (possibleMeshlets !== end - start || guaranteedMeshlets !== possibleMeshlets) return null;
  if (possibleMeshlets > FAST_PATH_MAX_MESHLETS) return null;
  return [start, end];
}

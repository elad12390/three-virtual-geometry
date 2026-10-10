/**
 * Builds the virtual-geometry meshlet LOD DAG for a single mesh.
 *
 * Ported from nanite-webgpu (MIT, Marcin Matuszczyk) `src/meshPreprocessing`,
 * using the npm `meshoptimizer` build instead of custom WASM, a JS partitioner
 * instead of METIS, and meshoptimizer clusterlod-style monotonic LOD bounds.
 *
 * Per level: partition meshlets into groups -> merge each group -> simplify to
 * ~50% with locked group borders -> split back into meshlets. All meshlets index
 * the same (original) vertex buffer, so normals/UVs survive simplification.
 */
import { MeshoptClusterizer, MeshoptSimplifier, type SimplifierFlags } from 'meshoptimizer';
import {
  ERROR_INFINITY,
  MAX_MESHLET_TRIANGLES,
  MAX_MESHLET_VERTICES,
  MESHLET_BOUNDS_STRIDE,
  MESHLET_INFO_STRIDE,
} from '../constants.js';
import { partitionMeshlets } from './partition.js';
import { buildVoxelProxy, VOXEL_ERROR_CELLS, type VoxelProxy } from './voxelProxy.js';
import { simplifyAggregate } from './aggregate.js';

export interface VirtualMeshSource {
  /** xyz per vertex */
  positions: Float32Array;
  /** xyz per vertex */
  normals: Float32Array;
  /** Optional rgb per vertex, multiplied with the material color. */
  colors?: Float32Array;
  /** Optional uv per vertex (texture coordinates). Vertices on a UV seam must be separate vertices. */
  uvs?: Float32Array;
  indices: Uint32Array;
}

export interface VirtualMeshBuildOptions {
  /** Meshlets per group before simplification. */
  groupSize?: number;
  maxLodLevels?: number;
  /** A group that keeps more than this fraction of its triangles stops refining (its meshlets become roots). */
  maxGroupKeepRatio?: number;
  /** Fraction of a group's triangles kept by its simplification (the per-level reduction factor). */
  simplifyRatio?: number;
  /** Stop when a whole level removed less than this fraction of triangles. */
  minLevelReduction?: number;
  /** Remove small disconnected components while simplifying (good for foliage). */
  prune?: boolean;
  /**
   * Keep simplifying below a single meshlet once the remaining roots form the whole mesh, so distant
   * instances can draw a handful of triangles (or nothing, with `prune`) instead of a full meshlet.
   */
  tailSimplify?: boolean;
  /** Stop the tail at this many triangles. */
  minRootTriangles?: number;
  /**
   * Let coarse LODs switch to voxel proxies when those are more accurate than simplified triangles
   * (foliage and other aggregate geometry; solid meshes keep their triangles). Default true.
   */
  voxelLods?: boolean;
  /**
   * Finest voxel grid for those proxies, in cells along the longest side. Default 64. A finer grid (128) lowers the
   * smallest error a voxel level can have, so foliage made of many separate pieces (palm fronds, leaves) switches to
   * its cheap coarse levels closer to the camera. Builds take longer and use more memory.
   */
  voxelResolution?: number;
  /**
   * Keep simplifying groups of separate small pieces (leaf cards, grass blades) when edge collapses stall: each piece
   * is simplified on its own, then neighbouring pieces are thinned in pairs, the kept piece widened to cover both
   * (see aggregate.ts). Without it such groups keep every piece up to the voxel levels. Default true.
   */
  aggregateLods?: boolean;
  onProgress?: (fraction: number) => void | Promise<void>;
}

export interface VirtualMeshData {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array | null;
  /** uv per vertex, or null when the source had none. */
  uvs: Float32Array | null;
  /** Concatenated meshlet triangle lists, indexing `positions`. */
  indices: Uint32Array;
  meshletCount: number;
  /**
   * 4 x vec4 per meshlet:
   * [0] LOD sphere (center, radius)  - shared by all meshlets created from the same group
   * [1] parent LOD sphere (center, radius)
   * [2] (error, parentError, 0, 0) - object-space; parentError = ERROR_INFINITY for roots
   * [3] culling sphere (center, radius) - the meshlet's own geometry
   */
  meshletBounds: Float32Array;
  /**
   * uvec4 per meshlet: (triangleCount, firstIndex, lodLevel, vertexOffset).
   * `firstIndex / 3` is also the meshlet's first entry in `meshletTriangles`.
   */
  meshletInfo: Uint32Array;
  /** Global vertex ids, per meshlet, in meshlet-local order (`vertexOffset` indexes this). */
  meshletVertices: Uint32Array;
  /** One packed triangle per entry: three meshlet-local vertex ids in bits 0-7, 8-15, 16-23. */
  meshletTriangles: Uint32Array;
  /** Meshlets are sorted by LOD level. Per level: [firstMeshlet, meshletCount]. */
  levelRanges: Uint32Array;
  /** Per level: [smallest own error, largest parent error] in object space (parent may be ERROR_INFINITY). */
  levelErrors: Float32Array;
  /**
   * Per level: [largest own error, smallest parent error] in object space. Used by the fast path: if every
   * meshlet of a level passes the LOD test at the nearest distance (largest own error) and at the farthest
   * (smallest parent error), the level needs no per-meshlet test.
   */
  levelGuardErrors: Float32Array;
  /** Sphere enclosing every LOD and parent sphere: bounds the distances the LOD test can see. */
  lodBoundsSphere: [number, number, number, number];
  /** Sphere enclosing every meshlet's culling sphere (all LODs): if an instance sphere is inside the frustum, so are they. */
  cullBoundsSphere: [number, number, number, number];
  /**
   * DAG links (CPU only, for `verifyCutCoverage`; not stored in .vgeo files). Meshlet i belongs to the group that
   * `replacementGroup[i]` names (-1 for the coarsest meshlets), and that group is replaced by the coarser meshlets
   * `replacementIndices[replacementStart[g] .. replacementStart[g + 1]]`. Every meshlet of a group shares one list, so
   * the size stays linear even when a simplification step merges thousands of meshlets at once.
   */
  replacementGroup: Int32Array;
  replacementStart: Uint32Array;
  replacementIndices: Uint32Array;
  /** Whole-object bounding sphere (center xyz, radius). */
  boundingSphere: [number, number, number, number];
  stats: {
    leafTriangles: number;
    leafMeshlets: number;
    rootTriangles: number;
    rootMeshlets: number;
    lodLevels: number;
    /** Coarse levels built from voxel proxies (see voxelProxy.ts). */
    voxelLevels: number;
    /** Groups simplified piece by piece (see aggregate.ts; absent in files baked before it). */
    aggregateGroups?: number;
    buildMs: number;
  };
}

type Sphere = [number, number, number, number];

interface WipMeshlet {
  indices: Uint32Array;
  lodLevel: number;
  lodSphere: Sphere;
  error: number;
  parentSphere: Sphere;
  parentError: number;
  cullSphere: Sphere;
  boundaryEdges: number[];
  center: [number, number, number];
  /** Coarser meshlets that replaced this one (the outputs of the group it was merged into). */
  replacements: WipMeshlet[];
  /** Global vertex ids of this meshlet's unique vertices (meshlet-local order). */
  vertices: Uint32Array;
  /** Per triangle corner: index into `vertices` (0..MAX_MESHLET_VERTICES-1). */
  localTriangles: Uint8Array;
}

const DEFAULTS = {
  groupSize: 12,
  maxLodLevels: 24,
  maxGroupKeepRatio: 0.9,
  simplifyRatio: 0.5,
  minLevelReduction: 0.03,
  prune: false,
  tailSimplify: true,
  minRootTriangles: 4,
  voxelLods: true,
  voxelResolution: 64,
  aggregateLods: true,
};

/** The tail (whole-object simplification, where voxel proxies can take over) starts at this many triangles. */
const TAIL_START_TRIANGLES = 1024;
/** Voxel proxy resolutions tried for each tail step (cells along the longest side), from `voxelResolution` down. */
const voxelResolutions = (finest: number) => {
  const list: number[] = [];
  for (let r = Math.max(4, Math.round(finest)); r >= 4; r = Math.floor(r / 2)) list.push(r);
  return list;
};

const yieldToBrowser = () => new Promise<void>((r) => setTimeout(r, 0));

export async function buildVirtualMesh(src: VirtualMeshSource, options: VirtualMeshBuildOptions = {}): Promise<VirtualMeshData> {
  const opts = { ...DEFAULTS, ...options };
  const t0 = performance.now();
  await MeshoptClusterizer.ready;
  await MeshoptSimplifier.ready;

  // Vertex buffers grow when voxel proxies add their own vertices (appendVertices).
  let positions = src.positions;
  let normals = src.normals;
  let colors = src.colors ?? null;
  let uvs = src.uvs ?? null;
  // Vertices sharing a position (UV/normal seams) are the same vertex topologically.
  let weld = MeshoptSimplifier.generatePositionRemap(positions, 3);
  let voxelLevels = 0;
  /** Groups simplified piece by piece (see aggregate.ts). */
  let aggregateLevels = 0;
  const flags: SimplifierFlags[] = ['LockBorder', 'ErrorAbsolute'];
  if (opts.prune) flags.push('Prune');

  const all: WipMeshlet[] = [];
  const proxyCache = new Map<number, VoxelProxy | null>();
  const leafs = splitIntoMeshlets(src.indices, 0, null, 0);
  let current = leafs;
  let lastTriangles = countTriangles(current);
  let lodLevels = 1;
  const estimatedTotal = Math.max(1, leafs.length * 2);

  for (let level = 1; level <= opts.maxLodLevels && current.length > 1; level++) {
    // Small enough to treat as one piece: the tail takes over (and may switch to voxel proxies).
    if (opts.tailSimplify && lastTriangles <= TAIL_START_TRIANGLES) break;
    // Aggregate geometry: once a voxel stand-in of the whole mesh is as accurate as this level with half its
    // triangles, the tail takes over too. Its voxel levels are measured against the original, so their error does
    // not pile up level after level like that of thinned pieces.
    if (opts.tailSimplify && opts.voxelLods && aggregateLevels > 0 && voxelBeats(current, lastTriangles)) break;
    const groups = partitionMeshlets(current, opts.groupSize);
    const next: WipMeshlet[] = [];
    const owner = opts.aggregateLods ? groupOwners(groups, current) : null;
    // Error of a voxel stand-in for the whole mesh with half this level's triangles: groups thinned piece by piece
    // must do better than that, or they are left to the voxel levels of the tail.
    let voxelLimit = NaN;

    for (const [g, groupIdx] of groups.entries()) {
      const group = groupIdx.map((i) => current[i]);
      const merged = concatIndices(group);
      const local = compact(merged, positions);
      const target = Math.max(3, Math.floor((merged.length * opts.simplifyRatio) / 3) * 3);
      const [simplified, simplifyError] = MeshoptSimplifier.simplify(local.indices, local.positions, 3, target, 1e10, flags);
      let globalIndices: Uint32Array = new Uint32Array(simplified.length);
      for (let i = 0; i < simplified.length; i++) globalIndices[i] = local.toGlobal[simplified[i]];
      // meshopt's error is vertex displacement; it misses silhouette loss (pruned parts, collapsed sheets).
      let shapeError = Math.max(simplifyError, extentShrink(local.indices, simplified, local.positions));

      // Stalled well short of the target, as groups of separate pieces do (leaves, grass): simplify piece by piece.
      if (owner && simplified.length > target * 1.25) {
        const aggregate = simplifyAggregate(merged, positions, weld, owner, g, target / 3);
        if (aggregate && Number.isNaN(voxelLimit)) voxelLimit = opts.tailSimplify && opts.voxelLods ? voxelError(lastTriangles) : Infinity;
        const childError = Math.max(...group.map((m) => m.error));
        if (aggregate && aggregate.indices.length < simplified.length && childError + aggregate.error < voxelLimit) {
          const s = aggregate.newSources;
          if (s.length > 0) {
            const pick = (src: Float32Array, size: number) => {
              const out = new Float32Array(s.length * size);
              s.forEach((v, i) => out.set(src.subarray(v * size, v * size + size), i * size));
              return out;
            };
            appendVertices(aggregate.newPositions, pick(normals, 3), colors ? pick(colors, 3) : null, uvs ? pick(uvs, 2) : null);
          }
          globalIndices = aggregate.indices;
          shapeError = Math.max(aggregate.error, extentShrink(merged, globalIndices, positions));
          aggregateLevels++;
        }
      }

      if (globalIndices.length > merged.length * opts.maxGroupKeepRatio) {
        continue; // cannot reduce this group any further: its meshlets stay roots
      }

      // Monotonic LOD bounds: the group sphere encloses every child LOD sphere,
      // and the group error is never smaller than any child error.
      const lodSphere = enclosingSphere(group.map((m) => m.lodSphere));
      const childError = Math.max(...group.map((m) => m.error));
      const error = childError + Math.max(shapeError, 1e-7);
      for (const child of group) {
        child.parentSphere = lodSphere;
        child.parentError = error;
      }

      if (globalIndices.length > 0) {
        const outputs = splitIntoMeshlets(globalIndices, level, lodSphere, error);
        next.push(...outputs);
        for (const child of group) child.replacements = outputs;
      }
    }

    await opts.onProgress?.(Math.min(0.99, all.length / estimatedTotal));
    await yieldToBrowser();

    if (next.length === 0) break;
    lodLevels = level + 1;
    current = next;

    const triangles = countTriangles(current);
    if (triangles > lastTriangles * (1 - opts.minLevelReduction)) break;
    lastTriangles = triangles;
  }

  // Every root, not only the last level: groups that could not be reduced (e.g. UV-seam fans at a sphere's
  // poles) leave roots on earlier levels, and together the roots still cover the whole mesh.
  if (opts.tailSimplify) lodLevels = simplifyTail(all.filter((m) => m.parentError >= ERROR_INFINITY), lodLevels);

  await opts.onProgress?.(1);
  return pack({ positions, normals, colors: colors ?? undefined, uvs: uvs ?? undefined, indices: src.indices }, all, lodLevels, voxelLevels, performance.now() - t0, aggregateLevels);

  /** Smallest error of a simplified voxel proxy with at most half of `triangles` (Infinity: none gets there). */
  function voxelError(triangles: number) {
    const target = Math.max(3, Math.floor(triangles / 2) * 3);
    let best = Infinity;
    for (const resolution of voxelResolutions(opts.voxelResolution)) {
      const proxy = voxelProxy(resolution);
      if (!proxy) continue;
      const [simplified, proxyError] = MeshoptSimplifier.simplify(proxy.indices, proxy.positions, 3, Math.min(target, proxy.indices.length), 1e10, ['ErrorAbsolute']);
      if (simplified.length > 0 && simplified.length <= target) best = Math.min(best, VOXEL_ERROR_CELLS * proxy.cellSize + proxyError);
    }
    return best;
  }

  /** True when a simplified voxel proxy with at most half of `triangles` has no more error than `meshlets`. */
  function voxelBeats(meshlets: WipMeshlet[], triangles: number) {
    let levelError = 0;
    for (const m of meshlets) levelError = Math.max(levelError, m.error);
    return voxelError(triangles) <= levelError;
  }

  /** Per welded vertex: the group whose meshlets use it, or -2 when meshlets of several groups do (-1: unused). */
  function groupOwners(groups: number[][], meshlets: WipMeshlet[]) {
    const owner = new Int32Array(weld.length).fill(-1);
    groups.forEach((group, g) => {
      for (const i of group) {
        const indices = meshlets[i].indices;
        for (let k = 0; k < indices.length; k++) {
          const w = weld[indices[k]];
          if (owner[w] === -1) owner[w] = g;
          else if (owner[w] !== g) owner[w] = -2;
        }
      }
    });
    return owner;
  }

  /** Adds vertices to the shared buffers and returns the index of the first one. */
  function appendVertices(p: Float32Array, n: Float32Array, c: Float32Array | null, uv: Float32Array | null) {
    const first = positions.length / 3;
    const count = p.length / 3;
    const grow = (a: Float32Array, b: Float32Array) => {
      const out = new Float32Array(a.length + b.length);
      out.set(a);
      out.set(b, a.length);
      return out;
    };
    positions = grow(positions, p);
    normals = grow(normals, n);
    if (colors) colors = grow(colors, c ?? new Float32Array(count * 3).fill(1));
    if (uvs) uvs = grow(uvs, uv ?? new Float32Array(count * 2));
    const w = new Uint32Array(weld.length + count);
    w.set(weld);
    for (let i = 0; i < count; i++) w[weld.length + i] = first + i;
    weld = w;
    return first;
  }

  /** Voxel proxies of the full-detail mesh, built on first use per resolution. */
  function voxelProxy(resolution: number): VoxelProxy | null {
    if (!proxyCache.has(resolution)) proxyCache.set(resolution, buildVoxelProxy(src.positions, src.colors, src.indices, resolution, src.uvs));
    return proxyCache.get(resolution)!;
  }

  /**
   * Below one meshlet: merge all remaining roots and keep halving. Borders can be unlocked because
   * the roots cover the whole mesh, so no other meshlet shares them.
   */
  function simplifyTail(roots: WipMeshlet[], levels: number): number {
    if (roots.length === 0) return levels;

    const tailFlags: SimplifierFlags[] = ['ErrorAbsolute'];
    if (opts.prune) tailFlags.push('Prune');
    let nodes = roots;
    let level = levels;
    for (; level < levels + opts.maxLodLevels; level++) {
      const merged = concatIndices(nodes);
      if (merged.length / 3 <= opts.minRootTriangles) break;
      const local = compact(merged, positions);
      const target = Math.max(3, Math.floor(merged.length / 2 / 3) * 3);
      const keepLimit = merged.length * opts.maxGroupKeepRatio;
      const childError = Math.max(...nodes.map((m) => m.error));

      // Candidate 1: simplify the current triangles (errors accumulate level over level).
      let [result, error] = MeshoptSimplifier.simplify(local.indices, local.positions, 3, target, 1e10, tailFlags);
      if (result.length > keepLimit) {
        // Topology blocks regular collapses (e.g. many disconnected parts): try sloppy simplification.
        const [sloppy, relativeError] = MeshoptSimplifier.simplifySloppy(local.indices, local.positions, 3, null, target, 1);
        if (sloppy.length < result.length) {
          result = sloppy;
          error = relativeError * MeshoptSimplifier.getScale(local.positions, 3);
        }
      }
      let globalIndices: Uint32Array | null = null;
      let parentError = Infinity;
      if (result.length <= keepLimit) {
        parentError = childError + Math.max(Math.max(error, extentShrink(local.indices, result, local.positions)), 1e-7);
        globalIndices = new Uint32Array(result.length);
        for (let i = 0; i < result.length; i++) globalIndices[i] = local.toGlobal[result[i]];
      }

      // Candidate 2: a simplified voxel proxy of the full-detail mesh. Its error is measured against the
      // original directly, so it does not accumulate. It wins when triangles can only shrink the shape.
      let voxel: { proxy: VoxelProxy; indices: Uint32Array } | null = null;
      if (opts.voxelLods) {
        for (const resolution of voxelResolutions(opts.voxelResolution)) {
          const proxy = voxelProxy(resolution);
          if (!proxy) continue;
          const proxyTarget = Math.min(target, proxy.indices.length);
          const [simplified, proxyError] = MeshoptSimplifier.simplify(proxy.indices, proxy.positions, 3, proxyTarget, 1e10, ['ErrorAbsolute']);
          if (simplified.length === 0 || simplified.length > keepLimit) continue;
          const voxelError = Math.max(childError * (1 + 1e-5) + 1e-7, VOXEL_ERROR_CELLS * proxy.cellSize + proxyError);
          if (voxelError < parentError) {
            parentError = voxelError;
            voxel = { proxy, indices: simplified };
          }
        }
      }
      if (!Number.isFinite(parentError)) break;
      if (voxel) {
        // Copy the proxy vertices this level uses into the shared vertex buffers.
        const used = compact(voxel.indices, voxel.proxy.positions);
        const pick = (src: Float32Array, size = 3) => {
          const out = new Float32Array(used.toGlobal.length * size);
          used.toGlobal.forEach((v, i) => out.set(src.subarray(v * size, v * size + size), i * size));
          return out;
        };
        const { proxy } = voxel;
        const first = appendVertices(used.positions, pick(proxy.normals), proxy.colors ? pick(proxy.colors) : null, proxy.uvs ? pick(proxy.uvs, 2) : null);
        globalIndices = used.indices.map((i) => i + first);
        voxelLevels++;
      }

      const lodSphere = enclosingSphere(nodes.map((m) => m.lodSphere));
      for (const node of nodes) {
        node.parentSphere = lodSphere;
        node.parentError = parentError;
      }
      if (globalIndices!.length === 0) return level + 1; // pruned away entirely beyond this distance

      const outputs = splitIntoMeshlets(globalIndices!, level, lodSphere, parentError);
      for (const node of nodes) node.replacements = outputs;
      nodes = outputs;
    }
    return level;
  }

  ////////////////

  function splitIntoMeshlets(indices: Uint32Array, lodLevel: number, lodSphere: Sphere | null, error: number): WipMeshlet[] {
    const local = compact(indices, positions);
    const built = MeshoptClusterizer.buildMeshlets(local.indices, local.positions, 3, MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES, 0);
    const result: WipMeshlet[] = [];

    for (let m = 0; m < built.meshletCount; m++) {
      const vertexOffset = built.meshlets[m * 4 + 0];
      const triangleOffset = built.meshlets[m * 4 + 1];
      const vertexCount = built.meshlets[m * 4 + 2];
      const triangleCount = built.meshlets[m * 4 + 3];
      const vertices = new Uint32Array(vertexCount);
      for (let v = 0; v < vertexCount; v++) vertices[v] = local.toGlobal[built.vertices[vertexOffset + v]];
      const localTriangles = built.triangles.slice(triangleOffset, triangleOffset + triangleCount * 3);
      const meshletIndices = new Uint32Array(triangleCount * 3);
      for (let i = 0; i < triangleCount * 3; i++) meshletIndices[i] = vertices[localTriangles[i]];

      const cullSphere = boundingSphere(meshletIndices, positions);
      const meshlet: WipMeshlet = {
        indices: meshletIndices,
        lodLevel,
        lodSphere: lodSphere ?? cullSphere,
        error,
        parentSphere: [0, 0, 0, 0],
        parentError: ERROR_INFINITY,
        cullSphere,
        boundaryEdges: findBoundaryEdges(meshletIndices, weld),
        center: [cullSphere[0], cullSphere[1], cullSphere[2]],
        replacements: [],
        vertices,
        localTriangles,
      };
      result.push(meshlet);
      all.push(meshlet);
    }
    return result;
  }
}

function pack(src: VirtualMeshSource, unsorted: WipMeshlet[], lodLevels: number, voxelLevels: number, buildMs: number, aggregateGroups: number): VirtualMeshData {
  // Sorted by LOD level so every level is a contiguous range (used for per-instance level ranges).
  const all = [...unsorted].sort((a, b) => a.lodLevel - b.lodLevel);
  const meshletCount = all.length;
  const indices = new Uint32Array(all.reduce((acc, m) => acc + m.indices.length, 0));
  const meshletBounds = new Float32Array(meshletCount * MESHLET_BOUNDS_STRIDE);
  const meshletInfo = new Uint32Array(meshletCount * MESHLET_INFO_STRIDE);
  const indexOf = new Map<WipMeshlet, number>();
  all.forEach((m, i) => indexOf.set(m, i));
  // Every meshlet of a group points at the same `replacements` array: store each group's list once.
  const replacementGroup = new Int32Array(meshletCount).fill(-1);
  const groupOf = new Map<WipMeshlet[], number>();
  const groupStarts: number[] = [0];
  let replacementTotal = 0;
  all.forEach((m, i) => {
    if (m.replacements.length === 0) return;
    let g = groupOf.get(m.replacements);
    if (g === undefined) {
      g = groupOf.size;
      groupOf.set(m.replacements, g);
      replacementTotal += m.replacements.length;
      groupStarts.push(replacementTotal);
    }
    replacementGroup[i] = g;
  });
  const replacementStart = Uint32Array.from(groupStarts);
  const replacementIndices = new Uint32Array(replacementTotal);
  for (const [list, g] of groupOf) list.forEach((r, k) => (replacementIndices[replacementStart[g] + k] = indexOf.get(r)!));

  const meshletVertices = new Uint32Array(all.reduce((acc, m) => acc + m.vertices.length, 0));
  const meshletTriangles = new Uint32Array(indices.length / 3);
  let firstIndex = 0;
  let vertexOffset = 0;
  const stats = { leafTriangles: 0, leafMeshlets: 0, rootTriangles: 0, rootMeshlets: 0, lodLevels, voxelLevels, aggregateGroups, buildMs };
  all.forEach((m, i) => {
    indices.set(m.indices, firstIndex);
    meshletVertices.set(m.vertices, vertexOffset);
    const t = m.localTriangles;
    for (let k = 0, tri = firstIndex / 3; k < t.length; k += 3, tri++) {
      meshletTriangles[tri] = t[k] | (t[k + 1] << 8) | (t[k + 2] << 16);
    }
    const b = i * MESHLET_BOUNDS_STRIDE;
    meshletBounds.set(m.lodSphere, b);
    meshletBounds.set(m.parentSphere, b + 4);
    meshletBounds[b + 8] = m.error;
    meshletBounds[b + 9] = m.parentError;
    meshletBounds.set(m.cullSphere, b + 12);

    const triangleCount = m.indices.length / 3;
    const o = i * MESHLET_INFO_STRIDE;
    meshletInfo[o + 0] = triangleCount;
    meshletInfo[o + 1] = firstIndex;
    meshletInfo[o + 2] = m.lodLevel;
    meshletInfo[o + 3] = vertexOffset;
    firstIndex += m.indices.length;
    vertexOffset += m.vertices.length;

    if (m.lodLevel === 0) {
      stats.leafTriangles += triangleCount;
      stats.leafMeshlets += 1;
    }
    if (m.parentError >= ERROR_INFINITY) {
      stats.rootTriangles += triangleCount;
      stats.rootMeshlets += 1;
    }
  });

  const whole = enclosingSphere(all.filter((m) => m.lodLevel === 0).map((m) => m.cullSphere));

  // Per level: [start, count] and [smallest own error, largest parent error] (object space).
  const levelCount = all.length ? all[all.length - 1].lodLevel + 1 : 0;
  const levelRanges = new Uint32Array(levelCount * 2);
  const levelErrors = new Float32Array(levelCount * 2);
  const levelGuardErrors = new Float32Array(levelCount * 2);
  for (let l = 0; l < levelCount; l++) {
    levelErrors[l * 2] = Infinity;
    levelErrors[l * 2 + 1] = 0;
    levelGuardErrors[l * 2] = 0;
    levelGuardErrors[l * 2 + 1] = Infinity;
  }
  all.forEach((m, i) => {
    const l = m.lodLevel;
    if (levelRanges[l * 2 + 1] === 0) levelRanges[l * 2] = i;
    levelRanges[l * 2 + 1]++;
    levelErrors[l * 2] = Math.min(levelErrors[l * 2], m.error);
    levelErrors[l * 2 + 1] = Math.max(levelErrors[l * 2 + 1], m.parentError);
    levelGuardErrors[l * 2] = Math.max(levelGuardErrors[l * 2], m.error);
    levelGuardErrors[l * 2 + 1] = Math.min(levelGuardErrors[l * 2 + 1], m.parentError);
  });
  // Encloses every sphere the LOD test can look at, so per-instance distance bounds stay conservative.
  const lodBoundsSphere = enclosingSphere([
    ...all.map((m) => m.lodSphere),
    ...all.filter((m) => m.parentError < ERROR_INFINITY).map((m) => m.parentSphere),
  ]);
  const cullBoundsSphere = enclosingSphere(all.map((m) => m.cullSphere));
  return {
    positions: src.positions,
    normals: src.normals,
    colors: src.colors ?? null,
    uvs: src.uvs ?? null,
    indices,
    meshletCount,
    meshletBounds,
    meshletInfo,
    replacementGroup,
    replacementStart,
    replacementIndices,
    meshletVertices,
    meshletTriangles,
    levelRanges,
    levelErrors,
    levelGuardErrors,
    lodBoundsSphere,
    cullBoundsSphere,
    boundingSphere: whole,
    stats,
  };
}

/**
 * How far the simplified mesh's bounding box retreats from the original's, per side (max over the 6
 * sides). An empty result counts as losing the whole extent. Used as a floor for the LOD error so a
 * version that lost its silhouette is only drawn once that loss is below the pixel threshold.
 */
export function extentShrink(before: Uint32Array, after: Uint32Array, positions: Float32Array): number {
  const box = (indices: Uint32Array) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < indices.length; i++) {
      const o = indices[i] * 3;
      for (let k = 0; k < 3; k++) {
        const v = positions[o + k];
        if (v < min[k]) min[k] = v;
        if (v > max[k]) max[k] = v;
      }
    }
    return { min, max };
  };
  const a = box(before);
  if (after.length === 0) return Math.max(a.max[0] - a.min[0], a.max[1] - a.min[1], a.max[2] - a.min[2]) / 2;
  const b = box(after);
  let shrink = 0;
  for (let k = 0; k < 3; k++) shrink = Math.max(shrink, b.min[k] - a.min[k], a.max[k] - b.max[k]);
  return shrink;
}

/**
 * Vertex id -> local id scratch for compact(), all -1 between calls. A typed-array lookup instead of a Map:
 * compact() runs several times per group, so this is one of the hottest loops of the build.
 */
let localOf = new Int32Array(0);

/** Re-index a triangle subset into a compact local vertex buffer so meshopt only touches what it needs. */
function compact(indices: Uint32Array, positions: Float32Array) {
  const vertexCount = positions.length / 3;
  if (localOf.length < vertexCount) localOf = new Int32Array(vertexCount).fill(-1);
  const toGlobalAll = new Uint32Array(indices.length);
  const localIndices = new Uint32Array(indices.length);
  let count = 0;
  for (let i = 0; i < indices.length; i++) {
    const g = indices[i];
    let l = localOf[g];
    if (l < 0) {
      l = count++;
      localOf[g] = l;
      toGlobalAll[l] = g;
    }
    localIndices[i] = l;
  }
  const toGlobal = toGlobalAll.subarray(0, count);
  const localPositions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const g = toGlobal[i];
    localOf[g] = -1;
    localPositions[i * 3 + 0] = positions[g * 3 + 0];
    localPositions[i * 3 + 1] = positions[g * 3 + 1];
    localPositions[i * 3 + 2] = positions[g * 3 + 2];
  }
  return { indices: localIndices, positions: localPositions, toGlobal };
}

/** Open-addressing scratch for findBoundaryEdges(): edge keys, use counts and insertion order. */
let edgeTableKeys = new Float64Array(1024);
let edgeTableCounts = new Uint8Array(1024);
let edgeOrder = new Int32Array(512);

/** Edges used by exactly one triangle, keyed by position-welded vertex ids (in first-seen order). */
function findBoundaryEdges(indices: Uint32Array, weld: Uint32Array): number[] {
  const edgeCount = indices.length;
  let size = edgeTableKeys.length;
  if (size < edgeCount * 2) {
    while (size < edgeCount * 2) size <<= 1;
    edgeTableKeys = new Float64Array(size);
    edgeTableCounts = new Uint8Array(size);
    edgeOrder = new Int32Array(size / 2);
  }
  const mask = size - 1;
  let inserted = 0;
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = weld[indices[t + k]];
      const b = weld[indices[t + ((k + 1) % 3)]];
      const key = edgeKey(a, b);
      let h = (Math.imul(a < b ? a : b, 0x9e3779b1) ^ Math.imul(a < b ? b : a, 0x85ebca6b)) & mask;
      while (edgeTableCounts[h] !== 0 && edgeTableKeys[h] !== key) h = (h + 1) & mask;
      if (edgeTableCounts[h] === 0) {
        edgeTableKeys[h] = key;
        edgeOrder[inserted++] = h;
      }
      if (edgeTableCounts[h] < 255) edgeTableCounts[h]++;
    }
  }
  const result: number[] = [];
  for (let i = 0; i < inserted; i++) {
    const h = edgeOrder[i];
    if (edgeTableCounts[h] === 1) result.push(edgeTableKeys[h]);
    edgeTableCounts[h] = 0;
  }
  return result;
}

/** Order-independent edge id. Safe while vertex ids stay below 2^26. */
export const edgeKey = (a: number, b: number) => (a < b ? a * 67108864 + b : b * 67108864 + a);

function concatIndices(meshlets: WipMeshlet[]) {
  const result = new Uint32Array(meshlets.reduce((acc, m) => acc + m.indices.length, 0));
  let offset = 0;
  for (const m of meshlets) {
    result.set(m.indices, offset);
    offset += m.indices.length;
  }
  return result;
}

const countTriangles = (meshlets: WipMeshlet[]) => meshlets.reduce((acc, m) => acc + m.indices.length / 3, 0);

function boundingSphere(indices: Uint32Array, positions: Float32Array): Sphere {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < indices.length; i++) {
    const o = indices[i] * 3;
    for (let k = 0; k < 3; k++) {
      const v = positions[o + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  const cx = (min[0] + max[0]) / 2;
  const cy = (min[1] + max[1]) / 2;
  const cz = (min[2] + max[2]) / 2;
  let r2 = 0;
  for (let i = 0; i < indices.length; i++) {
    const o = indices[i] * 3;
    const dx = positions[o] - cx;
    const dy = positions[o + 1] - cy;
    const dz = positions[o + 2] - cz;
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  return [cx, cy, cz, Math.sqrt(r2)];
}

/** Sphere enclosing all given spheres (meshoptimizer's computeSphereBounds), padded for float error. */
function enclosingSphere(spheres: Sphere[]): Sphere {
  if (spheres.length === 1) return [...spheres[0]] as Sphere;
  const centers = new Float32Array(spheres.length * 3);
  const radii = new Float32Array(spheres.length);
  spheres.forEach((s, i) => {
    centers.set([s[0], s[1], s[2]], i * 3);
    radii[i] = s[3];
  });
  const b = MeshoptClusterizer.computeSphereBounds(centers, 3, radii, 1);
  let radius = b.radius;
  // guarantee containment regardless of the approximation used
  for (const s of spheres) {
    const d = Math.hypot(s[0] - b.centerX, s[1] - b.centerY, s[2] - b.centerZ) + s[3];
    radius = Math.max(radius, d);
  }
  return [b.centerX, b.centerY, b.centerZ, radius * 1.0001];
}

import * as THREE from 'three/webgpu';
import type { VirtualMeshSource } from './core/preprocess/buildVirtualMesh.js';

type Attribute = THREE.BufferAttribute | THREE.InterleavedBufferAttribute;

export interface FromGeometryOptions {
  /**
   * Keep the `uv` attribute. Vertices are then welded by position and UV, so UV seams stay separate vertices
   * (the simplifier treats them as seams, not borders). Default: true when the geometry has a `uv` attribute.
   */
  uvs?: boolean;
  /** Keep the `color` attribute as per-vertex colors (multiplied with the material color). Default: false. */
  colors?: boolean;
  /**
   * 'compute' (default): smooth normals over the position-welded mesh, ignoring the `normal` attribute.
   * 'keep': use the geometry's normals; vertices with different normals stay separate, so hard edges stay hard.
   */
  normals?: 'compute' | 'keep';
  /** Only this range of the index buffer (or of the vertices, for non-indexed geometry), e.g. one of `geometry.groups`. */
  range?: { start: number; count: number };
}

/** Weld tolerance per channel: positions and UVs need to match almost exactly, normals and colors a little less. */
const POSITION_TOLERANCE = 1e-6;
const ATTRIBUTE_TOLERANCE = 1e-4;

/**
 * Weld vertices and produce a `VirtualMeshSource`. By default vertices are welded by position (and UV when the
 * geometry has UVs) and get smooth normals, so meshlets share edges. See `FromGeometryOptions`.
 */
export function fromBufferGeometry(geometry: THREE.BufferGeometry, options: FromGeometryOptions = {}): VirtualMeshSource {
  const position = geometry.attributes.position as Attribute;
  const uv = (options.uvs ?? true) ? (geometry.attributes.uv as Attribute | undefined) : undefined;
  const color = options.colors ? (geometry.attributes.color as Attribute | undefined) : undefined;
  const normal = options.normals === 'keep' ? (geometry.attributes.normal as Attribute | undefined) : undefined;
  const index = geometry.index;
  const total = index ? index.count : position.count;
  const start = Math.max(0, Math.min(total, options.range?.start ?? 0));
  const count = Math.floor((Math.min(total, start + (options.range?.count ?? total)) - start) / 3) * 3;
  const vertexOf = (corner: number) => (index ? index.getX(start + corner) : start + corner);

  // Position-only weld: the topology smooth normals are computed on (and the whole result when nothing else is kept).
  const byPosition = weld(count, vertexOf, [{ attribute: position, size: 3, tolerance: POSITION_TOLERANCE }]);
  const positions = gather(byPosition.firstVertex, position, 3);
  let smoothNormals: Float32Array | null = null;
  if (!normal) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setIndex(new THREE.BufferAttribute(byPosition.indices, 1));
    g.computeVertexNormals();
    smoothNormals = g.attributes.normal.array as Float32Array;
  }
  if (!uv && !color && !normal) return { positions, normals: smoothNormals!, indices: byPosition.indices };

  // Full weld: vertices that differ in any kept attribute stay separate (seams).
  const channels = [{ attribute: position, size: 3, tolerance: POSITION_TOLERANCE }];
  if (uv) channels.push({ attribute: uv, size: 2, tolerance: POSITION_TOLERANCE });
  if (normal) channels.push({ attribute: normal, size: 3, tolerance: ATTRIBUTE_TOLERANCE });
  if (color) channels.push({ attribute: color, size: 3, tolerance: ATTRIBUTE_TOLERANCE });
  const full = weld(count, vertexOf, channels);
  const result: VirtualMeshSource = { positions: gather(full.firstVertex, position, 3), normals: new Float32Array(0), indices: full.indices };
  if (normal) {
    result.normals = gather(full.firstVertex, normal, 3);
    for (let i = 0; i < result.normals.length; i += 3) {
      const l = Math.hypot(result.normals[i], result.normals[i + 1], result.normals[i + 2]) || 1;
      result.normals[i] /= l;
      result.normals[i + 1] /= l;
      result.normals[i + 2] /= l;
    }
  } else {
    // Every seam copy of a vertex gets the smooth normal of its position.
    result.normals = new Float32Array(full.firstVertex.length * 3);
    for (let w = 0; w < full.firstVertex.length; w++) {
      const p = byPosition.indices[full.firstCorner[w]];
      result.normals.set(smoothNormals!.subarray(p * 3, p * 3 + 3), w * 3);
    }
  }
  if (uv) result.uvs = gather(full.firstVertex, uv, 2);
  if (color) result.colors = gather(full.firstVertex, color, 3);
  return result;
}

function gather(vertices: Uint32Array, attribute: Attribute, size: number): Float32Array {
  const out = new Float32Array(vertices.length * size);
  for (let i = 0; i < vertices.length; i++) {
    for (let k = 0; k < size; k++) out[i * size + k] = attribute.getComponent(vertices[i], k);
  }
  return out;
}

/**
 * Same result as three's `mergeVertices` on the given channels (same quantization, vertices in first-use
 * order), but with an open-addressing hash table on the quantized values instead of string keys: about 10x
 * faster on million-triangle meshes. Returns, per welded vertex, the source vertex and corner that created it.
 */
function weld(
  count: number,
  vertexOf: (corner: number) => number,
  channels: { attribute: Attribute; size: number; tolerance: number }[]
) {
  const width = channels.reduce((a, c) => a + c.size, 0);
  const quantize = channels.map(({ tolerance }) => {
    const multiplier = Math.pow(10, Math.log10(1 / tolerance));
    return { multiplier, additive: tolerance * 0.5 * multiplier };
  });
  let size = 1;
  while (size < count * 2) size <<= 1;
  const mask = size - 1;
  const table = new Int32Array(size).fill(-1); // slot -> welded vertex
  const keys = new Float64Array(count * width); // quantized values per welded vertex (exact integers)
  const key = new Float64Array(width);
  const firstVertex = new Uint32Array(count);
  const firstCorner = new Uint32Array(count);
  const indices = new Uint32Array(count);
  let welded = 0;
  for (let i = 0; i < count; i++) {
    const v = vertexOf(i);
    let h = 0;
    let o = 0;
    for (let c = 0; c < channels.length; c++) {
      const { attribute, size: n } = channels[c];
      const { multiplier, additive } = quantize[c];
      for (let k = 0; k < n; k++, o++) {
        key[o] = Math.trunc(attribute.getComponent(v, k) * multiplier + additive);
        h = (h ^ Math.imul(key[o] | 0, HASH_PRIMES[o % HASH_PRIMES.length])) | 0;
      }
    }
    h &= mask;
    for (;;) {
      const w = table[h];
      if (w < 0) {
        table[h] = welded;
        keys.set(key, welded * width);
        firstVertex[welded] = v;
        firstCorner[welded] = i;
        indices[i] = welded++;
        break;
      }
      let same = true;
      for (let k = 0; k < width && same; k++) same = keys[w * width + k] === key[k];
      if (same) {
        indices[i] = w;
        break;
      }
      h = (h + 1) & mask;
    }
  }
  return { indices, firstVertex: firstVertex.slice(0, welded), firstCorner: firstCorner.slice(0, welded) };
}

const HASH_PRIMES = [73856093, 19349663, 83492791, 2654435761, 2246822519, 3266489917, 668265263, 374761393, 1181783497, 3735928559, 2028178513];

/**
 * Concatenate several already-welded meshes into one source mesh with per-vertex colors.
 * Each part uses its own `colors` if present, otherwise the given flat `color`.
 */
export function mergeSources(parts: { mesh: VirtualMeshSource; color?: THREE.ColorRepresentation }[]): VirtualMeshSource {
  const vertexCount = parts.reduce((a, p) => a + p.mesh.positions.length / 3, 0);
  const indexCount = parts.reduce((a, p) => a + p.mesh.indices.length, 0);
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const uvs = parts.some((p) => p.mesh.uvs) ? new Float32Array(vertexCount * 2) : null;
  const indices = new Uint32Array(indexCount);
  const c = new THREE.Color();
  let v = 0;
  let i = 0;
  for (const { mesh, color = 0xffffff } of parts) {
    const n = mesh.positions.length / 3;
    positions.set(mesh.positions, v * 3);
    normals.set(mesh.normals, v * 3);
    if (uvs && mesh.uvs) uvs.set(mesh.uvs, v * 2);
    if (mesh.colors) colors.set(mesh.colors, v * 3);
    else {
      c.set(color);
      for (let k = 0; k < n; k++) colors.set([c.r, c.g, c.b], (v + k) * 3);
    }
    for (let k = 0; k < mesh.indices.length; k++) indices[i + k] = mesh.indices[k] + v;
    v += n;
    i += mesh.indices.length;
  }
  return uvs ? { positions, normals, colors, uvs, indices } : { positions, normals, colors, indices };
}

/**
 * Shared GPU buffers and compute passes for many VirtualMeshes.
 *
 * Every mesh's vertices, meshlets, level table and instances are appended to one set of storage buffers, and one
 * fixed set of compute passes selects the cut for all of them. CPU cost, shader count and pipeline compilation
 * therefore do not grow with the number of unique meshes: a level with hundreds of different props costs the
 * same 12 dispatches per frame as a scene with one. Each mesh still gets its own indexed indirect draw (its own
 * material), into a region of the pool's shared index buffer laid out each frame by a prefix sum.
 *
 * Per frame (all on the GPU):
 *  0. reset    - clears the counters.
 *  1. cells    - one thread per cell (CELL_SIZE instances of one mesh, Morton-ordered): frustum, sub-pixel and
 *                draw-distance tests on the cell's bounding sphere. Visible cells are appended to a list.
 *  2. instance - one workgroup per visible cell: per-instance tests, occlusion, then the LOD level range per
 *                cut (camera and shadow). Instances whose few meshlets are all certain to pass emit them
 *                directly (fast path); the others go to a work list.
 *  3. meshlet  - one workgroup per work item: per-meshlet frustum/occlusion culling and the LOD test
 *                `parentError > t && error <= t`, widened to the blend band [t, t * k] when the threshold is high
 *                enough for level switches to show (see lodBlend.ts). Selected meshlets get a draw slot, their blend range and a range inside their
 *                mesh's region of the index buffer.
 *  4. prefix   - one thread: lays the meshes' regions out one after another and writes each mesh's draw args.
 *  5. expand   - one workgroup per drawn meshlet: writes 3 indices per triangle. An index encodes
 *                `drawSlot * 128 + meshletLocalVertex`; the material's positionNode decodes it and pulls the
 *                vertex from storage buffers, so any three.js NodeMaterial works unchanged.
 *
 * A pool is bounded by WebGPU's storage-buffer binding size; VirtualGeometry opens another pool when one is full.
 * Buffers grow by reallocation (the nodes are rebuilt, and the pipelines recompiled in the background).
 *
 * Based on nanite-webgpu (MIT, Marcin Matuszczyk): cullMeshletsPass + rasterizeHwPass.
 */
import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  bitcast,
  bool,
  dot,
  float,
  instanceIndex,
  length,
  localId,
  log,
  mat4,
  max,
  min,
  normalLocal,
  normalize,
  numWorkgroups,
  select,
  smoothstep,
  sqrt,
  storage,
  uint,
  uniform,
  unpackUnorm2x16,
  uvec4,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';
import { ERROR_INFINITY, FAST_PATH_MARGIN, FAST_PATH_MAX_MESHLETS, MAX_MESHLET_VERTICES, MESHLET_BOUNDS_STRIDE, MESHLET_INFO_STRIDE } from '../constants.js';
import type { VirtualMeshData } from '../preprocess/buildVirtualMesh.js';
import type { VirtualGeometry } from './VirtualGeometry.js';
import { vgUv } from './vgMaterial.js';
import { FULL_FADE_BITS, LOD_FADE_STEPS, vgFadeVarying } from './lodBlend.js';

const minNode = min as (a: unknown, b: unknown) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
const maxNode = max as (a: unknown, b: unknown) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
const logNode = log as (a: unknown) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
type Node = any; // TSL node typings are too strict for the graph code below; runtime is verified in the browser

/** World-space normal of the current vertex, usable in material color/roughness nodes. */
export const vgWorldNormal = varyingProperty('vec3', 'vVgWorldNormal');
export const vgMeshletVarying = varyingProperty('float', 'vVgMeshlet');
export const vgLodVarying = varyingProperty('float', 'vVgLod');
export const vgInstanceVarying = varyingProperty('float', 'vVgInstance');
export const vgTintVarying = varyingProperty('vec3', 'vVgTint');
/** World position of the current vertex's instance origin (vertex stage): e.g. to make whole trees sway from their base. */
export const vgInstanceOrigin = varyingProperty('vec3', 'vVgInstanceOrigin');

/** Bits of an index used for the meshlet-local vertex; the rest is the draw slot. */
export const LOCAL_VERTEX_BITS = Math.ceil(Math.log2(MAX_MESHLET_VERTICES));
const LOCAL_VERTEX_MASK = (1 << LOCAL_VERTEX_BITS) - 1;
const MAX_DISPATCH = 65535;
/** Instances per cell. A cell is a contiguous range of one mesh's Morton-sorted instances. */
export const CELL_SIZE = 128;
const CULL_GROUP = 64;
const MESHLET_GROUP = 64;
const EXPAND_GROUP = 64;
/** Level-table entries cached in workgroup memory by the instance pass (2 vec4 per level). */
const MAX_LEVELS = 48;
/** vec4s per mesh record in the mesh table. */
const MESH_STRIDE = 6;
/** vec4s per instance: 4 matrix columns, then (tint rgb, mesh slot). */
const INSTANCE_STRIDE = 5;
/** vec4s per cell: bounding sphere, then (max instance radius, mesh slot, first instance, instance count). */
const CELL_STRIDE = 2;
/** Fraction of `maxDrawDistance` over which instances shrink to nothing before they are culled. */
export const DRAW_DISTANCE_FADE = 0.15;
/** Stand-in for an infinite draw distance on the GPU. */
export const NO_DRAW_DISTANCE = 1e30;
/** Global counters: [0] visible cells, [1] occluded instances, then CUT_COUNTERS per cut. */
const CUT_COUNTERS = 5;
const C_SELECTED = 0;
const C_DRAWN = 1;
const C_SURVIVED = 2;
const C_WORK = 3;
const C_REQUESTED = 4;
const PER_MESH_COUNTERS = 2 + 2 * CUT_COUNTERS;
/**
 * GPU-resident capacity controller, per pool: per cut the scale applied to the error threshold, the fill of the last
 * cut (fraction of its draw buffer) and whether this frame's cut is being redone. Fixed point in the counters buffer,
 * after the per-mesh counters (a separate buffer would exceed WebGPU's 8 storage buffers per stage in the instance
 * pass); the reset pass leaves them alone, so they carry over from frame to frame.
 */
const S_SCALE = 0;
const S_FILL = 2;
const S_RETRY = 4;
const LOD_STATE_SIZE = 8;
/** The state is stored in fixed point: value * STATE_FIXED as an integer. */
const STATE_FIXED = 10000;
/** Extra selection rounds a frame may run when its cut overflows a draw buffer (each at a coarser threshold). */
const RETRY_ROUNDS = 2;
/** Fill the controller aims for, and the band in which it holds still. */
const FILL_TARGET = 0.82;
const FILL_HIGH = 0.9;
const FILL_LOW = 0.75;
/** Effective camera threshold (pixels) below which levels switch without blending, and from which the band is full. */
const LOD_BLEND_FROM = 1.25;
const LOD_BLEND_FULL = 2.5;

export interface PoolCapacity {
  /** Max meshlets drawn per frame, per cut. */
  meshlets: number;
  /** Max triangles drawn per frame by the camera cut (its index buffer). */
  triangles: number;
  /** Max triangles drawn per frame into shadow maps. */
  shadowTriangles: number;
}

/** CPU-side record of one mesh's place in the pool. */
export interface PoolEntry {
  slot: number;
  data: VirtualMeshData;
  vertexBase: number;
  meshletBase: number;
  meshletVertexBase: number;
  triangleBase: number;
  levelBase: number;
  levelCount: number;
  instanceBase: number;
  instanceCount: number;
  cellBase: number;
  cellCount: number;
  uvScale: [number, number];
  uvOffset: [number, number];
}

/** Growable typed array with a GPU attribute; growth replaces the attribute (the pool then rebuilds its nodes). */
class GrowBuffer<T extends Float32Array | Uint32Array> {
  array: T;
  attribute: THREE.StorageBufferAttribute;
  used = 0;
  constructor(
    private readonly make: (n: number) => T,
    readonly itemSize: number,
    capacity: number,
    /** Largest capacity (items) whose binding stays under the pool's byte limit. */
    private readonly maxItems = Infinity
  ) {
    this.array = make(Math.max(1, capacity) * itemSize);
    this.attribute = new THREE.StorageBufferAttribute(this.array, itemSize);
  }
  get capacity() {
    return this.array.length / this.itemSize;
  }
  /** Reserves `count` items; returns the first, or -1 when the buffer must grow first. */
  reserve(count: number) {
    if (this.used + count > this.capacity) return -1;
    const first = this.used;
    this.used += count;
    return first;
  }
  /** Reallocates to hold at least `needed` items (keeps the contents). */
  grow(needed: number) {
    // Grow ahead (fewer rebuilds), but never past the binding limit: the pool checked that `needed` fits.
    const next = this.make(Math.min(Math.max(needed, Math.ceil(this.capacity * 1.6)), Math.max(needed, this.maxItems)) * this.itemSize);
    next.set(this.array);
    this.array = next;
    this.attribute = new THREE.StorageBufferAttribute(this.array, this.itemSize);
  }
  /** Marks items [first, first + count) for upload. */
  touch(first: number, count: number) {
    if (count <= 0) return;
    this.attribute.addUpdateRange(first * this.itemSize, count * this.itemSize);
    this.attribute.needsUpdate = true;
  }
}

const f32 = (n: number) => new Float32Array(n);
const u32 = (n: number) => new Uint32Array(n);

export class GeometryPool {
  readonly entries: (PoolEntry | null)[] = [];
  /** Highest used mesh slot + 1 (the prefix pass loops over this many). */
  readonly meshSlots = uniform(0, 'uint');
  readonly cellCount = uniform(0, 'uint');
  readonly maxBytes: number;
  readonly capacity: PoolCapacity;

  // geometry
  readonly vertexData: GrowBuffer<Uint32Array>; // uvec4: position float bits xyz + octahedral normal
  readonly vertexAttrs: GrowBuffer<Uint32Array>; // uvec2: rgba8 color + uv (2 x unorm16 over the mesh's uv bounds)
  readonly meshletBounds: GrowBuffer<Float32Array>; // 4 x vec4 per meshlet
  readonly meshletInfo: GrowBuffer<Uint32Array>; // uvec4: triangleCount, firstIndex, lodLevel, vertexOffset (pool-global)
  readonly meshletVertices: GrowBuffer<Uint32Array>; // pool-global vertex ids
  readonly meshletTriangles: GrowBuffer<Uint32Array>; // 3 x u8 local ids
  // scene
  readonly meshTable: GrowBuffer<Float32Array>; // MESH_STRIDE vec4 per slot, then the level tables (2 vec4 per level)
  readonly instances: GrowBuffer<Float32Array>; // INSTANCE_STRIDE vec4 per instance
  readonly cells: GrowBuffer<Float32Array>; // CELL_STRIDE vec4 per cell
  private meshSlotCapacity: number;
  private levelUsed = 0;

  // per frame (rebuilt with the nodes)
  private countersAttribute!: THREE.StorageBufferAttribute;
  /** First word of the capacity controller's state in the counters buffer (see S_SCALE). */
  private lodStateBase = 0;
  /** From the last readback: the GPU controller is still moving its scale, so the cut must be redone every frame. */
  controllerActive = false;
  drawArgsAttribute!: THREE.IndirectStorageBufferAttribute;
  cameraIndex!: THREE.StorageBufferAttribute;
  shadowIndex!: THREE.StorageBufferAttribute;
  computeNodes: THREE.ComputeNode[] = [];
  positionNode: Node = null;
  /** Bumped on every rebuild: meshes re-point their geometry and material when it changes. */
  version = 0;
  private needsRebuild = true;
  private indexBuffersReady = false;
  pipelineState: 'none' | 'compiling' | 'ready' = 'none';
  /** The cut needs recomputing (camera moved, instances or settings changed). */
  dirty = true;
  /** True once a cut was computed (its camera index buffer can then serve as occluders). */
  hasCut = false;
  /** Holds one mesh too large for a shared pool, sized for it (no other mesh joins). */
  dedicated = false;
  /** Set when the device cannot bind this pool's buffers: it is skipped entirely and its meshes are hidden. */
  disabled = false;

  constructor(
    private readonly context: VirtualGeometry,
    capacity: PoolCapacity,
    maxBytes: number
  ) {
    this.capacity = capacity;
    this.maxBytes = maxBytes;
    this.meshSlotCapacity = 64;
    const limit = (bytesPerItem: number) => Math.floor(maxBytes / bytesPerItem);
    this.vertexData = new GrowBuffer(u32, 4, 1 << 16, limit(16));
    this.vertexAttrs = new GrowBuffer(u32, 2, 1 << 16, limit(8));
    this.meshletBounds = new GrowBuffer(f32, 16, 1 << 12, limit(64));
    this.meshletInfo = new GrowBuffer(u32, 4, 1 << 12, limit(16));
    this.meshletVertices = new GrowBuffer(u32, 1, 1 << 18, limit(4));
    this.meshletTriangles = new GrowBuffer(u32, 1, 1 << 18, limit(4));
    this.meshTable = new GrowBuffer(f32, 4, this.meshSlotCapacity * MESH_STRIDE + 1024, limit(16));
    this.meshTable.used = this.meshSlotCapacity * MESH_STRIDE;
    // In vec4s: whole instances only, and the work list (2 x 16 bytes per instance) stays smaller.
    this.instances = new GrowBuffer(f32, 4, 1 << 12, limit(16 * INSTANCE_STRIDE) * INSTANCE_STRIDE);
    this.cells = new GrowBuffer(f32, 4, 1 << 8, limit(16));
  }

  /** Bytes the geometry of `data` takes in this pool (the largest single binding is what must fit). */
  static bytesOf(data: VirtualMeshData) {
    return Math.max((data.positions.length / 3) * 16, data.meshletVertices.length * 4, data.meshletTriangles.length * 4, data.meshletCount * 64);
  }

  /** True when `data` (and `instanceCount` instances) still fit under the binding-size limit. */
  fits(data: VirtualMeshData, instanceCount: number) {
    const after = (used: number, add: number, bytesPerItem: number) => (used + add) * bytesPerItem <= this.maxBytes;
    return (
      after(this.vertexData.used, data.positions.length / 3, 16) &&
      after(this.meshletBounds.used, data.meshletCount, 64) &&
      after(this.meshletVertices.used, data.meshletVertices.length, 4) &&
      after(this.meshletTriangles.used, data.meshletTriangles.length, 4) &&
      after(this.instances.used, instanceCount * INSTANCE_STRIDE, 16)
    );
  }

  // ------------------------------------------------------------------ meshes

  /** Appends a mesh's geometry and instances. Returns its entry (`slot` is its index in the mesh table). */
  add(data: VirtualMeshData, matrices: Float32Array, colors: Float32Array | undefined, gpuToInput: Uint32Array): PoolEntry {
    let slot = this.entries.indexOf(null);
    if (slot === -1) slot = this.entries.length;
    if (slot >= this.meshSlotCapacity) this.growMeshSlots(slot + 1);

    const vertexCount = data.positions.length / 3;
    const levelCount = data.levelRanges.length / 2;
    const instanceCount = matrices.length / 16;
    const cellCount = Math.ceil(instanceCount / CELL_SIZE);
    const at = <T extends Float32Array | Uint32Array>(buffer: GrowBuffer<T>, count: number) => {
      let first = buffer.reserve(count);
      if (first === -1) {
        buffer.grow(buffer.used + count);
        this.needsRebuild = true;
        first = buffer.reserve(count);
      }
      return first;
    };
    const entry: PoolEntry = {
      slot,
      data,
      vertexBase: at(this.vertexData, vertexCount),
      meshletBase: at(this.meshletBounds, data.meshletCount),
      meshletVertexBase: at(this.meshletVertices, data.meshletVertices.length),
      triangleBase: at(this.meshletTriangles, data.meshletTriangles.length),
      levelBase: 0,
      levelCount,
      instanceBase: at(this.instances, instanceCount * INSTANCE_STRIDE) / INSTANCE_STRIDE,
      instanceCount,
      cellBase: at(this.cells, cellCount * CELL_STRIDE) / CELL_STRIDE,
      cellCount,
      uvScale: [1, 1],
      uvOffset: [0, 0],
    };
    at(this.meshletInfo, data.meshletCount);
    at(this.vertexAttrs, vertexCount);
    // Level table entries live after the mesh records.
    entry.levelBase = this.meshSlotCapacity * MESH_STRIDE + this.levelUsed;
    if (entry.levelBase + levelCount * 2 > this.meshTable.capacity) {
      this.meshTable.grow(entry.levelBase + levelCount * 2);
      this.needsRebuild = true;
    }
    this.levelUsed += levelCount * 2;
    this.meshTable.used = Math.max(this.meshTable.used, entry.levelBase + levelCount * 2);

    this.entries[slot] = entry;
    this.writeGeometry(entry);
    this.writeInstances(entry, matrices, colors, gpuToInput);
    this.writeMeshRecord(entry, { maxDrawDistance: NO_DRAW_DISTANCE, minPixelRadius: 0, enabled: 1, shadow: 1, occlusion: 0 });
    this.meshSlots.value = Math.max(this.meshSlots.value, slot + 1);
    this.cellCount.value = this.cells.used / CELL_STRIDE;
    this.dirty = true;
    return entry;
  }

  /** Frees a mesh's slot (its buffer ranges are reclaimed when the pool is next repacked). */
  remove(entry: PoolEntry) {
    this.writeFlags(entry, { enabled: 0 });
    this.entries[entry.slot] = null;
    // Its cells stay in the cell buffer but point at a disabled slot, so they are rejected at once.
    this.dirty = true;
  }

  private growMeshSlots(needed: number) {
    // The level tables start after the mesh records: move them up.
    const old = this.meshSlotCapacity;
    const next = Math.max(needed, old * 2);
    const shift = (next - old) * MESH_STRIDE;
    const levelsStart = old * MESH_STRIDE;
    const grownSize = this.meshTable.used + shift;
    if (grownSize > this.meshTable.capacity) this.meshTable.grow(grownSize);
    const table = this.meshTable.array;
    table.copyWithin((levelsStart + shift) * 4, levelsStart * 4, this.meshTable.used * 4);
    table.fill(0, levelsStart * 4, (levelsStart + shift) * 4);
    this.meshTable.used += shift;
    this.meshSlotCapacity = next;
    for (const entry of this.entries) {
      if (!entry) continue;
      entry.levelBase += shift;
      table[(entry.slot * MESH_STRIDE + 3) * 4 + 3] = entry.levelBase;
    }
    this.needsRebuild = true;
  }

  private writeGeometry(e: PoolEntry) {
    const d = e.data;
    const vertexCount = d.positions.length / 3;
    // Vertices: position bits + octahedral normal.
    const vd = this.vertexData.array;
    const asFloat = new Float32Array(vd.buffer, vd.byteOffset, vd.length);
    for (let i = 0; i < vertexCount; i++) {
      const o = (e.vertexBase + i) * 4;
      asFloat[o] = d.positions[i * 3];
      asFloat[o + 1] = d.positions[i * 3 + 1];
      asFloat[o + 2] = d.positions[i * 3 + 2];
      vd[o + 3] = encodeOctNormal(d.normals[i * 3], d.normals[i * 3 + 1], d.normals[i * 3 + 2]);
    }
    this.vertexData.touch(e.vertexBase, vertexCount);
    // Per-vertex color (white without colors) and uv (quantized over the mesh's uv bounds).
    const va = this.vertexAttrs.array;
    const uv = d.uvs ? packUvs(d.uvs) : null;
    if (uv) {
      e.uvScale = uv.scale;
      e.uvOffset = uv.offset;
    }
    const q = (x: number) => Math.round(Math.max(0, Math.min(1, x)) * 255);
    for (let i = 0; i < vertexCount; i++) {
      const o = (e.vertexBase + i) * 2;
      va[o] = d.colors ? (q(d.colors[i * 3]) | (q(d.colors[i * 3 + 1]) << 8) | (q(d.colors[i * 3 + 2]) << 16) | (255 << 24)) >>> 0 : 0xffffffff;
      va[o + 1] = uv ? uv.packed[i] : 0;
    }
    this.vertexAttrs.touch(e.vertexBase, vertexCount);
    // Meshlets, rebased to pool-global vertex / index ranges.
    this.meshletBounds.array.set(d.meshletBounds, e.meshletBase * 16);
    this.meshletBounds.touch(e.meshletBase, d.meshletCount);
    const info = this.meshletInfo.array;
    for (let m = 0; m < d.meshletCount; m++) {
      const o = (e.meshletBase + m) * 4;
      const s = m * MESHLET_INFO_STRIDE;
      info[o] = d.meshletInfo[s];
      info[o + 1] = d.meshletInfo[s + 1] + e.triangleBase * 3;
      info[o + 2] = d.meshletInfo[s + 2];
      info[o + 3] = d.meshletInfo[s + 3] + e.meshletVertexBase;
    }
    this.meshletInfo.touch(e.meshletBase, d.meshletCount);
    const mv = this.meshletVertices.array;
    for (let i = 0; i < d.meshletVertices.length; i++) mv[e.meshletVertexBase + i] = d.meshletVertices[i] + e.vertexBase;
    this.meshletVertices.touch(e.meshletVertexBase, d.meshletVertices.length);
    this.meshletTriangles.array.set(d.meshletTriangles, e.triangleBase);
    this.meshletTriangles.touch(e.triangleBase, d.meshletTriangles.length);
    // Level table: (first meshlet, count, smallest own error, largest parent error), (largest own, smallest parent).
    const t = this.meshTable.array;
    for (let l = 0; l < e.levelCount; l++) {
      const o = (e.levelBase + l * 2) * 4;
      t[o] = d.levelRanges[l * 2] + e.meshletBase;
      t[o + 1] = d.levelRanges[l * 2 + 1];
      t[o + 2] = d.levelErrors[l * 2];
      t[o + 3] = d.levelErrors[l * 2 + 1];
      t[o + 4] = d.levelGuardErrors[l * 2];
      t[o + 5] = d.levelGuardErrors[l * 2 + 1];
      t[o + 6] = 0;
      t[o + 7] = 0;
    }
    this.meshTable.touch(e.levelBase, e.levelCount * 2);
  }

  /** Writes a mesh record: bounding spheres, levers, level table location, uv packing, flags. */
  writeMeshRecord(e: PoolEntry, o: { maxDrawDistance: number; minPixelRadius: number; enabled: number; shadow: number; occlusion: number }) {
    const t = this.meshTable.array;
    const base = e.slot * MESH_STRIDE * 4;
    t.set(e.data.boundingSphere, base);
    t.set(e.data.lodBoundsSphere, base + 4);
    t.set(e.data.cullBoundsSphere, base + 8);
    t.set([o.maxDrawDistance, o.minPixelRadius, e.levelCount, e.levelBase], base + 12);
    t.set([e.uvScale[0], e.uvScale[1], e.uvOffset[0], e.uvOffset[1]], base + 16);
    t.set([o.enabled, o.shadow, o.occlusion, 0], base + 20);
    this.meshTable.touch(e.slot * MESH_STRIDE, MESH_STRIDE);
    this.dirty = true;
  }

  /** Updates some of a mesh's per-frame levers (only what changed is uploaded). */
  writeFlags(e: PoolEntry, o: { maxDrawDistance?: number; minPixelRadius?: number; enabled?: number; shadow?: number; occlusion?: number }) {
    const t = this.meshTable.array;
    const base = e.slot * MESH_STRIDE * 4;
    const set = (index: number, value: number | undefined) => {
      if (value === undefined || t[base + index] === value) return;
      t[base + index] = value;
      this.meshTable.touch(e.slot * MESH_STRIDE + (index >> 2), 1);
      this.dirty = true;
    };
    set(12, o.maxDrawDistance);
    set(13, o.minPixelRadius);
    set(20, o.enabled);
    set(21, o.shadow);
    set(22, o.occlusion);
  }

  readFlag(e: PoolEntry, index: 'maxDrawDistance' | 'minPixelRadius' | 'enabled' | 'shadow' | 'occlusion') {
    const offsets = { maxDrawDistance: 12, minPixelRadius: 13, enabled: 20, shadow: 21, occlusion: 22 };
    return this.meshTable.array[e.slot * MESH_STRIDE * 4 + offsets[index]];
  }

  private writeInstances(e: PoolEntry, matrices: Float32Array, colors: Float32Array | undefined, gpuToInput: Uint32Array) {
    const a = this.instances.array;
    for (let g = 0; g < e.instanceCount; g++) {
      const src = gpuToInput[g];
      const o = (e.instanceBase + g) * INSTANCE_STRIDE * 4;
      a.set(matrices.subarray(src * 16, src * 16 + 16), o);
      if (colors) a.set(colors.subarray(src * 4, src * 4 + 3), o + 16);
      else a.fill(1, o + 16, o + 19);
      a[o + 19] = e.slot;
    }
    this.instances.touch(e.instanceBase * INSTANCE_STRIDE, e.instanceCount * INSTANCE_STRIDE);
    for (let c = 0; c < e.cellCount; c++) this.computeCell(e, c);
    this.cells.touch(e.cellBase * CELL_STRIDE, e.cellCount * CELL_STRIDE);
  }

  /** Writes one instance matrix (GPU order index within the mesh). Call `commitInstances` afterwards. */
  setMatrix(e: PoolEntry, gpuIndex: number, matrix: THREE.Matrix4) {
    matrix.toArray(this.instances.array, (e.instanceBase + gpuIndex) * INSTANCE_STRIDE * 4);
  }

  /** Uploads changed instances and refreshes the bounds of the given cells. */
  commitInstances(e: PoolEntry, cells: Iterable<number>) {
    this.instances.touch(e.instanceBase * INSTANCE_STRIDE, e.instanceCount * INSTANCE_STRIDE);
    for (const c of cells) {
      this.computeCell(e, c);
      this.cells.touch((e.cellBase + c) * CELL_STRIDE, CELL_STRIDE);
    }
    this.dirty = true;
  }

  /**
   * Bounding sphere of one cell's instances: the AABB of their world-space object spheres gives the center, and
   * the radius reaches every instance sphere. Both radii are inflated a little so float32 rounding on the GPU can
   * never make the cell test stricter than the per-instance test.
   */
  private computeCell(e: PoolEntry, cell: number) {
    const m = this.instances.array;
    const first = cell * CELL_SIZE;
    const end = Math.min(first + CELL_SIZE, e.instanceCount);
    const s = new Float64Array(4);
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let maxRadius = 0;
    for (let g = first; g < end; g++) {
      instanceSphere(m, (e.instanceBase + g) * INSTANCE_STRIDE * 4, e.data.boundingSphere, s);
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], s[a] - s[3]);
        hi[a] = Math.max(hi[a], s[a] + s[3]);
      }
      maxRadius = Math.max(maxRadius, s[3]);
    }
    const cx = (lo[0] + hi[0]) / 2;
    const cy = (lo[1] + hi[1]) / 2;
    const cz = (lo[2] + hi[2]) / 2;
    let radius = 0;
    for (let g = first; g < end; g++) {
      instanceSphere(m, (e.instanceBase + g) * INSTANCE_STRIDE * 4, e.data.boundingSphere, s);
      radius = Math.max(radius, Math.hypot(s[0] - cx, s[1] - cy, s[2] - cz) + s[3]);
    }
    const o = (e.cellBase + cell) * CELL_STRIDE * 4;
    this.cells.array.set([cx, cy, cz, radius * (1 + 1e-4) + 1e-3, maxRadius * (1 + 1e-4) + 1e-4, e.slot, e.instanceBase + first, end - first], o);
  }

  // ------------------------------------------------------------------ GPU

  /**
   * Index buffers are written by compute and read by draws. three.js picks GPU buffer usage from the first use,
   * so create them as index buffers (INDEX | STORAGE) before any compute. Also rebuilds the nodes after growth.
   */
  prepare(renderer: THREE.WebGPURenderer) {
    this.ensureBuilt();
    if (this.indexBuffersReady) return;
    const backend = renderer.backend as unknown as { createIndexAttribute(a: unknown): void };
    backend.createIndexAttribute(this.cameraIndex);
    backend.createIndexAttribute(this.shadowIndex);
    this.indexBuffersReady = true;
  }

  /** Builds the nodes if the pool's buffers grew since the last build (or it was never built). */
  ensureBuilt() {
    if (this.needsRebuild) this.build();
  }

  /** Recreates the per-frame buffers and every node (after the pool's buffers grew). */
  build() {
    this.needsRebuild = false;
    this.indexBuffersReady = false;
    this.pipelineState = 'none';
    this.hasCut = false;
    this.version++;
    for (const node of this.computeNodes) node.dispose();

    const ctx = this.context;
    const cap = this.capacity;
    const slots = this.meshSlotCapacity;
    const maxDrawn = cap.meshlets;
    const instanceCapacity = Math.floor(this.instances.capacity / INSTANCE_STRIDE);
    const cellCapacity = Math.floor(this.cells.capacity / CELL_STRIDE);
    const cameraIndexCapacity = cap.triangles * 3;
    const shadowIndexCapacity = cap.shadowTriangles * 3;

    // Attribute objects of the growable buffers (replaced when they grew).
    const storageOf = (buffer: GrowBuffer<Float32Array | Uint32Array>, type: string): Node => (storage as Node)(buffer.attribute, type, buffer.capacity);
    const vertexData: Node = storageOf(this.vertexData, 'uvec4').toReadOnly();
    const vertexAttrs: Node = storageOf(this.vertexAttrs, 'uvec2').toReadOnly();
    const bounds: Node = storageOf(this.meshletBounds, 'vec4').toReadOnly();
    const info: Node = storageOf(this.meshletInfo, 'uvec4').toReadOnly();
    const meshletVertices: Node = storageOf(this.meshletVertices, 'uint').toReadOnly();
    const meshletTriangles: Node = storageOf(this.meshletTriangles, 'uint').toReadOnly();
    const meshTable: Node = storageOf(this.meshTable, 'vec4').toReadOnly();
    const instances: Node = storageOf(this.instances, 'vec4').toReadOnly();
    const cells: Node = storageOf(this.cells, 'vec4').toReadOnly();

    // Per-frame buffers.
    // counters: PER_MESH_COUNTERS globals, then per cut, mesh slot and part (requested indices, region start). Part 0
    // holds the meshlets drawn over every pixel, part 1 those inside an LOD blend band: only part 1 is drawn with
    // the blend mask, since a shader that may discard loses early depth testing (and hidden-surface removal).
    this.lodStateBase = PER_MESH_COUNTERS + 2 * slots * 4;
    const counterWords = new Uint32Array(this.lodStateBase + LOD_STATE_SIZE);
    counterWords.fill(STATE_FIXED, this.lodStateBase, this.lodStateBase + 2); // threshold scale 1 for both cuts
    this.countersAttribute = new THREE.StorageBufferAttribute(counterWords, 1);
    // drawIndexedIndirect args per cut and mesh slot: indexCount, instanceCount, firstIndex, baseVertex, firstInstance.
    this.drawArgsAttribute = new THREE.IndirectStorageBufferAttribute(new Uint32Array(2 * slots * 2 * 5), 1);
    this.cameraIndex = new THREE.StorageBufferAttribute(new Uint32Array(cameraIndexCapacity), 1);
    this.shadowIndex = new THREE.StorageBufferAttribute(new Uint32Array(shadowIndexCapacity), 1);
    const cellListAttribute = new THREE.StorageBufferAttribute(new Uint32Array(cellCapacity * 4), 4);
    // (instance, first meshlet, end meshlet, mesh slot | occlusion << 31) per cut.
    const workListAttribute = new THREE.StorageBufferAttribute(new Uint32Array(instanceCapacity * 2 * 4), 4);
    // (instance, meshlet, first index within the mesh's region, triangle count | blend range << 8: lo in bits 8-15, hi in
    // bits 16-23, see lodBlend.ts); slots are absolute over both cuts.
    const drawListAttribute = new THREE.StorageBufferAttribute(new Uint32Array(maxDrawn * 2 * 4), 4);
    const dispatchAttr = () => new THREE.IndirectStorageBufferAttribute(new Uint32Array(3), 3);
    const instanceArgsAttribute = dispatchAttr();
    const meshletArgsAttributes = [dispatchAttr(), dispatchAttr()];
    const expandArgsAttributes = [dispatchAttr(), dispatchAttr()];

    const retryInstanceArgsAttribute = dispatchAttr();
    const counters: Node = storage(this.countersAttribute, 'uint', this.countersAttribute.count).toAtomic();
    const stateWord = (i: number) => counters.element(this.lodStateBase + i);
    const getState = (i: number): Node => float((atomicLoad(stateWord(i)) as Node).toVar()).div(STATE_FIXED);
    const setState = (i: number, value: Node | number) => atomicStore(stateWord(i), uint(float(value).mul(STATE_FIXED).round()));
    const retryInstanceArgs: Node = storage(retryInstanceArgsAttribute, 'uint', 3);
    const drawArgs: Node = storage(this.drawArgsAttribute, 'uint', this.drawArgsAttribute.count);
    const cameraIndex: Node = storage(this.cameraIndex, 'uint', cameraIndexCapacity);
    const shadowIndex: Node = storage(this.shadowIndex, 'uint', shadowIndexCapacity);
    const cellList: Node = storage(cellListAttribute, 'uvec4', cellCapacity);
    const cellListRead: Node = storage(cellListAttribute, 'uvec4', cellCapacity).toReadOnly();
    const workList: Node = storage(workListAttribute, 'uvec4', instanceCapacity * 2);
    const drawList: Node = storage(drawListAttribute, 'uvec4', maxDrawn * 2);
    const drawListRead: Node = storage(drawListAttribute, 'uvec4', maxDrawn * 2).toReadOnly();
    const instanceArgs: Node = storage(instanceArgsAttribute, 'uint', 3);
    const meshletArgs = meshletArgsAttributes.map((a) => storage(a, 'uint', 3) as Node);
    const expandArgs = expandArgsAttributes.map((a) => storage(a, 'uint', 3) as Node);

    const counter = (cut: number, i: number) => counters.element(2 + cut * CUT_COUNTERS + i);
    const meshCount = (cut: number, slot: Node, part: Node | number) => counters.element(uint(PER_MESH_COUNTERS + cut * slots * 4).add(slot.mul(4)).add(uint(part).mul(2)));
    const meshStart = (cut: number, slot: Node, part: Node | number) => counters.element(uint(PER_MESH_COUNTERS + cut * slots * 4 + 1).add(slot.mul(4)).add(uint(part).mul(2)));
    /** Part of a drawn meshlet from its packed blend range (see drawList): 0 drawn everywhere, 1 blended. */
    const partOf = (packed: Node) => select(packed.bitAnd(0xffff00).equal(FULL_FADE_BITS), uint(0), uint(1));
    const cutCapacity = [cameraIndexCapacity, shadowIndexCapacity];
    const indexBuffers = [cameraIndex, shadowIndex];

    // Per cut, the band of thresholds whose cuts are drawn, [t, t * k] (see lodBlend.ts), where t is the error
    // threshold times the scale of the capacity controller (1 unless the pool's draw buffers would overflow). Shadow
    // maps use a coarser band. The band widens with the camera's effective threshold: at about a pixel of error a
    // level switch is not visible, so levels switch directly (no blending cost); from LOD_BLEND_FULL pixels up the
    // band is `lodBlend` wide. loadBands() reads all this into variables at the top of each kernel that needs it (the
    // scale is an atomic, and atomics are statements in TSL, not expressions).
    let bands: { low: Node; high: Node; fadeScale: Node }[] = [];
    const loadBands = () => {
      const cameraScale = getState(S_SCALE).toVar();
      const shadowScale = getState(S_SCALE + 1).toVar();
      const width: Node = float(1).add((ctx.lodBlendMax as Node).sub(1).mul(smoothstep(LOD_BLEND_FROM, LOD_BLEND_FULL, (ctx.errorThreshold as Node).mul(cameraScale)))).toVar();
      const blending: Node = width.greaterThan(1.001).toVar();
      const fadeScale = select(blending, float(LOD_FADE_STEPS).div(logNode(width)), float(0)).toVar();
      bands = [0, 1].map((cut) => {
        // Shadows are never finer than the camera's detail: they coarsen with it, and on their own when their buffer fills.
        const scale = cut === 0 ? cameraScale : maxNode(cameraScale, shadowScale);
        const t: Node = (cut === 0 ? ctx.errorThreshold : ctx.errorThreshold.mul(ctx.shadowErrorScale)).mul(scale).toVar();
        return { low: t, high: select(blending, t.mul(width), t).toVar(), fadeScale };
      });
    };
    const minPixelsGlobal: Node = ctx.minPixelRadius;
    const occlusion = ctx.occlusion;

    const instanceColumn = (instanceId: Node, k: number) => instances.element(instanceId.mul(INSTANCE_STRIDE).add(k));
    const modelOf = (instanceId: Node) => {
      const c = [0, 1, 2, 3].map((k) => instanceColumn(instanceId, k).toVar());
      const model = mat4(c[0], c[1], c[2], c[3]).toVar();
      const scale = max(length(c[0].xyz), max(length(c[1].xyz), length(c[2].xyz))).toVar();
      return { model, scale, columns: c };
    };
    const meshRecord = (slot: Node, k: number) => meshTable.element(slot.mul(MESH_STRIDE).add(k));

    const projectedError = (model: Node, scale: Node, sphere: Node, error: Node) => {
      const viewCenter: Node = (ctx.viewMatrix as Node).mul(model.mul(vec4(sphere.xyz, 1)));
      const distance = max(length(viewCenter.xyz).sub(sphere.w.mul(scale)), ctx.near) as Node;
      const pixels = error.mul(scale).div(distance).mul(ctx.projScale);
      return select(error.greaterThanEqual(ERROR_INFINITY * 0.5), float(ERROR_INFINITY), pixels);
    };

    /**
     * Signed distance of a sphere to the camera frustum (negative radius-or-more = fully outside). With `swept`,
     * the sphere is swept along shadowSweep (the light direction): a caster outside the view can still throw a
     * shadow into it, so it is kept if ANY point of that segment is inside the plane: d(center) + max(0, dot(n, sweep)).
     */
    const frustumDistance = (center: Node, swept: Node | boolean) => {
      let minDistance: Node = float(1e30);
      for (const plane of ctx.frustumPlanes) {
        let d: Node = dot(plane.xyz, center).add(plane.w);
        if (swept !== false) d = d.add(max(dot(plane.xyz, ctx.shadowSweep), 0).mul(swept === true ? 1 : select(swept, float(1), float(0))));
        minDistance = min(minDistance, d);
      }
      return minDistance;
    };

    /** Hidden behind the occluders; the sphere grows by the error threshold (occluders are LOD-simplified). */
    const occludedSphere = (occlusionOn: Node, worldCenter: Node, radius: Node) => {
      const viewCenter: Node = (ctx.viewMatrix as Node).mul(vec4(worldCenter, 1)).xyz.toVar();
      const margin = viewCenter.z.abs().mul(bands[0].high).div(ctx.projScale);
      return occlusionOn.and((occlusion.activeNode as Node).equal(1)).and(occlusion.occluded(viewCenter, radius.add(margin)));
    };

    const dispatchArgs = (target: Node, count: Node) => {
      target.element(0).assign(minNode(count, uint(MAX_DISPATCH)));
      target.element(1).assign(count.add(MAX_DISPATCH - 1).div(MAX_DISPATCH));
      target.element(2).assign(uint(1));
    };

    // ---------- 0. reset ----------
    const resetNode = Fn(() => {
      const i = instanceIndex;
      If(i.lessThan(uint(PER_MESH_COUNTERS)), () => {
        atomicStore(counters.element(i), uint(0));
      });
      If(i.lessThan(this.meshSlots.mul(8)), () => {
        // count and start of both parts of slot i>>2, for both cuts
        const cut = i.div(this.meshSlots.mul(4));
        const rest = i.mod(this.meshSlots.mul(4));
        atomicStore(counters.element(uint(PER_MESH_COUNTERS).add(cut.mul(slots * 4)).add(rest)), uint(0));
      });
    })().compute(Math.max(PER_MESH_COUNTERS, slots * 8), [64]);

    // ---------- 1. cells ----------
    const cellNode = Fn(() => {
      If(instanceIndex.greaterThanEqual(this.cellCount), () => {
        Return();
      });
      const cell = instanceIndex.toVar();
      const sphere = cells.element(cell.mul(CELL_STRIDE)).toVar();
      const record = cells.element(cell.mul(CELL_STRIDE).add(1)).toVar();
      const slot = uint(record.y).toVar();
      const flags = meshRecord(slot, 5).toVar();
      If(flags.x.equal(0), () => {
        Return(); // removed or invisible mesh
      });
      const center = sphere.xyz.toVar();
      const radius = sphere.w.toVar();
      If(ctx.frustumCulling.equal(1).and(frustumDistance(center, flags.y.greaterThan(0.5)).lessThan(radius.negate())), () => {
        Return();
      });
      const levers = meshRecord(slot, 3).toVar();
      const minPixels = max(minPixelsGlobal, levers.y);
      // Every instance lies within `radius` of the cell center, so its view distance is at least
      // viewDistance - radius, and its radius is at most maxRadius.
      const maxRadius = record.x;
      const viewDistance = length((ctx.viewMatrix as Node).mul(vec4(center, 1)).xyz).toVar();
      const gap = viewDistance.sub(radius).toVar();
      If(gap.greaterThan(maxRadius).and(maxRadius.mul(ctx.projScale).div(max(gap, 1e-6)).lessThan(minPixels)), () => {
        Return();
      });
      If(gap.greaterThan(levers.x), () => {
        Return(); // beyond the draw distance
      });
      const at = (atomicAdd(counters.element(0), uint(1)) as Node).toVar();
      cellList.element(at).assign(uvec4(slot, uint(record.z), uint(record.w), cell));
    })().compute(cellCapacity, [CULL_GROUP]);

    const cellArgsNode = Fn(() => {
      dispatchArgs(instanceArgs, (atomicLoad(counters.element(0)) as Node).toVar());
    })().compute(1);

    // ---------- 2. instances ----------
    // The mesh's level table is cached in workgroup memory: every instance of a cell belongs to one mesh.
    const levelCache: Node = workgroupArray('vec4', MAX_LEVELS * 2);

    /** Emits the level range of one instance for one cut, then its fast path or its work-list entry. */
    const selectLevels = (
      cut: number,
      slot: Node,
      levelCount: Node,
      occlusionFlag: Node,
      cullBounds: Node,
      instanceId: Node,
      model: Node,
      scale: Node,
      pixelsAtMin: Node,
      pixelsAtMax: Node
    ) => {
      const { low, high } = bands[cut];
      const first = uint(0xffffffff).toVar();
      const last = uint(0).toVar();
      const possibleMeshlets = uint(0).toVar();
      const guaranteedMeshlets = uint(0).toVar();
      Loop({ start: uint(0), end: levelCount, type: 'uint', condition: '<' }, ({ i }: { i: Node }) => {
        const l0 = levelCache.element(i.mul(2)).toVar(); // first meshlet, count, smallest own error, largest parent error
        const l1 = levelCache.element(i.mul(2).add(1)).toVar(); // largest own error, smallest parent error
        const levelFirst = uint(l0.x).toVar();
        const size = uint(l0.y).toVar();
        const possible = l0.z
          .mul(pixelsAtMax)
          .lessThanEqual(high)
          .and(l0.w.greaterThanEqual(ERROR_INFINITY * 0.5).or(l0.w.mul(pixelsAtMin).greaterThan(low)));
        // Every meshlet of the level is drawn over every pixel for any distance in [min, max]: largest own error
        // at the nearest distance below the band, smallest parent error at the farthest above it. Relative margin
        // keeps float rounding on our side.
        const sure = l1.x
          .mul(pixelsAtMin)
          .lessThanEqual(low.mul(1 - FAST_PATH_MARGIN))
          .and(l1.y.greaterThanEqual(ERROR_INFINITY * 0.5).or(l1.y.mul(pixelsAtMax).greaterThan(high.mul(1 + FAST_PATH_MARGIN))));
        If(possible, () => {
          first.assign(minNode(first, levelFirst));
          last.assign(maxNode(last, levelFirst.add(size)));
          possibleMeshlets.addAssign(size);
          If(sure, () => {
            guaranteedMeshlets.addAssign(size);
          });
        });
      });

      const cullCenter = model.mul(vec4(cullBounds.xyz, 1)).xyz.toVar();
      const cullRadius = cullBounds.w.mul(scale).toVar();
      // Possible levels must be contiguous, all guaranteed, few, and every meshlet inside the frustum.
      const fastPath = possibleMeshlets
        .greaterThan(uint(0))
        .and(possibleMeshlets.equal(last.sub(first)))
        .and(guaranteedMeshlets.equal(possibleMeshlets))
        .and(possibleMeshlets.lessThanEqual(uint(FAST_PATH_MAX_MESHLETS)))
        .and(ctx.frustumCulling.equal(0).or(frustumDistance(cullCenter, cut === 1).greaterThanEqual(cullRadius)))
        .toVar();

      If(last.greaterThan(first), () => {
        atomicAdd(counter(cut, C_SURVIVED), uint(1));
        If(fastPath, () => {
          const base = (atomicAdd(counter(cut, C_SELECTED), possibleMeshlets) as Node).toVar();
          for (let k = 0; k < FAST_PATH_MAX_MESHLETS; k++) {
            If(uint(k).lessThan(possibleMeshlets).and(base.add(k).lessThan(uint(maxDrawn))), () => {
              const meshletId = first.add(k).toVar();
              const triangleCount = info.element(meshletId).x.toVar();
              const firstIndex = (atomicAdd(meshCount(cut, slot, 0), triangleCount.mul(3)) as Node).toVar();
              drawList.element(base.add(k + cut * maxDrawn)).assign(uvec4(instanceId, meshletId, firstIndex, triangleCount.bitOr(FULL_FADE_BITS)));
            });
          }
        }).Else(() => {
          const item = (atomicAdd(counter(cut, C_WORK), uint(1)) as Node).toVar();
          const tag = slot.bitOr(select(occlusionFlag, uint(0x80000000), uint(0)));
          workList.element(item.add(cut * instanceCapacity)).assign(uvec4(instanceId, first, last, tag));
        });
      });
    };

    /** Instance pass; `retry`: a later round of the same frame, redoing only the cuts that overflowed. */
    const instancePass = (retry: boolean) => Fn(() => {
      // No early return before the barrier (WGSL needs uniform control flow there): workgroups past the
      // visible-cell count (the dispatch is rounded up) load no levels and process no instances.
      const cellIndex = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x)).toVar();
      const valid = cellIndex.lessThan(atomicLoad(counters.element(0)) as Node).toVar();
      const cell = cellListRead.element(minNode(cellIndex, uint(cellCapacity - 1))).toVar(); // (mesh slot, first instance, count, cell)
      const slot = cell.x;
      const levers = meshRecord(slot, 3).toVar(); // draw distance, min pixel radius, level count, level base
      const levelCount = select(valid, minNode(uint(levers.z), uint(MAX_LEVELS)), uint(0)).toVar();
      const levelBase = uint(levers.w).toVar();
      Loop({ start: localId.x, end: levelCount.mul(2), type: 'uint', condition: '<', update: CULL_GROUP }, ({ i }: { i: Node }) => {
        levelCache.element(i).assign(meshTable.element(levelBase.add(i)));
      });
      workgroupBarrier();
      If(valid.not(), () => {
        Return();
      });
      loadBands();

      const objectSphere = meshRecord(slot, 0).toVar();
      const lodBounds = meshRecord(slot, 1).toVar();
      const cullBounds = meshRecord(slot, 2).toVar();
      const flags = meshRecord(slot, 5).toVar(); // enabled, shadow, occlusion
      const shadowOn = flags.y.greaterThan(0.5);
      const occlusionOn = flags.z.greaterThan(0.5);
      const minPixels = max(minPixelsGlobal, levers.y).toVar();
      const maxDrawDistance = levers.x.toVar();
      const end = cell.y.add(cell.z);
      Loop({ start: cell.y.add(localId.x), end, type: 'uint', condition: '<', update: CULL_GROUP }, ({ i }: { i: Node }) => {
        const instanceId = i.toVar();
        const { model, scale } = modelOf(instanceId);
        const center = model.mul(vec4(objectSphere.xyz, 1)).xyz.toVar();
        const radius = objectSphere.w.mul(scale).toVar();
        const noFrustum = ctx.frustumCulling.equal(0);
        const inCamera = noFrustum.or(frustumDistance(center, false).greaterThanEqual(radius.negate())).toVar();
        const inShadow = shadowOn.and(noFrustum.or(frustumDistance(center, true).greaterThanEqual(radius.negate()))).toVar();
        if (retry) {
          // Atomics are statements in TSL: load into variables before using them in expressions.
          const redoCamera = getState(S_RETRY).toVar();
          const redoShadow = getState(S_RETRY + 1).toVar();
          inCamera.assign(inCamera.and(redoCamera.greaterThan(0.5)));
          inShadow.assign(inShadow.and(redoShadow.greaterThan(0.5)));
        }
        // Sub-pixel instances and instances beyond the draw distance: both cuts.
        const viewDistance = length((ctx.viewMatrix as Node).mul(vec4(center, 1)).xyz).toVar();
        const tooSmall = viewDistance.greaterThan(radius).and(radius.mul(ctx.projScale).div(viewDistance).lessThan(minPixels));
        const tooFar = viewDistance.sub(radius).greaterThan(maxDrawDistance);
        const keep = inCamera.or(inShadow).and(tooSmall.not()).and(tooFar.not());
        If(keep, () => {
          If(inCamera.and(occludedSphere(occlusionOn, center, radius)), () => {
            inCamera.assign(bool(false));
            atomicAdd(counters.element(1), uint(1));
          });

          const lodCenter: Node = (ctx.viewMatrix as Node).mul(model.mul(vec4(lodBounds.xyz, 1)));
          const d = length(lodCenter.xyz).toVar();
          const r = lodBounds.w.mul(scale);
          const pixelsAtMin = scale.mul(ctx.projScale).div(max(d.sub(r), ctx.near)).toVar();
          const pixelsAtMax = scale.mul(ctx.projScale).div(max(d.add(r), ctx.near)).toVar();
          If(inCamera, () => {
            selectLevels(0, slot, levelCount, occlusionOn, cullBounds, instanceId, model, scale, pixelsAtMin, pixelsAtMax);
          });
          If(inShadow, () => {
            selectLevels(1, slot, levelCount, bool(false), cullBounds, instanceId, model, scale, pixelsAtMin, pixelsAtMax);
          });
        });
      });
    })().compute((retry ? retryInstanceArgsAttribute : instanceArgsAttribute) as unknown as number, [CULL_GROUP]);

    // ---------- 3. meshlets ----------
    const meshletPass = (cut: number) => {
      const argsNode = Fn(() => {
        dispatchArgs(meshletArgs[cut], (atomicLoad(counter(cut, C_WORK)) as Node).toVar());
      })().compute(1);
      // In a retry round, only a cut being redone runs again (the others keep their work list and selection).
      const retryArgsNode = Fn(() => {
        const redo = getState(S_RETRY + cut).toVar();
        const work = (atomicLoad(counter(cut, C_WORK)) as Node).toVar();
        dispatchArgs(meshletArgs[cut], select(redo.greaterThan(0.5), work, uint(0)).toVar());
      })().compute(1);
      const passNode = Fn(() => {
        const item = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x)).toVar();
        If(item.greaterThanEqual(atomicLoad(counter(cut, C_WORK)) as Node), () => {
          Return();
        });
        const work = workList.element(item.add(cut * instanceCapacity)).toVar();
        const instanceId = work.x;
        const slot = work.w.bitAnd(0x7fffffff).toVar();
        const occlusionOn = work.w.shiftRight(31).equal(1);
        const { model, scale } = modelOf(instanceId);
        loadBands();
        const { low, high, fadeScale } = bands[cut];
        Loop({ start: work.y.add(localId.x), end: work.z, type: 'uint', condition: '<', update: MESHLET_GROUP }, ({ i }: { i: Node }) => {
          const meshletId = i;
          const boundsBase = meshletId.mul(MESHLET_BOUNDS_STRIDE / 4).toVar();
          const cullSphere = bounds.element(boundsBase.add(3));
          const center = model.mul(vec4(cullSphere.xyz, 1)).xyz;
          let visible: Node = ctx.frustumCulling.equal(0).or(frustumDistance(center, cut === 1).greaterThanEqual(cullSphere.w.mul(scale).negate()));
          if (cut === 0) visible = visible.and(occludedSphere(occlusionOn, center, cullSphere.w.mul(scale)).not());
          // LOD selection: draw this meshlet iff it is in the cut for some threshold of the band: its own error
          // is small enough for the band's top, its parent's too large for the band's bottom.
          const errors = bounds.element(boundsBase.add(2));
          const ownError = projectedError(model, scale, bounds.element(boundsBase), errors.x).toVar();
          const parentError = projectedError(model, scale, bounds.element(boundsBase.add(1)), errors.y).toVar();
          // Blend range: the steps of the band (log scale) between its own and its parent's error. Equal errors
          // give equal steps, so a meshlet and the ones replacing it split the pixels between them exactly.
          // Shadow maps blend the same way, per shadow-map texel (smoothed by the shadow filter).
          const step = (error: Node) => uint(logNode(maxNode(error, 1e-30).div(low)).mul(fadeScale).round().clamp(0, LOD_FADE_STEPS));
          const fadeLow = select(fadeScale.greaterThan(0), step(ownError), uint(0)).toVar();
          const fadeHigh = select(fadeScale.greaterThan(0), step(parentError), uint(LOD_FADE_STEPS)).toVar();
          const fade = fadeLow.shiftLeft(8).bitOr(fadeHigh.shiftLeft(16));
          const drawn = visible.and(ownError.lessThanEqual(high)).and(parentError.greaterThan(low)).and(fadeHigh.greaterThan(fadeLow));
          If(drawn, () => {
            const at = (atomicAdd(counter(cut, C_SELECTED), uint(1)) as Node).toVar();
            If(at.lessThan(uint(maxDrawn)), () => {
              const triangleCount = info.element(meshletId).x.toVar();
              const packed = triangleCount.bitOr(fade).toVar();
              const firstIndex = (atomicAdd(meshCount(cut, slot, partOf(packed)), triangleCount.mul(3)) as Node).toVar();
              drawList.element(at.add(cut * maxDrawn)).assign(uvec4(instanceId, meshletId, firstIndex, packed));
            });
          });
        });
      })().compute(meshletArgsAttributes[cut] as unknown as number, [MESHLET_GROUP]);
      return { argsNode, retryArgsNode, passNode };
    };

    // ---------- 4. prefix: lay the meshes' regions out, write draw args and the expand dispatch ----------
    const prefixPass = (cut: number) =>
      Fn(() => {
        const start = uint(0).toVar();
        const capacity = uint(cutCapacity[cut]);
        Loop({ start: uint(0), end: this.meshSlots, type: 'uint', condition: '<' }, ({ i }: { i: Node }) => {
          for (const part of [0, 1]) {
            const requested = (atomicLoad(meshCount(cut, i, part)) as Node).toVar();
            atomicStore(meshStart(cut, i, part), start);
            const room = select(start.lessThan(capacity), capacity.sub(start), uint(0));
            const args = uint(cut * slots * 10 + part * 5).add(i.mul(10)).toVar();
            drawArgs.element(args).assign(minNode(requested, room));
            drawArgs.element(args.add(1)).assign(uint(1));
            drawArgs.element(args.add(2)).assign(start);
            drawArgs.element(args.add(3)).assign(uint(0));
            drawArgs.element(args.add(4)).assign(uint(0));
            start.addAssign(requested);
          }
        });
        atomicStore(counter(cut, C_REQUESTED), start);
        const drawn = minNode(atomicLoad(counter(cut, C_SELECTED)) as Node, uint(maxDrawn)).toVar();
        atomicStore(counter(cut, C_DRAWN), drawn);
        dispatchArgs(expandArgs[cut], drawn);
      })().compute(1);

    // ---------- 5. expand ----------
    const expandPass = (cut: number) =>
      Fn(() => {
        const at = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x)).toVar();
        If(at.greaterThanEqual(atomicLoad(counter(cut, C_DRAWN)) as Node), () => {
          Return();
        });
        const drawSlot = at.add(cut * maxDrawn).toVar();
        const entry = drawListRead.element(drawSlot).toVar();
        const slot = uint(instanceColumn(entry.x, 4).w).toVar();
        const regionStart = (atomicLoad(meshStart(cut, slot, partOf(entry.w))) as Node).toVar();
        const firstTriangle = info.element(entry.y).y.div(3).toVar();
        const base = drawSlot.shiftLeft(LOCAL_VERTEX_BITS).toVar();
        const buffer = indexBuffers[cut];
        const triangleCount = entry.w.bitAnd(255).toVar(); // the upper bits hold the blend range
        Loop({ start: localId.x, end: triangleCount, type: 'uint', condition: '<', update: EXPAND_GROUP }, ({ i }: { i: Node }) => {
          const index = regionStart.add(entry.z).add(i.mul(3)).toVar();
          If(index.add(2).lessThan(uint(cutCapacity[cut])), () => {
            const packed = meshletTriangles.element(firstTriangle.add(i)).toVar();
            buffer.element(index).assign(base.bitOr(packed.bitAnd(255)));
            buffer.element(index.add(1)).assign(base.bitOr(packed.shiftRight(8).bitAnd(255)));
            buffer.element(index.add(2)).assign(base.bitOr(packed.shiftRight(16).bitAnd(255)));
          });
        });
      })().compute(expandArgsAttributes[cut] as unknown as number, [EXPAND_GROUP]);

    // ---------- capacity: redo an overflowing cut in the same frame, steer the threshold scale between frames ----------
    // A cut that does not fit its draw buffers would lose meshlets in arbitrary order (holes that flicker). The
    // controller is on the GPU, so it reacts in the frame where the demand jumps (fast zoom, camera cut), not frames
    // later after a readback: an overflowing cut is selected again at a coarser threshold before anything is drawn.
    const fillOf = (cut: number): Node => {
      const requested = (atomicLoad(counter(cut, C_REQUESTED)) as Node).toVar();
      const selected = (atomicLoad(counter(cut, C_SELECTED)) as Node).toVar();
      return maxNode(float(requested).div(cutCapacity[cut]), float(selected).div(maxDrawn));
    };
    const checkNode = Fn(() => {
      const any = bool(false).toVar();
      for (const cut of [0, 1]) {
        const fill = fillOf(cut).toVar();
        const scale = getState(S_SCALE + cut).toVar();
        setState(S_RETRY + cut, 0);
        If(fill.greaterThan(1).and(scale.lessThan(ctx.lodScaleLimit)), () => {
          // Triangles go with about 1 / threshold^2: aim at the target fill, at least 10% coarser.
          setState(S_SCALE + cut, minNode(scale.mul(sqrt(fill.div(FILL_TARGET)).clamp(1.1, 4)), ctx.lodScaleLimit));
          setState(S_RETRY + cut, 1);
          any.assign(true);
          for (const c of [C_SELECTED, C_DRAWN, C_SURVIVED, C_WORK, C_REQUESTED]) atomicStore(counter(cut, c), uint(0));
          if (cut === 0) atomicStore(counters.element(1), uint(0));
          Loop({ start: uint(0), end: this.meshSlots, type: 'uint', condition: '<' }, ({ i }: { i: Node }) => {
            atomicStore(meshCount(cut, i, 0), uint(0));
            atomicStore(meshCount(cut, i, 1), uint(0));
          });
        });
      }
      retryInstanceArgs.element(0).assign(select(any, instanceArgs.element(0), uint(0)));
      retryInstanceArgs.element(1).assign(select(any, instanceArgs.element(1), uint(1)));
      retryInstanceArgs.element(2).assign(uint(1));
    })().compute(1);
    // Between frames: coarsen a little when a buffer is nearly full, give detail back while there is room.
    const steerNode = Fn(() => {
      for (const cut of [0, 1]) {
        const fill = fillOf(cut).toVar();
        const scale = getState(S_SCALE + cut).toVar();
        If(fill.greaterThan(FILL_HIGH), () => {
          scale.assign(scale.mul(minNode(sqrt(fill.div(FILL_TARGET)), 1.03)));
        }).ElseIf(fill.lessThan(FILL_LOW), () => {
          scale.assign(scale.div(1.01));
        });
        // Within [1, limit] (the limit drops when the user raises the threshold).
        setState(S_SCALE + cut, maxNode(minNode(scale, ctx.lodScaleLimit), 1));
        setState(S_FILL + cut, fill);
      }
    })().compute(1);

    const meshlets = [meshletPass(0), meshletPass(1)];
    const prefixes = [prefixPass(0), prefixPass(1)];
    const instanceNode = instancePass(false);
    const instanceRetryNode = instancePass(true);
    const retryRounds: THREE.ComputeNode[] = [];
    for (let round = 0; round < RETRY_ROUNDS; round++) {
      retryRounds.push(checkNode, instanceRetryNode);
      for (const m of meshlets) retryRounds.push(m.retryArgsNode, m.passNode);
      retryRounds.push(...prefixes);
    }
    this.computeNodes = [
      resetNode,
      cellNode,
      cellArgsNode,
      instanceNode,
      ...meshlets.flatMap((m) => [m.argsNode, m.passNode]),
      ...prefixes,
      ...retryRounds,
      steerNode,
      expandPass(0),
      expandPass(1),
    ];

    // ---------- vertex pulling (shared by every mesh of the pool, both cuts) ----------
    this.positionNode = Fn(() => {
      const slot = vertexIndex.shiftRight(LOCAL_VERTEX_BITS);
      const localVertex = vertexIndex.bitAnd(LOCAL_VERTEX_MASK);
      const entry = drawListRead.element(slot).toVar();
      const instanceId = entry.x;
      const meshletId = entry.y;
      const meshletInfo = info.element(meshletId).toVar();
      const vertexId: Node = meshletVertices.element(meshletInfo.w.add(localVertex)).toVar();

      const { model, columns } = modelOf(instanceId);
      const tintAndSlot = instanceColumn(instanceId, 4).toVar();
      const meshSlot = uint(tintAndSlot.w).toVar();
      const levers = meshRecord(meshSlot, 3);
      const uvPacking = meshRecord(meshSlot, 4).toVar();
      const packedVertex = vertexData.element(vertexId).toVar();
      const attrs = vertexAttrs.element(vertexId).toVar();
      const localPosition: Node = bitcast(packedVertex.xyz, 'vec3');
      // Draw-distance fade: shrink toward the instance origin over the last DRAW_DISTANCE_FADE of the distance.
      // Distance from the main camera (context.viewMatrix), also in the shadow pass, so shadows fade with it.
      const maxDrawDistance = levers.x;
      const origin = columns[3].xyz;
      vgInstanceOrigin.assign(origin);
      const originDistance = length((ctx.viewMatrix as Node).mul(vec4(origin, 1)).xyz);
      const fade = maxDrawDistance.sub(originDistance).div(maxDrawDistance.mul(DRAW_DISTANCE_FADE)).clamp(0, 1);
      const worldPosition = origin.add(model.mul(vec4(localPosition, 0)).xyz.mul(fade));
      const worldNormal = normalize(model.mul(vec4(decodeOctNormal(packedVertex.w), 0)).xyz);
      normalLocal.assign(worldNormal);
      vgWorldNormal.assign(worldNormal);

      vgMeshletVarying.assign(float(meshletId));
      vgFadeVarying.assign(vec2(float(entry.w.shiftRight(8).bitAnd(255)), float(entry.w.shiftRight(16).bitAnd(255))));
      vgLodVarying.assign(float(meshletInfo.z));
      vgInstanceVarying.assign(float(instanceId));
      vgTintVarying.assign(tintAndSlot.xyz.mul(unpackRgba8(attrs.x).xyz));
      vgUv.assign((unpackUnorm2x16(attrs.y) as Node).mul(uvPacking.xy).add(uvPacking.zw));
      return worldPosition;
    })();
  }

  /** Stats of the last computed cut, from a readback of the counters buffer. */
  async readCounters(renderer: THREE.WebGPURenderer) {
    const words = new Uint32Array(await renderer.getArrayBufferAsync(this.countersAttribute));
    const state = Array.from(words.subarray(this.lodStateBase, this.lodStateBase + LOD_STATE_SIZE), (w) => w / STATE_FIXED);
    const scale = [state[S_SCALE], state[S_SCALE + 1]];
    const fill = [state[S_FILL], state[S_FILL + 1]];
    // Still steering: keep redoing the cut each frame until it settles (a still camera would otherwise skip it).
    this.controllerActive = fill.some((f, c) => f > FILL_HIGH || (f < FILL_LOW && scale[c] > 1.0001));
    const slots = this.meshSlotCapacity;
    const cut = (c: number) => ({
      selected: words[2 + c * CUT_COUNTERS + C_SELECTED],
      drawn: words[2 + c * CUT_COUNTERS + C_DRAWN],
      survived: words[2 + c * CUT_COUNTERS + C_SURVIVED],
      requestedTriangles: words[2 + c * CUT_COUNTERS + C_REQUESTED] / 3,
    });
    const perMesh = (c: number, slot: number) => (words[PER_MESH_COUNTERS + (c * slots + slot) * 4] + words[PER_MESH_COUNTERS + (c * slots + slot) * 4 + 2]) / 3;
    return { camera: cut(0), shadow: cut(1), occluded: words[1], perMesh, scale, fill };
  }

  /**
   * Byte offset of a mesh's draw args for a cut (0: camera, 1: shadow) and part (0: meshlets drawn over every pixel,
   * 1: meshlets inside an LOD blend band, drawn with the blend mask) in `drawArgsAttribute`.
   */
  drawArgsOffset(cut: number, slot: number, part = 0) {
    return ((cut * this.meshSlotCapacity + slot) * 2 + part) * 20;
  }

  dispose() {
    for (const node of this.computeNodes) node.dispose();
    this.computeNodes = [];
  }
}

/**
 * World-space bounding sphere of the instance whose column-major matrix starts at `base`, computed the same way
 * the instance pass does: center = model * sphere center, radius = sphere radius * max column length.
 */
export function instanceSphere(m: ArrayLike<number>, base: number, sphere: number[], out: Float64Array) {
  const [sx, sy, sz, sr] = sphere;
  out[0] = m[base] * sx + m[base + 4] * sy + m[base + 8] * sz + m[base + 12];
  out[1] = m[base + 1] * sx + m[base + 5] * sy + m[base + 9] * sz + m[base + 13];
  out[2] = m[base + 2] * sx + m[base + 6] * sy + m[base + 10] * sz + m[base + 14];
  const c0 = Math.hypot(m[base], m[base + 1], m[base + 2]);
  const c1 = Math.hypot(m[base + 4], m[base + 5], m[base + 6]);
  const c2 = Math.hypot(m[base + 8], m[base + 9], m[base + 10]);
  out[3] = sr * Math.max(c0, c1, c2);
}

/** Octahedral normal (two snorm16 in one u32) -> unit vector. */
function decodeOctNormal(packed: Node): Node {
  const bits: Node = bitcast(packed, 'int');
  const u = float(bits.shiftLeft(16).shiftRight(16)).div(32767);
  const v = float(bits.shiftRight(16)).div(32767);
  const z = float(1).sub(u.abs()).sub(v.abs());
  const t = maxNode(z.negate(), 0);
  const x = u.add(select(u.greaterThanEqual(0), t.negate(), t));
  const y = v.add(select(v.greaterThanEqual(0), t.negate(), t));
  return normalize(vec3(x, y, z));
}

function unpackRgba8(packed: Node): Node {
  return vec4(float(packed.bitAnd(255)), float(packed.shiftRight(8).bitAnd(255)), float(packed.shiftRight(16).bitAnd(255)), float(packed.shiftRight(24))).div(255);
}

export function encodeOctNormal(nx: number, ny: number, nz: number): number {
  const l1 = Math.abs(nx) + Math.abs(ny) + Math.abs(nz) || 1;
  let u = nx / l1;
  let v = ny / l1;
  if (nz < 0) {
    const pu = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);
    const pv = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
    u = pu;
    v = pv;
  }
  const su = Math.round(Math.max(-1, Math.min(1, u)) * 32767) & 0xffff;
  const sv = Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff;
  return (su | (sv << 16)) >>> 0;
}

/** xyz as float bits + octahedral-encoded normal, 4 x u32 per vertex. */
export function packVertices(positions: Float32Array, normals: Float32Array): Uint32Array {
  const count = positions.length / 3;
  const out = new Uint32Array(count * 4);
  const asFloat = new Float32Array(out.buffer);
  for (let i = 0; i < count; i++) {
    asFloat[i * 4 + 0] = positions[i * 3 + 0];
    asFloat[i * 4 + 1] = positions[i * 3 + 1];
    asFloat[i * 4 + 2] = positions[i * 3 + 2];
    out[i * 4 + 3] = encodeOctNormal(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
  }
  return out;
}

/**
 * uv per vertex as two unorm16 in one u32, quantized over the bounds of all UVs (also tiling UVs outside 0..1).
 * Decode: unpackUnorm2x16(packed) * scale + offset. One step is 1/65535 of the range: well below a texel of a 4K
 * texture for 0..1 UVs.
 */
export function packUvs(uvs: Float32Array): { packed: Uint32Array; scale: [number, number]; offset: [number, number] } {
  const count = uvs.length / 2;
  const lo = [Infinity, Infinity];
  const hi = [-Infinity, -Infinity];
  for (let i = 0; i < uvs.length; i++) {
    const k = i & 1;
    if (uvs[i] < lo[k]) lo[k] = uvs[i];
    if (uvs[i] > hi[k]) hi[k] = uvs[i];
  }
  if (count === 0) lo[0] = lo[1] = hi[0] = hi[1] = 0;
  const scale: [number, number] = [hi[0] - lo[0] || 1, hi[1] - lo[1] || 1];
  const packed = new Uint32Array(count);
  const q = (v: number, k: number) => Math.round(Math.max(0, Math.min(1, (v - lo[k]) / scale[k])) * 65535);
  for (let i = 0; i < count; i++) packed[i] = (q(uvs[i * 2], 0) | (q(uvs[i * 2 + 1], 1) << 16)) >>> 0;
  return { packed, scale, offset: [lo[0], lo[1]] };
}

/** Spread the low 10 bits of v so that two zero bits sit between consecutive ones. */
function spreadBits10(v: number): number {
  v &= 0x3ff;
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

/**
 * Instance indices sorted along a 30-bit Morton curve over the instance centers, so that consecutive slots are
 * spatially close and a fixed-size run of slots forms a compact cell.
 */
export function mortonOrder(matrices: Float32Array, sphere: number[]): Uint32Array {
  const n = matrices.length / 16;
  const s = new Float64Array(4);
  const centers = new Float64Array(n * 3);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    instanceSphere(matrices, i * 16, sphere, s);
    for (let a = 0; a < 3; a++) {
      centers[i * 3 + a] = s[a];
      lo[a] = Math.min(lo[a], s[a]);
      hi[a] = Math.max(hi[a], s[a]);
    }
  }
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
  const codes = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const q = (a: number) => Math.min(1023, Math.max(0, Math.floor(((centers[i * 3 + a] - lo[a]) / extent) * 1024)));
    codes[i] = (spreadBits10(q(0)) | (spreadBits10(q(1)) << 1) | (spreadBits10(q(2)) << 2)) >>> 0;
  }
  return radixOrder(codes);
}

/**
 * Indices sorted by 30-bit key: LSD radix sort, two passes of 15 bits. O(n), and stable, so equal keys keep index
 * order (same result as a stable comparison sort). Millions of instances sort in tens of milliseconds.
 */
function radixOrder(keys: Uint32Array): Uint32Array {
  const n = keys.length;
  let order = Uint32Array.from({ length: n }, (_, i) => i);
  let next = new Uint32Array(n);
  const counts = new Uint32Array(1 << 15);
  for (const shift of [0, 15]) {
    counts.fill(0);
    for (let i = 0; i < n; i++) counts[(keys[i] >>> shift) & 0x7fff]++;
    let sum = 0;
    for (let b = 0; b < counts.length; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const index = order[i];
      next[counts[(keys[index] >>> shift) & 0x7fff]++] = index;
    }
    [order, next] = [next, order];
  }
  return order;
}

/**
 * A renderable set of instances of one preprocessed mesh (a three.js Mesh, so it goes into any scene).
 *
 * The geometry and instances live in a GeometryPool shared with other meshes; the pool's compute passes select
 * the cut for all of its meshes at once (see GeometryPool). This object holds the material and two geometries,
 * the camera cut and the shadow cut, which draw this mesh's region of the pool's index buffers.
 */
import * as THREE from 'three/webgpu';
import { cameraViewMatrix, faceDirection, hash, materialColor, normalize, select, vec3, vec4 } from 'three/tsl';
import { VG_DEBUG_MODES } from '../constants.js';
import type { VirtualMeshData } from '../preprocess/buildVirtualMesh.js';
import type { VirtualGeometry } from './VirtualGeometry.js';
import {
  CELL_SIZE,
  NO_DRAW_DISTANCE,
  mortonOrder,
  vgInstanceOrigin,
  vgInstanceVarying,
  vgLodVarying,
  vgMeshletVarying,
  vgTintVarying,
  vgWorldNormal,
  type GeometryPool,
  type PoolEntry,
} from './GeometryPool.js';
import { bindVirtualGeometryTextures } from './vgMaterial.js';

export { vgWorldNormal, vgInstanceOrigin, encodeOctNormal, packVertices, packUvs } from './GeometryPool.js';

export interface VirtualMeshInstances {
  /** 16 floats (column-major Matrix4) per instance. */
  matrices: Float32Array;
  /** Optional rgba tint per instance (multiplied with the material color). */
  colors?: Float32Array;
}

export interface VirtualMeshOptions {
  /**
   * Cull distance in world units: instances farther than this are not drawn (and cost almost nothing, since
   * whole cells are rejected at once). They shrink to nothing over the last 15% of the distance, so there is
   * no visible pop. Default: Infinity. Can be changed later via `mesh.maxDrawDistance`.
   */
  maxDrawDistance?: number;
  /**
   * Hide instances whose bounding sphere is smaller than this many pixels (radius) on screen. Combined with
   * `VirtualGeometry.minPixelRadius` (the larger one wins). Default: 0. Also settable via `mesh.minPixelRadius`.
   */
  minPixelRadius?: number;
  /**
   * Moves vertices in the vertex shader, e.g. foliage swaying in the wind: gets the world-space position node and
   * returns a displaced one (three.js TSL). `instanceOrigin` is the instance's world position, so a tree can bend
   * from its base. Culling and level of detail use the undisplaced geometry, so keep the displacement small (well
   * under a metre).
   */
  deform?: (worldPosition: any, context: { instanceOrigin: any }) => any; // eslint-disable-line @typescript-eslint/no-explicit-any -- TSL nodes
  /** @deprecated Draw capacity is shared per pool now: see `VirtualGeometryOptions`. Ignored. */
  maxDrawnMeshlets?: number;
  /** @deprecated Draw capacity is shared per pool now: see `VirtualGeometryOptions`. Ignored. */
  maxDrawnTriangles?: number;
}

type Node = any; // TSL node typings are too strict for the graph code below; runtime is verified in the browser

/**
 * Untouched copy of every material a VirtualMesh has taken over. A mesh rewires its material's nodes, so two
 * meshes cannot share one material object: the second one gets a clone of the original instead, as if the caller
 * had passed separate materials.
 */
const pristineMaterials = new WeakMap<THREE.NodeMaterial, THREE.NodeMaterial>();

function claimMaterial(material: THREE.NodeMaterial): THREE.NodeMaterial {
  const pristine = pristineMaterials.get(material);
  if (pristine) return pristine.clone();
  pristineMaterials.set(material, material.clone());
  return material;
}

function placeholderGeometry() {
  const geometry = new THREE.BufferGeometry();
  // Placeholder attributes; real vertex data is pulled from storage buffers in the vertex shader.
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(3), 3));
  return geometry;
}

export class VirtualMesh extends THREE.Mesh<THREE.BufferGeometry, THREE.NodeMaterial> {
  readonly isVirtualMesh = true;
  readonly data: VirtualMeshData;
  readonly instanceCount: number;
  readonly fullDetailTriangles: number;
  /** The pool holding this mesh's geometry and instances, and its record there. Set by `VirtualGeometry`. */
  pool!: GeometryPool;
  entry!: PoolEntry;
  /** Geometries drawing this mesh's region of the pool's camera and shadow index buffers. */
  readonly cameraGeometry = placeholderGeometry();
  readonly shadowGeometry = placeholderGeometry();
  /** `context.frameCount` of the last shadow-map render of this mesh. */
  lastShadowFrame = 0;
  /**
   * Lever: draw this mesh into the occlusion depth pass, so it can hide other geometry. Default: solid
   * materials (opaque, single-sided, no alpha test). Foliage is mostly holes, so it only gets tested.
   */
  occluder: boolean;
  /** Lever: let occlusion culling remove hidden instances and meshlets of this mesh. */
  occlusionCulling = true;
  /** GPU instance slot of each caller instance (inverse of the Morton order). */
  readonly gpuIndexOf: Uint32Array;
  /** Morton order: caller instance of each GPU slot. */
  readonly gpuToInput: Uint32Array;
  private readonly dirtyCells = new Set<number>();
  private drawingShadow = false;
  private poolVersion = -1;
  /** `options.deform`, applied to the pool's position node. */
  private readonly deform: ((worldPosition: Node, context: { instanceOrigin: Node }) => Node) | undefined;
  private readonly context: VirtualGeometry;

  /** Cull distance in world units (Infinity: never). See `VirtualMeshOptions.maxDrawDistance`. */
  get maxDrawDistance() {
    const d = this.pool.readFlag(this.entry, 'maxDrawDistance');
    return d >= NO_DRAW_DISTANCE ? Infinity : d;
  }
  set maxDrawDistance(distance: number) {
    this.pool.writeFlags(this.entry, { maxDrawDistance: Number.isFinite(distance) ? Math.max(0, distance) : NO_DRAW_DISTANCE });
  }

  /** Per-mesh minimum on-screen radius in pixels. See `VirtualMeshOptions.minPixelRadius`. */
  get minPixelRadius() {
    return this.pool.readFlag(this.entry, 'minPixelRadius');
  }
  set minPixelRadius(pixels: number) {
    this.pool.writeFlags(this.entry, { minPixelRadius: Math.max(0, pixels) });
  }

  /** The pool's per-frame draw capacity (shared by every mesh of the pool). */
  get capacity() {
    return { meshlets: this.pool.capacity.meshlets, triangles: this.pool.capacity.triangles };
  }

  constructor(context: VirtualGeometry, data: VirtualMeshData, material: THREE.NodeMaterial, instances: VirtualMeshInstances, options: VirtualMeshOptions = {}) {
    material = claimMaterial(material);
    const geometry = placeholderGeometry();
    super(geometry, material);
    this.deform = options.deform;
    this.context = context;
    this.data = data;
    this.frustumCulled = false; // culling happens on the GPU
    this.instanceCount = instances.matrices.length / 16;
    this.fullDetailTriangles = data.stats.leafTriangles * this.instanceCount;
    this.occluder = material.side !== THREE.DoubleSide && !material.transparent && material.alphaTest === 0 && !material.alphaHash;

    // The pool stores the instances in Morton order, so each cell is a contiguous range of slots.
    this.gpuToInput = mortonOrder(instances.matrices, data.boundingSphere);
    this.gpuIndexOf = new Uint32Array(this.instanceCount);
    for (let g = 0; g < this.instanceCount; g++) this.gpuIndexOf[this.gpuToInput[g]] = g;
    context.register(this, instances);
    if (options.maxDrawDistance !== undefined) this.maxDrawDistance = options.maxDrawDistance;
    if (options.minPixelRadius !== undefined) this.minPixelRadius = options.minPixelRadius;

    // Material: textures sample the pulled UVs, per-instance tint, debug views. The position node comes from
    // the pool (set in syncPool, again whenever the pool rebuilds its buffers).
    bindVirtualGeometryTextures(material);
    const base = vec4(material.colorNode ?? (materialColor as Node)).toVar() as Node;
    const randomColor = (seed: Node) => vec3(hash(seed), hash(seed.add(17.17)), hash(seed.add(43.43)));
    const debugColor = select(
      context.debugMode.equal(VG_DEBUG_MODES.meshlets),
      randomColor(vgMeshletVarying.add(vgInstanceVarying.mul(0.618))),
      select(context.debugMode.equal(VG_DEBUG_MODES.lodLevel), randomColor(vgLodVarying.add(3)), randomColor(vgInstanceVarying))
    );
    // Alpha passes through, so alpha-tested and transparent materials (and their shadows) keep working.
    material.colorNode = vec4(select(context.debugMode.equal(VG_DEBUG_MODES.shaded), base.xyz.mul(vgTintVarying), debugColor), base.w);
    // Two-sided foliage: back faces must light with the outward normal (normalNode is in view space).
    if (material.side === THREE.DoubleSide && !material.normalNode) {
      material.normalNode = normalize((cameraViewMatrix as Node).mul(vec4(vgWorldNormal, 0)).xyz.mul(faceDirection as Node));
    }
    this.syncPool();
    this.geometry = this.cameraGeometry;
    geometry.dispose();

    // The shadow pass calls onBeforeShadow and then onBeforeRender for each caster; the camera pass only
    // onBeforeRender. Each pass keeps its own render object and picks up `geometry` at draw time, so swapping
    // it here makes the shadow pass draw the shadow cut and the camera pass the camera cut.
    this.onBeforeShadow = (_renderer: unknown, _object: unknown, _camera: unknown, shadowCamera: THREE.Camera) => {
      this.lastShadowFrame = context.frameCount;
      context.noteShadowCamera(shadowCamera);
      this.drawingShadow = true;
      this.geometry = this.shadowGeometry;
    };
    // context.register() installed the auto-update hook as onBeforeRender: keep calling it after the swap.
    const autoUpdateHook = this.onBeforeRender;
    this.onBeforeRender = (...args: Parameters<THREE.Object3D['onBeforeRender']>) => {
      if (this.drawingShadow) this.drawingShadow = false;
      else this.geometry = this.cameraGeometry;
      autoUpdateHook.apply(this, args);
    };
  }

  /** Points the geometries and the material at the pool's current buffers and nodes (after a pool rebuild). */
  syncPool() {
    const pool = this.pool;
    if (this.poolVersion === pool.version || !pool.positionNode) return;
    this.poolVersion = pool.version;
    this.cameraGeometry.setIndex(pool.cameraIndex as unknown as THREE.BufferAttribute);
    this.cameraGeometry.setIndirect(pool.drawArgsAttribute, pool.drawArgsOffset(0, this.entry.slot));
    this.shadowGeometry.setIndex(pool.shadowIndex as unknown as THREE.BufferAttribute);
    this.shadowGeometry.setIndirect(pool.drawArgsAttribute, pool.drawArgsOffset(1, this.entry.slot));
    this.material.positionNode = this.deform ? this.deform(pool.positionNode, { instanceOrigin: vgInstanceOrigin }) : pool.positionNode;
    this.material.needsUpdate = true;
  }

  /** Force the next `VirtualGeometry.update` to recompute the cut. */
  markCutDirty() {
    this.pool.dirty = true;
  }

  /** Update one instance transform. Call `commitInstances()` after a batch of changes. */
  setMatrixAt(index: number, matrix: THREE.Matrix4) {
    const slot = this.gpuIndexOf[index];
    this.pool.setMatrix(this.entry, slot, matrix);
    this.dirtyCells.add(Math.floor(slot / CELL_SIZE));
  }

  commitInstances() {
    this.pool.commitInstances(this.entry, this.dirtyCells);
    this.dirtyCells.clear();
  }

  /**
   * Stop rendering this mesh and drop it from its context. The GPU buffers are released when the
   * renderer frees them; call this before removing the object from the scene for good.
   */
  dispose(context?: VirtualGeometry) {
    this.visible = false;
    (context ?? this.context).unregister(this);
    this.removeFromParent();
  }

  /** Triangles drawn for this mesh by the last computed cut (camera and shadow), from one readback. */
  async readStats(renderer: THREE.WebGPURenderer) {
    const c = await this.pool.readCounters(renderer);
    return { drawnTriangles: c.perMesh(0, this.entry.slot), shadowTriangles: c.perMesh(1, this.entry.slot) };
  }
}

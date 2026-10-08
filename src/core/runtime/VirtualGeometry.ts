import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { VirtualMesh, VirtualMeshInstances, VirtualMeshOptions } from './VirtualMesh';
import { OcclusionCulling } from './OcclusionCulling';
import { VirtualMesh as VirtualMeshClass } from './VirtualMesh';
import type { VirtualMeshData } from '../preprocess/buildVirtualMesh';
import { GeometryPool, NO_DRAW_DISTANCE, type PoolCapacity } from './GeometryPool';
import { virtualMeshesFromObject3D, type VirtualGeometryImport, type VirtualGeometryImportOptions } from '../import/fromObject3D';

export { VG_DEBUG_MODES } from '../constants';

/** Length of the camera + settings vector compared by `VirtualGeometry.cutInputsChanged`. */
const CUT_INPUT_COUNT = 16 + 16 + 12;

export interface VirtualGeometryStats {
  drawnMeshlets: number;
  drawnTriangles: number;
  /** Triangles if every instance were drawn at full detail (no LOD, no culling). */
  fullDetailTriangles: number;
  instances: number;
  /** Instances that survived instance culling (frustum, sub-pixel). */
  visibleInstances: number;
  /** Triangles the LOD cut asked for. Above drawnTriangles means some mesh ran out of index buffer. */
  requestedTriangles: number;
  /** True when a mesh dropped meshlets for lack of capacity (visible as flicker). */
  overflow: boolean;
  /** Fullest draw buffer of any mesh, as a fraction of its capacity (above 1: overflow). */
  capacityUse: number;
  /** Instances hidden from the camera by occlusion culling. */
  occludedInstances: number;
  /** Triangles drawn into shadow maps (the coarser shadow cut). */
  shadowTriangles: number;
}

/**
 * Per-frame state shared by every `VirtualMesh`: camera uniforms used by the
 * GPU culling/LOD-selection kernels, plus global settings.
 */
const ZERO = new THREE.Vector3();

/**
 * Threshold that brings a too-full draw buffer back to ~85%: triangles scale with about 1/threshold^2, so
 * one step lands near the target even from a large overflow (at most 2x per step, at least 4%).
 */
function capacityStep(threshold: number, use: number) {
  return Math.min(64, threshold * Math.min(2, Math.max(1.04, Math.sqrt(use / 0.85))));
}

export interface VirtualGeometryOptions {
  /** Max triangles drawn per frame per pool (camera). Default 10M (120 MB of indices, under the 128 MB binding limit). */
  maxTriangles?: number;
  /** Max triangles drawn into shadow maps per frame per pool. Default 4M. */
  maxShadowTriangles?: number;
  /** Max meshlets drawn per frame per pool. Default 1M. */
  maxMeshlets?: number;
  /**
   * Largest storage-buffer binding a pool may use, in bytes. A pool that would exceed it is closed and a new one
   * opened. Default 120 MB (WebGPU guarantees 128 MB; raise it if you request a higher device limit).
   */
  maxPoolBytes?: number;
}

export class VirtualGeometry {
  readonly viewMatrix = uniform(new THREE.Matrix4());
  readonly projectionMatrix = uniform(new THREE.Matrix4());
  /** `projection[1][1] * viewportHeight / 2`: converts view-space error to pixels. */
  readonly projScale = uniform(1);
  readonly near = uniform(0.1);
  /** Max projected error, in pixels, a meshlet may have to be drawn. */
  readonly errorThreshold = uniform(1);

  /**
   * Lever: target number of drawn triangles per frame. When > 0, the error threshold is adjusted (slowly, see
   * adaptThreshold) to keep the drawn triangle count near this budget. Off by default: a fixed threshold gives
   * the steadiest image, because LODs then only switch with distance, never all at once.
   */
  triangleBudget = 0;
  /** Bounds for the automatically adjusted error threshold, in pixels. */
  thresholdRange: [number, number] = [0.5, 3];
  /** Latest GPU readback, updated every `statsInterval` frames while a budget or HUD is active. */
  lastStats: VirtualGeometryStats | null = null;
  statsInterval = 10;
  /** Frames updated so far. */
  frameCount = 0;
  /** The user's threshold while the capacity guard holds a coarser one (no budget), else null. */
  private guardBase: number | null = null;
  /** Consecutive stats readbacks with the drawn triangles outside the budget's deadband. */
  private outOfBand = 0;
  private guardWritten = NaN;
  private statsPending = false;
  readonly frustumCulling = uniform(1, 'uint');
  readonly debugMode = uniform(0, 'uint');
  /**
   * Instances whose bounding sphere projects to a smaller radius than this (pixels) are skipped. 0 disables.
   * The sphere is never smaller than the object, so at 0.7 nothing wider than ~1.4 px on screen is dropped.
   */
  readonly minPixelRadius = uniform(0.7);
  readonly frustumPlanes = Array.from({ length: 6 }, () => uniform(new THREE.Vector4()));
  /**
   * Direction * length that light travels (e.g. sun direction * 300). When non-zero, frustum culling
   * keeps meshlets whose shadow, swept along this vector, can reach the view frustum, so shadow
   * casters just outside the view still cast shadows. Zero disables it.
   */
  readonly shadowSweep = uniform(new THREE.Vector3());
  /**
   * Set `shadowSweep` automatically from the shadow cameras of the lights that render the meshes (direction
   * of an orthographic shadow camera, length = its larger extent). Turn off to set `shadowSweep` yourself.
   */
  autoShadowSweep = true;
  /**
   * Shadow maps use a coarser cut: the error threshold times this. Shadows are soft and seen from the camera
   * at a distance, so this saves most of the shadow-pass triangles with no visible change. 1 = camera detail.
   */
  readonly shadowErrorScale = uniform(3);
  private readonly detectedSweep = new THREE.Vector3();
  private detectedSweepFrame = -Infinity;

  /**
   * Occlusion culling (on by default): geometry hidden behind solid meshes is not drawn. Levers:
   * `occlusion.enabled`, `occlusion.resolutionScale`, and per mesh `occluder` / `occlusionCulling`.
   */
  readonly occlusion = new OcclusionCulling(this.projectionMatrix, this.near);

  /** Skip LOD selection/culling and keep drawing the last selection (for inspecting the cut). */
  freeze = false;
  /**
   * Run `update()` automatically before every render of a scene that contains this context's meshes, with the
   * camera that render uses (default true). Shadow-map renders reuse the cut of the camera being rendered.
   * Calling `update()` yourself still works: a render right after it with the same camera does not repeat it.
   */
  autoUpdate = true;
  /**
   * Measure the error threshold in CSS pixels (default) instead of device pixels. On a 2x display,
   * device pixels would request ~4x the triangles for detail nobody can see.
   */
  errorInCssPixels = true;

  readonly meshes: VirtualMesh[] = [];
  /** Shared buffers and compute passes; meshes go into the first pool with room (see GeometryPool). */
  readonly pools: GeometryPool[] = [];
  private readonly capacity: PoolCapacity;
  private readonly maxPoolBytes: number;

  constructor(options: VirtualGeometryOptions = {}) {
    this.capacity = {
      triangles: options.maxTriangles ?? 10_000_000,
      shadowTriangles: options.maxShadowTriangles ?? 4_000_000,
      meshlets: Math.min(options.maxMeshlets ?? 1 << 20, 1 << 24),
    };
    this.maxPoolBytes = options.maxPoolBytes ?? 120 * 1024 * 1024;
  }

  private readonly frustum = new THREE.Frustum();
  private readonly projScreen = new THREE.Matrix4();
  private readonly size = new THREE.Vector2();
  /** Compute nodes dispatched this frame (reused to avoid per-frame allocation). */
  private readonly computeList: THREE.ComputeNode[] = [];
  private readonly dispatchedPools: GeometryPool[] = [];
  /** Frames where `update` ran no compute pass because nothing the cut depends on changed. */
  skippedFrames = 0;
  /** Inputs of the cut as of the last frame that changed them (NaN until the first frame). */
  private readonly cutInputs = new Float64Array(CUT_INPUT_COUNT).fill(NaN);
  private readonly cutInputsScratch = new Float64Array(CUT_INPUT_COUNT);
  /** Scenes whose onBeforeRender runs the auto update (installed the first time one of our meshes renders in them). */
  private readonly sceneHooks = new WeakMap<THREE.Object3D, THREE.Object3D['onBeforeRender']>();
  /** `renderer.info.render.calls` and camera of the last automatic and the last manual update. */
  private autoCall = -1;
  private autoCamera: THREE.Camera | null = null;
  private manualCall = -1;
  private manualCamera: THREE.Camera | null = null;
  private updatingAuto = false;

  /**
   * Converts every static mesh under `object` (e.g. `gltf.scene`) to VirtualGeometry geometry, in place: meshes sharing
   * geometry and material become one VirtualMesh with many instances, materials are converted to node materials
   * with their textures, and the originals stop rendering. Skinned/morphing meshes, points and lines are left
   * alone. Returns the created meshes and a `dispose()` that restores the originals. See `VirtualGeometryImportOptions`.
   */
  add(object: THREE.Object3D, options?: VirtualGeometryImportOptions): Promise<VirtualGeometryImport> {
    return virtualMeshesFromObject3D(this, object, options);
  }

  /** Create a renderable mesh from a built DAG. Add the result to any three.js scene. */
  createMesh(data: VirtualMeshData, material: THREE.NodeMaterial, instances: VirtualMeshInstances, options?: VirtualMeshOptions): VirtualMesh {
    return new VirtualMeshClass(this, data, material, instances, options);
  }

  /**
   * Lever: compile a pool's compute pipelines in the background before its first dispatch (default), so adding
   * meshes never freezes the page; they appear once ready. False: compile on first use (a hitch).
   */
  asyncCompile = true;

  private compilePool(renderer: THREE.WebGPURenderer, pool: GeometryPool) {
    pool.pipelineState = 'compiling';
    const version = pool.version;
    const done = () => {
      if (pool.version !== version) return; // rebuilt meanwhile: compiles again
      pool.pipelineState = 'ready';
      pool.dirty = true;
    };
    renderer.compileComputeAsync(pool.computeNodes).then(done, done);
  }

  /** Builds pools whose buffers grew and points their meshes at the new buffers. */
  private preparePools(renderer: THREE.WebGPURenderer) {
    for (const pool of this.pools) pool.prepare(renderer);
    // A pool may also have been rebuilt while meshes were being added: syncPool is a version check.
    for (const mesh of this.meshes) mesh.syncPool();
  }

  /**
   * Optional, for loading screens: compiles every pipeline the scene needs (compute and render), so the first
   * frames neither freeze nor show meshes popping in. Call after adding the meshes, before the first render.
   */
  async compileAsync(renderer: THREE.WebGPURenderer, scene: THREE.Object3D, camera: THREE.Camera) {
    this.preparePools(renderer);
    const pending = this.pools.filter((p) => p.pipelineState !== 'ready');
    for (const pool of pending) pool.pipelineState = 'compiling';
    await Promise.all(pending.map((p) => renderer.compileComputeAsync(p.computeNodes).catch(() => undefined)));
    for (const pool of pending) {
      pool.pipelineState = 'ready';
      pool.dirty = true;
    }
    await renderer.compileAsync(scene, camera);
  }

  /** Called by meshes from onBeforeShadow: remembers the light direction for `autoShadowSweep`. */
  noteShadowCamera(shadowCamera: THREE.Camera) {
    if (this.detectedSweepFrame === this.frameCount) return;
    const ortho = shadowCamera as THREE.OrthographicCamera;
    if (!ortho.isOrthographicCamera) return; // spot/point lights: no single direction to sweep along
    const extent = Math.max(ortho.top - ortho.bottom, ortho.right - ortho.left) / ortho.zoom;
    shadowCamera.getWorldDirection(this.detectedSweep).multiplyScalar(extent);
    this.detectedSweepFrame = this.frameCount;
  }

  /** Puts the mesh's geometry and instances into the first pool with room (a new pool when none has). */
  register(mesh: VirtualMesh, instances: VirtualMeshInstances) {
    const instanceCount = instances.matrices.length / 16;
    if (GeometryPool.bytesOf(mesh.data) > this.maxPoolBytes) {
      throw new Error(`VirtualGeometry: a single mesh needs more than maxPoolBytes (${this.maxPoolBytes} bytes) of GPU buffer`);
    }
    let pool = this.pools.find((p) => p.fits(mesh.data, instanceCount));
    if (!pool) {
      pool = new GeometryPool(this, this.capacity, this.maxPoolBytes);
      this.pools.push(pool);
    }
    mesh.pool = pool;
    mesh.entry = pool.add(mesh.data, instances.matrices, instances.colors, mesh.gpuToInput);
    pool.ensureBuilt(); // nodes are only rebuilt when the pool's buffers grew
    this.meshes.push(mesh);
    mesh.onBeforeRender = (renderer, scene, camera) => this.autoUpdateFor(renderer as unknown as THREE.WebGPURenderer, scene, camera, false);
  }

  /**
   * Auto update, from a VirtualMesh's onBeforeRender. The first time, this also hooks the scene's onBeforeRender,
   * which runs before the render pass starts (so shadow maps rendered inside the pass already see the new cut);
   * later frames update from there. A compute submitted from inside a pass still runs before that pass on the
   * GPU, because the pass is submitted when it ends.
   */
  private autoUpdateFor(renderer: THREE.WebGPURenderer, scene: THREE.Object3D, camera: THREE.Camera, fromScene: boolean) {
    if (!this.autoUpdate || this.meshes.length === 0) return;
    // Shadow maps draw the cut of the camera being rendered (see shadowSweep), never their own.
    if ((scene as THREE.Scene).overrideMaterial && (scene as THREE.Scene & { overrideMaterial: { isShadowPassMaterial?: boolean } }).overrideMaterial.isShadowPassMaterial) return;
    if (!fromScene && (scene as THREE.Scene).isScene) {
      // The scene hook already ran for this render; (re)install it if missing, e.g. replaced by user code.
      if (scene.onBeforeRender === this.sceneHooks.get(scene)) return;
      const previous = scene.onBeforeRender;
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const context = this;
      const hook: THREE.Object3D['onBeforeRender'] = function (this: THREE.Object3D, r, s, c, ...rest) {
        context.autoUpdateFor(r as unknown as THREE.WebGPURenderer, s, c, true);
        previous.call(this, r, s, c, ...rest);
      };
      scene.onBeforeRender = hook;
      this.sceneHooks.set(scene, hook);
    }
    // Once per render call, and not for the render right after a manual update() with the same camera.
    const call = renderer.info.render.calls;
    if (camera === this.autoCamera && call === this.autoCall) return;
    if (camera === this.manualCamera && call === this.manualCall + 1) return;
    this.autoCall = call;
    this.autoCamera = camera;
    this.updatingAuto = true;
    try {
      this.update(renderer, camera as THREE.PerspectiveCamera);
    } finally {
      this.updatingAuto = false;
    }
  }

  unregister(mesh: VirtualMesh) {
    const i = this.meshes.indexOf(mesh);
    if (i === -1) return;
    this.meshes.splice(i, 1);
    mesh.pool.remove(mesh.entry);
    this.occlusion.unregisterMesh(mesh);
  }

  /** Dispose every mesh created by this context. */
  dispose() {
    for (const mesh of [...this.meshes]) mesh.dispose(this);
    for (const pool of this.pools) pool.dispose();
    this.pools.length = 0;
    this.occlusion.dispose();
  }

  /**
   * Runs LOD selection + culling on the GPU for `camera`. With `autoUpdate` (default) the renderer calls this
   * for you before each render; call it yourself before `renderer.render` when `autoUpdate` is false.
   * When the camera, the settings and every mesh's instances are unchanged, the compute passes are
   * skipped and the previous cut (index buffer + draw args) is drawn again.
   */
  update(renderer: THREE.WebGPURenderer, camera: THREE.PerspectiveCamera) {
    if (!this.updatingAuto) {
      this.manualCall = renderer.info.render.calls;
      this.manualCamera = camera;
    }
    if (this.freeze) return;
    this.occlusion.tick(performance.now());

    camera.updateMatrixWorld();
    this.viewMatrix.value.copy(camera.matrixWorldInverse);
    if (!this.projectionMatrix.value.equals(camera.projectionMatrix)) this.projectionMatrix.value.copy(camera.projectionMatrix);
    renderer.getDrawingBufferSize(this.size);
    // Errors are measured in CSS pixels, so a threshold means the same thing at any devicePixelRatio.
    const cssHeight = this.size.y / (this.errorInCssPixels ? renderer.getPixelRatio() : 1);
    // Only write uniforms that changed: a static camera then costs no uniform updates at all.
    const projScale = camera.projectionMatrix.elements[5] * cssHeight * 0.5;
    if (this.projScale.value !== projScale) this.projScale.value = projScale;
    if (this.near.value !== camera.near) this.near.value = camera.near;

    this.projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projScreen, THREE.WebGPUCoordinateSystem);
    for (let i = 0; i < 6; i++) {
      const { normal, constant } = this.frustum.planes[i];
      const p = this.frustumPlanes[i].value;
      if (p.x !== normal.x || p.y !== normal.y || p.z !== normal.z || p.w !== constant) {
        p.set(normal.x, normal.y, normal.z, constant);
      }
    }

    if (this.autoShadowSweep) {
      const sweep = this.frameCount - this.detectedSweepFrame <= 2 ? this.detectedSweep : ZERO;
      if (!this.shadowSweep.value.equals(sweep)) this.shadowSweep.value.copy(sweep);
    }
    // Per-mesh flags in the pools' mesh tables: visibility, shadows (only for meshes a shadow-casting light
    // rendered recently, reported from onBeforeShadow; a newly lit mesh starts casting one frame later) and
    // occlusion. writeFlags only uploads (and dirties the pool) when a value changes.
    const occlusionWanted = this.occlusion.wantsFrame(renderer, camera, this.frameCount);
    for (const mesh of this.meshes) {
      mesh.pool.writeFlags(mesh.entry, {
        enabled: mesh.visible ? 1 : 0,
        shadow: mesh.castShadow && this.frameCount - mesh.lastShadowFrame <= 2 ? 1 : 0,
        occlusion: occlusionWanted && mesh.occlusionCulling ? 1 : 0,
      });
    }

    // Anything the cut depends on changed: every pool needs a new cut.
    if (this.cutInputsChanged(camera, renderer.getPixelRatio())) {
      for (const pool of this.pools) pool.dirty = true;
    }

    this.preparePools(renderer);
    const nodes = this.computeList;
    nodes.length = 0;
    const dispatched = this.dispatchedPools;
    dispatched.length = 0;
    for (const pool of this.pools) {
      if (!pool.dirty || pool.entries.length === 0) continue;
      if (pool.pipelineState !== 'ready') {
        if (this.asyncCompile) {
          // Compile in the background: the pool's meshes show up a few frames later instead of a freeze.
          if (pool.pipelineState === 'none') this.compilePool(renderer, pool);
          continue;
        }
        pool.pipelineState = 'ready';
      }
      for (const node of pool.computeNodes) nodes.push(node);
      dispatched.push(pool);
    }
    if (nodes.length > 0) {
      // Depth of last frame's solid geometry from this frame's camera: what this frame's cut is tested against.
      const hzbNodes = occlusionWanted ? this.occlusion.beginFrame(renderer, camera, this.meshes) : [];
      if (hzbNodes.length) nodes.unshift(...hzbNodes);
      renderer.compute(nodes);
      for (const pool of dispatched) {
        pool.dirty = false;
        pool.hasCut = true;
      }
    } else {
      this.skippedFrames++;
    }

    this.frameCount++;
    // One small readback every statsInterval frames feeds the budget controller, the capacity guard (always
    // on: an overflow would flicker) and the occlusion auto pause.
    if (this.meshes.length > 0 && !this.statsPending && this.frameCount % this.statsInterval === 0) {
      this.statsPending = true;
      this.readStats(renderer)
        .then((s) => {
          this.lastStats = s;
          this.adaptThreshold(s);
        })
        .finally(() => (this.statsPending = false));
    }
  }


  /**
   * True when anything the GPU cut reads from the context differs from the last frame that was computed
   * (and records the new values). Exact comparison on purpose: an unmoved camera recomputes the same
   * matrices bit for bit, while any real change, including a budget-controller step, differs.
   */
  private cutInputsChanged(camera: THREE.PerspectiveCamera, pixelRatio: number): boolean {
    const cur = this.cutInputsScratch;
    let i = 0;
    for (let k = 0; k < 16; k++) cur[i++] = camera.matrixWorldInverse.elements[k];
    for (let k = 0; k < 16; k++) cur[i++] = camera.projectionMatrix.elements[k];
    cur[i++] = camera.near;
    cur[i++] = this.size.x;
    cur[i++] = this.size.y;
    cur[i++] = pixelRatio;
    cur[i++] = this.errorInCssPixels ? 1 : 0;
    cur[i++] = this.errorThreshold.value;
    cur[i++] = this.minPixelRadius.value;
    cur[i++] = this.frustumCulling.value;
    cur[i++] = this.debugMode.value;
    const sweep = this.shadowSweep.value;
    cur[i++] = sweep.x;
    cur[i++] = sweep.y;
    cur[i++] = sweep.z;

    for (let k = 0; k < CUT_INPUT_COUNT; k++) {
      if (cur[k] !== this.cutInputs[k]) {
        this.cutInputs.set(cur);
        return true;
      }
    }
    return false;
  }

  /**
   * Slow multiplicative controller. Drawn triangles grow roughly with 1/threshold^2. Every change of the
   * threshold re-picks LODs across the whole screen, so it moves rarely (deadband) and in small steps.
   *
   * Draw capacity is a hard constraint: an overflowing mesh drops meshlets in a different order every frame,
   * which flickers. So the controller coarsens *before* a buffer is full, never refines into one, and may go
   * above `thresholdRange[1]` only for capacity, coming back down as soon as there is room.
   */
  private adaptThreshold(stats: VirtualGeometryStats) {
    const t = this.errorThreshold.value;
    const use = stats.capacityUse;
    if (this.triangleBudget <= 0) {
      this.guardCapacity(stats, t);
      return;
    }
    this.guardBase = null;
    const [lo, hi] = this.thresholdRange;
    if (stats.overflow || use > 0.9) {
      this.errorThreshold.value = capacityStep(t, use);
      return;
    }
    if (stats.drawnTriangles <= 0) return;
    const ratio = stats.drawnTriangles / this.triangleBudget;
    // Every change re-picks LODs across the whole screen at once, which is far more visible than the
    // distance-driven switches of a fixed threshold. So: a wide deadband, and only after the count stayed out of
    // it for several readbacks in a row (moving through a scene makes it swing briefly all the time).
    if (Math.abs(ratio - 1) < 0.2) {
      this.outOfBand = 0;
      return;
    }
    if (++this.outOfBand < 3) return;
    let next = t * Math.min(1.03, Math.max(0.97, Math.sqrt(ratio)));
    // Back inside the range: gradually from above (capacity pushed it there), clamped otherwise.
    next = t > hi ? Math.max(hi, Math.min(next, t * 0.97)) : Math.min(hi, Math.max(lo, next));
    // Refining multiplies the drawn triangles by about (t / next)^2: never step into a full buffer.
    if (next < t && use * (t / next) ** 2 > 0.85) return;
    this.errorThreshold.value = next;
  }

  /**
   * Without a budget the threshold is the user's, except that draw capacity stays a hard limit: coarsen
   * temporarily before a buffer fills up, and return to the user's value once there is room again.
   */
  private guardCapacity(stats: VirtualGeometryStats, t: number) {
    // Changed since the guard last wrote it (by the user, or the budget was just switched off): new target.
    if (t !== this.guardWritten) this.guardBase = null;
    let next = t;
    if (stats.overflow || stats.capacityUse > 0.9) {
      this.guardBase ??= t;
      next = capacityStep(t, stats.capacityUse);
    } else if (this.guardBase !== null && t > this.guardBase) {
      const back = Math.max(this.guardBase, t * 0.97);
      if (stats.capacityUse * (t / back) ** 2 <= 0.85) next = back;
      if (next === this.guardBase) this.guardBase = null;
    }
    if (next !== t) this.errorThreshold.value = next;
    this.guardWritten = this.errorThreshold.value;
  }

  /** GPU -> CPU readback of the last culling result. Async; never call every frame. One small readback per pool. */
  async readStats(renderer: THREE.WebGPURenderer): Promise<VirtualGeometryStats> {
    const result: VirtualGeometryStats = {
      drawnMeshlets: 0,
      drawnTriangles: 0,
      fullDetailTriangles: 0,
      instances: 0,
      visibleInstances: 0,
      requestedTriangles: 0,
      overflow: false,
      capacityUse: 0,
      occludedInstances: 0,
      shadowTriangles: 0,
    };
    for (const mesh of this.meshes) {
      if (!mesh.visible) continue;
      result.fullDetailTriangles += mesh.fullDetailTriangles;
      result.instances += mesh.instanceCount;
    }
    for (const pool of this.pools) {
      if (!pool.hasCut) continue;
      const c = await pool.readCounters(renderer);
      const cap = pool.capacity;
      result.drawnMeshlets += c.camera.drawn;
      result.drawnTriangles += Math.min(c.camera.requestedTriangles, cap.triangles);
      result.requestedTriangles += c.camera.requestedTriangles;
      result.visibleInstances += c.camera.survived;
      result.occludedInstances += c.occluded;
      result.shadowTriangles += Math.min(c.shadow.requestedTriangles, cap.shadowTriangles);
      if (c.camera.selected > cap.meshlets || c.camera.requestedTriangles > cap.triangles) result.overflow = true;
      if (c.shadow.selected > cap.meshlets || c.shadow.requestedTriangles > cap.shadowTriangles) result.overflow = true;
      result.capacityUse = Math.max(
        result.capacityUse,
        c.camera.requestedTriangles / cap.triangles,
        c.camera.selected / cap.meshlets,
        c.shadow.requestedTriangles / cap.shadowTriangles,
        c.shadow.selected / cap.meshlets
      );
    }
    return result;
  }
}

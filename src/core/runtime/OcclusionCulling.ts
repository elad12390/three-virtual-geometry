/**
 * Occlusion culling against a hierarchical depth buffer (HZB), on by default.
 *
 * Per frame, before the cut is computed:
 *  1. occluder pass - the occluder meshes' previous-frame camera cuts (their index buffers still hold them)
 *     are drawn depth-only with the CURRENT camera into a small depth target. That geometry was visible last
 *     frame, so it is a subset of the real occluders.
 *  2. HZB build - mip 0 is that depth (copied into a storage buffer), each further mip is the MAX of its
 *     2x2 children: a texel only hides what lies behind all of its occluders.
 *  3. cull - an instance or meshlet sphere is dropped from the camera cut when its projected box, read at
 *     the mip where it covers at most 2x2 texels, is entirely behind the max depth there.
 *
 * Why it is safe: removing occluders can only push the depth buffer back, so anything hidden by the subset
 * is hidden by the full scene. Stale or missing occluders cost efficiency, never correctness. The occluders
 * are LOD-simplified (they may bulge up to the error threshold in front of the true surface), so the tested
 * sphere is grown by that error first.
 *
 * Only meshes flagged `occluder` are drawn in step 1 (by default: opaque, single-sided, no alpha test, i.e.
 * solid geometry such as terrain, rocks, buildings; foliage is mostly holes). Every mesh is tested, unless
 * its `occlusionCulling` is off. Shadow cuts are never occlusion culled: a caster hidden from the camera can
 * still shadow something visible.
 *
 * Storage: the whole HZB (a header with each mip's offset and size, then all mips) is one storage buffer,
 * sized for `maxSize`, so nothing is reallocated on resize.
 */
import * as THREE from 'three/webgpu';
import { Fn, If, Return, ceil, clamp, exp2, float, floor, instanceIndex, int, ivec2, log2, max, min, select, storage, texture, uint, uniform, vec3, vec4 } from 'three/tsl';
import type { VirtualMesh } from './VirtualMesh.js';

/** Upper bound on mip levels (maxSize up to 32768). */
const MAX_LEVELS = 16;
/** Header at the start of the HZB buffer: per level [offset, width, height, 0], then the level count. */
const HEADER_SIZE = MAX_LEVELS * 4 + 4;
const LEVEL_COUNT_SLOT = MAX_LEVELS * 4;
const WORKGROUP = 64;
const NO_NODES: THREE.ComputeNode[] = [];

/** Mean frame time: with pipelined or batched submission, single frame times vary; the mean is the throughput. */
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

const minNode = min as (a: unknown, b: unknown) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
const maxNode = max as (a: unknown, b: unknown) => any; // eslint-disable-line @typescript-eslint/no-explicit-any
type Node = any; // TSL node typings are too strict for the graph code below; runtime is verified in the browser

export class OcclusionCulling {
  /** Lever: set false to turn occlusion culling off (no occluder pass, no HZB, nothing culled by it). */
  enabled = true;
  /**
   * Lever: decide by measurement. Whether occlusion pays off depends on the scene and the GPU: the occluder
   * pass has a fixed cost, which wins in cities and interiors and loses in open landscapes. With `auto`,
   * frame times are compared with occlusion off and on (`autoProbeFrames` each), the faster setting is kept
   * for `autoHoldFrames`, then measured again. Within 3% (e.g. both capped by vsync) it stays off.
   * Set false to keep it on whenever `enabled`.
   */
  auto = true;
  autoProbeFrames = 40;
  autoHoldFrames = 900;
  /** The auto setting currently in use (true: occlusion on). */
  autoChoice = false;
  // Starts in 'hold' (off) and probes after a 120-frame warm-up: the first frames compile pipelines.
  private phase: 'probeOff' | 'probeOn' | 'hold' = 'hold';
  private phaseFrames = this.autoHoldFrames - 120;
  private samples: number[] = [];
  private offMean = 0;
  private lastTick = 0;
  /** Depth resolution relative to the drawing buffer. Lower is cheaper and culls a little less. */
  resolutionScale = 0.5;
  /** Largest HZB dimension in texels (memory ~1.33 * maxSize^2 * 4 bytes). */
  readonly maxSize: number;
  /** True when this frame's HZB holds at least one occluder (the cull kernels test against it only then). */
  active = false;
  /** 1 while this frame's HZB is valid; the cull kernels only occlusion-cull then. */
  readonly activeNode = uniform(0, 'uint');

  private readonly depthScene = new THREE.Scene();
  private readonly proxies = new Map<VirtualMesh, THREE.Mesh>();
  private readonly projection: Node;
  private readonly near: Node;
  private readonly levelCapacity: number;
  private readonly levelOffsets: number[] = [];
  private readonly hzbAttribute: THREE.StorageBufferAttribute;
  private readonly hzbWrite: Node;
  private readonly hzbRead: Node;
  /** Copy + reduce kernels for the current depth size, dispatched with this frame's cut (one submission). */
  private buildNodes: THREE.ComputeNode[] = [];
  private renderTarget: THREE.RenderTarget | null = null;
  private width = 0;
  private height = 0;
  private levels = 0;
  private readonly bufferSize = new THREE.Vector2();

  constructor(projection: Node, near: Node, maxSize = 1024) {
    this.maxSize = Math.max(1, Math.min(1 << (MAX_LEVELS - 1), maxSize));
    this.projection = projection;
    this.near = near;

    let dim = this.maxSize;
    let total = HEADER_SIZE;
    let levelCapacity = 0;
    for (;;) {
      this.levelOffsets.push(total);
      total += dim * dim;
      levelCapacity++;
      if (dim === 1) break;
      dim = Math.ceil(dim / 2);
    }
    this.levelCapacity = levelCapacity;
    this.hzbAttribute = new THREE.StorageBufferAttribute(new Float32Array(total), 1);
    this.hzbWrite = storage(this.hzbAttribute, 'float', total);
    // Separate node: toReadOnly() changes the node it is called on.
    this.hzbRead = storage(this.hzbAttribute, 'float', total).toReadOnly();

  }

  /**
   * True only when a sphere (view-space center, radius) is certainly hidden. Any doubt (touching the near
   * plane, outside the screen) keeps it.
   */
  occluded(viewCenter: Node, radius: Node): Node {
    // Closed-form bounds of the sphere's view-space box (perspective camera, looking down -z): over the box,
    // x / depth is extreme at its corners, so each axis needs only the near and far depth. Same result as
    // projecting the 8 corners, at a fraction of the cost.
    const p: Node = this.projection;
    const depth = viewCenter.z.negate();
    const nearDepth = depth.sub(radius).toVar();
    const farDepth = depth.add(radius);
    const xLo = viewCenter.x.sub(radius);
    const xHi = viewCenter.x.add(radius);
    const yLo = viewCenter.y.sub(radius);
    const yHi = viewCenter.y.add(radius);
    // ndc.x = P00 * x / depth - P02 (P02, P12 are zero unless the frustum is off-center); P00, P11 > 0.
    const sx = p.element(0).x;
    const sy = p.element(1).y;
    const minX = minNode(xLo.div(nearDepth), xLo.div(farDepth)).mul(sx).sub(p.element(2).x);
    const maxX = maxNode(xHi.div(nearDepth), xHi.div(farDepth)).mul(sx).sub(p.element(2).x);
    const minY = minNode(yLo.div(nearDepth), yLo.div(farDepth)).mul(sy).sub(p.element(2).y);
    const maxY = maxNode(yHi.div(nearDepth), yHi.div(farDepth)).mul(sy).sub(p.element(2).y);
    // Depth of the nearest point: ndc.z = -P22 + P23 / depth.
    const minZ = p.element(3).z.div(nearDepth).sub(p.element(2).z);
    const behind = nearDepth.lessThanEqual(this.near);

    // Footprint in mip-0 texels. Row 0 is the top of the screen (WebGPU texture origin).
    const hzb = this.hzbRead;
    const W = hzb.element(1);
    const H = hzb.element(2);
    const zero = float(0);
    const x0 = clamp(floor(minX.mul(0.5).add(0.5).mul(W)), zero, W.sub(1));
    const x1 = clamp(floor(maxX.mul(0.5).add(0.5).mul(W)), zero, W.sub(1));
    const y0 = clamp(floor(float(0.5).sub(maxY.mul(0.5)).mul(H)), zero, H.sub(1));
    const y1 = clamp(floor(float(0.5).sub(minY.mul(0.5)).mul(H)), zero, H.sub(1));

    // The mip whose texels are at least as big as the footprint: then it spans at most 2x2 of them.
    const extent = maxNode(x1.sub(x0), y1.sub(y0)).add(1);
    const levelF = minNode(ceil(log2(extent)), hzb.element(LEVEL_COUNT_SLOT).sub(1));
    const entry = uint(levelF).mul(4);
    const scale = exp2(levelF);
    const offset = hzb.element(entry);
    const levelW = hzb.element(entry.add(1));
    const levelH = hzb.element(entry.add(2));
    const sampleAt = (fx: Node, fy: Node): Node => {
      const tx = clamp(floor(fx.div(scale)), zero, levelW.sub(1));
      const ty = clamp(floor(fy.div(scale)), zero, levelH.sub(1));
      return hzb.element(uint(offset.add(ty.mul(levelW)).add(tx)));
    };
    const hzbMax = maxNode(maxNode(sampleAt(x0, y0), sampleAt(x1, y0)), maxNode(sampleAt(x0, y1), sampleAt(x1, y1)));

    // Hidden when the nearest point of the box is behind every occluder in its on-screen footprint (parts
    // off screen are not visible anyway).
    return behind.not().and(minZ.lessThan(1)).and(minZ.greaterThan(hzbMax.add(1e-6)));
  }

  /** Called once per frame by the context: feeds the auto A/B measurement (see `auto`). */
  tick(now: number) {
    const dt = this.lastTick ? now - this.lastTick : 0;
    this.lastTick = now;
    if (!this.auto || !this.enabled) return;
    this.phaseFrames++;
    // Skip the first frames after a switch (pipelines, stale cuts), and stalls (tab switches, loading).
    if (this.phase !== 'hold' && this.phaseFrames > 5 && dt > 0 && dt < 250) this.samples.push(dt);
    if (this.phase === 'probeOff' && this.samples.length >= this.autoProbeFrames) {
      this.offMean = mean(this.samples);
      this.next('probeOn');
    } else if (this.phase === 'probeOn' && this.samples.length >= this.autoProbeFrames) {
      this.autoChoice = mean(this.samples) < this.offMean * 0.97;
      this.next('hold');
    } else if (this.phase === 'hold' && this.phaseFrames >= this.autoHoldFrames) {
      this.next('probeOff');
    }
  }

  private next(phase: 'probeOff' | 'probeOn' | 'hold') {
    this.phase = phase;
    this.phaseFrames = 0;
    this.samples = [];
  }

  private autoWanted() {
    return this.phase === 'probeOn' || (this.phase === 'hold' && this.autoChoice);
  }

  /** Depth-only proxy of a mesh's camera cut, created on first use and kept in sync with its pool. */
  private proxyOf(mesh: VirtualMesh) {
    let proxy = this.proxies.get(mesh);
    if (!proxy) {
      // The meshlets drawn over every pixel only: the blended ones are left out, which only makes the occluders a
      // little less complete (never culls anything visible), and keeps this depth pass free of discards.
      proxy = new THREE.Mesh(mesh.cameraGeometry, new THREE.MeshBasicNodeMaterial({ colorWrite: false }));
      proxy.frustumCulled = false;
      this.depthScene.add(proxy);
      this.proxies.set(mesh, proxy);
    }
    const material = proxy.material as THREE.MeshBasicNodeMaterial;
    if (material.positionNode !== mesh.pool.positionNode) {
      material.positionNode = mesh.pool.positionNode; // same vertex pulling as the shaded mesh
      material.needsUpdate = true;
    }
    return proxy;
  }

  unregisterMesh(mesh: VirtualMesh) {
    const proxy = this.proxies.get(mesh);
    if (!proxy) return;
    this.proxies.delete(mesh);
    proxy.removeFromParent();
    (proxy.material as THREE.Material).dispose();
  }

  /**
   * Whether occlusion culling runs this frame (enabled, a supported camera and depth mode, and the auto tuner
   * wants it). Meshes get their per-mesh flag from this before the cut is computed.
   */
  wantsFrame(renderer: THREE.WebGPURenderer, camera: THREE.Camera, _frame: number) {
    const r = renderer as unknown as { reversedDepthBuffer?: boolean; logarithmicDepthBuffer?: boolean };
    // The test assumes a perspective camera and standard [0, 1] depth.
    const supported = (camera as THREE.PerspectiveCamera).isPerspectiveCamera && !r.reversedDepthBuffer && !r.logarithmicDepthBuffer;
    return this.enabled && supported && (!this.auto || this.autoWanted());
  }

  /**
   * Renders the occluders' last camera cuts into the depth target and returns the compute kernels that turn
   * it into the HZB; run them before the cut (the caller prepends them to the same compute submission).
   * Sets `activeNode`, which the cull kernels check: without occluders this frame, nothing is occlusion culled.
   */
  beginFrame(renderer: THREE.WebGPURenderer, camera: THREE.Camera, meshes: readonly VirtualMesh[]): THREE.ComputeNode[] {
    let any = false;
    for (const mesh of meshes) {
      const occluder = mesh.visible && mesh.occluder && mesh.pool.hasCut && !!mesh.pool.positionNode;
      if (!occluder && !this.proxies.has(mesh)) continue;
      const proxy = this.proxyOf(mesh);
      proxy.visible = occluder;
      any = any || occluder;
    }
    this.active = any;
    this.activeNode.value = any ? 1 : 0;
    if (!any) return NO_NODES;

    renderer.getDrawingBufferSize(this.bufferSize);
    const s = Math.min(this.resolutionScale, this.maxSize / Math.max(this.bufferSize.x, this.bufferSize.y));
    const w = Math.max(1, Math.round(this.bufferSize.x * s));
    const h = Math.max(1, Math.round(this.bufferSize.y * s));
    if (w !== this.width || h !== this.height) this.resize(w, h);

    // The depth scene has no lights, so this renders no shadow maps.
    const previous = renderer.getRenderTarget();
    // This runs inside the caller's render, which may be a pass with MRT outputs: depth only here.
    const previousMrt = renderer.getMRT();
    renderer.setMRT(null);
    renderer.setRenderTarget(this.renderTarget);
    renderer.render(this.depthScene, camera);
    renderer.setRenderTarget(previous);
    renderer.setMRT(previousMrt);
    return this.buildNodes;
  }

  private resize(w: number, h: number) {
    const old = this.buildNodes;
    this.renderTarget?.dispose();
    const depthTexture = new THREE.DepthTexture(w, h);
    depthTexture.type = THREE.FloatType;
    this.renderTarget = new THREE.RenderTarget(w, h, { depthBuffer: true, depthTexture });
    this.width = w;
    this.height = h;

    const header = this.hzbAttribute.array as Float32Array;
    header.fill(0, 0, HEADER_SIZE);
    let lw = w;
    let lh = h;
    let levels = 0;
    for (let i = 0; i < this.levelCapacity; i++) {
      header[i * 4] = this.levelOffsets[i];
      header[i * 4 + 1] = lw;
      header[i * 4 + 2] = lh;
      levels++;
      if (lw === 1 && lh === 1) break;
      lw = Math.max(1, Math.ceil(lw / 2));
      lh = Math.max(1, Math.ceil(lh / 2));
    }
    this.levels = levels;
    header[LEVEL_COUNT_SLOT] = levels;
    this.hzbAttribute.needsUpdate = true;

    // Mip 0: depth texture -> the first region of the HZB buffer.
    const depth = texture(depthTexture);
    const hzb = this.hzbWrite;
    const copyNode = Fn(() => {
      const idx = instanceIndex;
      const dstW = uint(hzb.element(1));
      const dstH = uint(hzb.element(2));
      If(idx.greaterThanEqual(dstW.mul(dstH)), () => {
        Return();
      });
      const x = idx.mod(dstW);
      const y = idx.div(dstW);
      // Through a var: an inline uint() of a storage element is dropped by the WGSL builder (f32 + u32).
      const dstOffset = uint(hzb.element(0)).toVar();
      hzb.element(dstOffset.add(idx)).assign(depth.load(ivec2(int(x), int(y))));
    })().compute(w * h, [WORKGROUP]);
    this.buildNodes = [copyNode];

    // Mip i = max of the 2x2 texels of mip i-1 (disjoint regions of the same read-write binding).
    lw = w;
    lh = h;
    for (let i = 1; i < levels; i++) {
      lw = Math.max(1, Math.ceil(lw / 2));
      lh = Math.max(1, Math.ceil(lh / 2));
      this.buildNodes.push(
        Fn(() => {
          const idx = instanceIndex;
          const dstOffset = uint(hzb.element(i * 4)).toVar();
          const dstW = uint(hzb.element(i * 4 + 1));
          const dstH = uint(hzb.element(i * 4 + 2));
          If(idx.greaterThanEqual(dstW.mul(dstH)), () => {
            Return();
          });
          const x = idx.mod(dstW);
          const y = idx.div(dstW);
          const srcOffset = uint(hzb.element((i - 1) * 4)).toVar();
          const srcW = uint(hzb.element((i - 1) * 4 + 1));
          const srcH = uint(hzb.element((i - 1) * 4 + 2));
          const x0 = x.mul(2);
          const y0 = y.mul(2);
          const x1 = minNode(x0.add(1), srcW.sub(1));
          const y1 = minNode(y0.add(1), srcH.sub(1));
          const row0 = srcOffset.add(y0.mul(srcW));
          const row1 = srcOffset.add(y1.mul(srcW));
          const v = maxNode(maxNode(hzb.element(row0.add(x0)), hzb.element(row0.add(x1))), maxNode(hzb.element(row1.add(x0)), hzb.element(row1.add(x1))));
          hzb.element(dstOffset.add(idx)).assign(v);
        })().compute(lw * lh, [WORKGROUP])
      );
    }
    for (const node of old) node.dispose();
  }

  dispose() {
    this.renderTarget?.dispose();
    this.renderTarget = null;
    for (const mesh of [...this.proxies.keys()]) this.unregisterMesh(mesh);
  }
}

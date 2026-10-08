/**
 * Large world: `?scene=world`. A 6 km x 6 km landscape (9x the area of `?scene=map`) with mountain
 * ranges, lakes, forests and meadows, and a few million instances (`?quality=low`: ~1/3 of them and a
 * coarser terrain). Everything goes through VirtualMesh.
 *
 * Generation is built to scale to millions of instances:
 *  - the height field and the forest/meadow masks are sampled once on grids, and all placement reads
 *    them bilinearly (no noise evaluation per candidate);
 *  - placement is a jittered grid with a per-type density, so the cost is linear in the area;
 *  - the terrain is split into TILES x TILES VirtualGeometry meshes built from the same grid. Their shared edges are
 *    open borders, which the DAG builder keeps locked at every level (tail simplification is disabled), so
 *    neighbouring tiles at different LODs never crack. Tiles also cull and select LOD independently.
 */
import * as THREE from 'three/webgpu';
import { color, mix, mx_noise_float, positionWorld, smoothstep as tslSmoothstep } from 'three/tsl';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { buildVirtualMeshCached, VirtualMesh, vgWorldNormal, type VirtualMeshBuildOptions, type VirtualMeshData, type VirtualMeshSource } from '../../src/index';
import type { DemoApp } from './app';
import { makeBroadleaf, makeBush, makeConifer, makeGrassClump, makeRock } from './assets';
import { Perlin, mulberry32, smoothstep } from './noise';

const WORLD_SIZE = 6144;
const HALF = WORLD_SIZE / 2;
const WATER_LEVEL = 8;
const TILES = 4;
/** VirtualMesh keeps one mat4 per instance in one storage buffer; 1M x 64 B stays under WebGPU's 128 MB default. */
const MAX_PER_MESH = 1_000_000;

/** Scalar field sampled on a (res + 1)^2 grid over the world, read back with bilinear filtering. */
class Grid {
  readonly values: Float32Array;
  readonly cell: number;
  constructor(readonly res: number, fn: (x: number, z: number) => number) {
    this.cell = WORLD_SIZE / res;
    this.values = new Float32Array((res + 1) * (res + 1));
    for (let j = 0; j <= res; j++) {
      const z = -HALF + j * this.cell;
      for (let i = 0; i <= res; i++) this.values[j * (res + 1) + i] = fn(-HALF + i * this.cell, z);
    }
  }
  at(i: number, j: number) {
    return this.values[j * (this.res + 1) + i];
  }
  sample(x: number, z: number) {
    const fx = Math.min(this.res - 1e-6, Math.max(0, (x + HALF) / this.cell));
    const fz = Math.min(this.res - 1e-6, Math.max(0, (z + HALF) / this.cell));
    const i = Math.floor(fx);
    const j = Math.floor(fz);
    const tx = fx - i;
    const tz = fz - j;
    const a = this.at(i, j) + (this.at(i + 1, j) - this.at(i, j)) * tx;
    const b = this.at(i, j + 1) + (this.at(i + 1, j + 1) - this.at(i, j + 1)) * tx;
    return a + (b - a) * tz;
  }
}

/** Normal y component of the height grid at (x, z), from central differences. */
function uprightness(h: Grid, x: number, z: number) {
  const e = h.cell;
  const dx = (h.sample(x + e, z) - h.sample(x - e, z)) / (2 * e);
  const dz = (h.sample(x, z + e) - h.sample(x, z - e)) / (2 * e);
  return 1 / Math.sqrt(1 + dx * dx + dz * dz);
}

/** One terrain tile straight from the height grid: indexed quads, normals from grid differences. */
function makeTerrainTile(h: Grid, ti: number, tj: number): VirtualMeshSource {
  const n = h.res / TILES;
  const i0 = ti * n;
  const j0 = tj * n;
  const verts = (n + 1) * (n + 1);
  const positions = new Float32Array(verts * 3);
  const normals = new Float32Array(verts * 3);
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const gi = i0 + i;
      const gj = j0 + j;
      const v = j * (n + 1) + i;
      positions[v * 3] = -HALF + gi * h.cell;
      positions[v * 3 + 1] = h.at(gi, gj);
      positions[v * 3 + 2] = -HALF + gj * h.cell;
      // Global grid differences, so vertices on a shared tile edge get identical normals in both tiles.
      const dx = h.at(Math.min(h.res, gi + 1), gj) - h.at(Math.max(0, gi - 1), gj);
      const dz = h.at(gi, Math.min(h.res, gj + 1)) - h.at(gi, Math.max(0, gj - 1));
      const nx = -dx;
      const ny = 2 * h.cell;
      const nz = -dz;
      const len = Math.hypot(nx, ny, nz);
      normals.set([nx / len, ny / len, nz / len], v * 3);
    }
  }
  const indices = new Uint32Array(n * n * 6);
  let k = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      const b = a + 1;
      const c = a + n + 1;
      const d = c + 1;
      indices.set([a, c, b, b, c, d], k);
      k += 6;
    }
  }
  return { positions, normals, indices };
}

interface AssetSpec {
  name: string;
  make: () => VirtualMeshSource;
  build?: VirtualMeshBuildOptions;
}

export async function createWorldScene(app: DemoApp) {
  const quality = new URLSearchParams(location.search).get('quality') === 'low' ? 'low' : 'high';
  const density = quality === 'low' ? 0.35 : 1;
  const { scene, camera, controls, vg } = app;
  const perlin = new Perlin(4242);
  const tick = () => new Promise((r) => setTimeout(r, 0));

  // ---------------------------------------------------------------- terrain shape
  const lakes = [
    { x: 300, z: 700, r: 520 },
    { x: -1500, z: 1300, r: 650 },
    { x: 1900, z: 1800, r: 480 },
    { x: -700, z: -350, r: 380 },
    { x: 2300, z: -300, r: 420 },
    { x: -2300, z: -900, r: 360 },
  ];
  const heightAt = (x: number, z: number) => {
    const hills = perlin.fbm2(x / 1100, z / 1100, 5) * 70;
    const detail = perlin.fbm2(x / 110, z / 110, 3) * 4;
    // Mountains along the north edge, plus ranges wherever a very low frequency mask rises.
    const range = Math.max(smoothstep(-700, -2300, z), smoothstep(0.12, 0.42, perlin.fbm2(x / 2600 + 7, z / 2600 - 3, 2)) * 0.85);
    const mountains = range > 0 ? perlin.ridged2(x / 720 + 10, z / 720 - 4, 6) * 400 * range : 0;
    let lake = 0;
    for (const l of lakes) lake = Math.min(lake, smoothstep(l.r, l.r * 0.3, Math.hypot(x - l.x, z - l.z)) * -75);
    return 32 + hills + detail + mountains + lake;
  };

  const segments = quality === 'low' ? 768 : 1536; // 8 m / 4 m between terrain vertices
  app.progress('Sampling terrain…', 0);
  await tick();
  const height = new Grid(segments, heightAt);
  // Masks change slowly: a coarse grid is plenty.
  const forest = new Grid(384, (x, z) => perlin.fbm2(x / 420 + 50, z / 420 - 20, 4));
  const meadow = new Grid(384, (x, z) => perlin.fbm2(x / 260 - 31, z / 260 + 17, 3));

  // ---------------------------------------------------------------- assets
  const terrainBuild: VirtualMeshBuildOptions = { tailSimplify: false };
  const specs: AssetSpec[] = [];
  for (let tj = 0; tj < TILES; tj++) {
    for (let ti = 0; ti < TILES; ti++) specs.push({ name: `terrain-${ti}-${tj}`, make: () => makeTerrainTile(height, ti, tj), build: terrainBuild });
  }
  specs.push(
    { name: 'boulder', make: () => makeRock(11, quality === 'low' ? 48 : 80) },
    { name: 'stone', make: () => makeRock(23, quality === 'low' ? 32 : 56) },
    { name: 'conifer', make: () => makeConifer(5) },
    { name: 'broadleaf', make: () => makeBroadleaf(9) },
    { name: 'bush', make: () => makeBush(3) },
    { name: 'grass', make: () => makeGrassClump(7, 56), build: { prune: true } },
    { name: 'grass2', make: () => makeGrassClump(29, 44), build: { prune: true } },
    { name: 'flowers', make: () => makeGrassClump(13, 36, 10), build: { prune: true } }
  );

  const assets: Record<string, VirtualMeshData> = {};
  let buildMs = 0;
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    const label = `Building ${spec.name} (${i + 1}/${specs.length})`;
    app.progress(`${label}…`, i / specs.length);
    await tick();
    assets[spec.name] = await buildVirtualMeshCached(spec.make(), {
      ...spec.build,
      onProgress: (f) => app.progress(`${label}: meshlet DAG…`, (i + f) / specs.length),
    });
    buildMs += assets[spec.name].stats.buildMs;
  }

  // ---------------------------------------------------------------- scatter
  const rand = mulberry32(99);
  type Density = (x: number, z: number, h: number, up: number, f: number, m: number) => number;
  /**
   * Jittered grid over the world with one candidate per `spacing` x `spacing` cell, kept with probability
   * `accept`. Returns (x, y, z) triples. Linear in the area, no per-candidate noise. Above `limit`, the
   * result is thinned evenly (not truncated, which would leave part of the map empty).
   */
  const scatter = (spacing: number, accept: Density, limit = MAX_PER_MESH) => {
    const cells = Math.floor(WORLD_SIZE / spacing);
    const out: number[] = [];
    for (let j = 0; j < cells; j++) {
      for (let i = 0; i < cells; i++) {
        const x = -HALF + (i + rand()) * spacing;
        const z = -HALF + (j + rand()) * spacing;
        const h = height.sample(x, z);
        const p = accept(x, z, h, h > WATER_LEVEL ? uprightness(height, x, z) : 0, forest.sample(x, z), meadow.sample(x, z));
        if (p > 0 && rand() < p) out.push(x, h, z);
      }
    }
    const count = out.length / 3;
    if (count <= limit) return new Float32Array(out);
    const kept = new Float32Array(limit * 3);
    for (let k = 0; k < limit; k++) {
      const i = Math.floor((k * count) / limit);
      kept.set(out.slice(i * 3, i * 3 + 3), k * 3);
    }
    return kept;
  };
  // Spacing grows with 1/sqrt(density) so `?quality=low` keeps the same look with fewer instances.
  const sp = (s: number) => s / Math.sqrt(density);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const p = new THREE.Vector3();
  const s = new THREE.Vector3();
  const toInstances = (points: Float32Array, opts: { scale: [number, number]; sink?: number; tilt?: number; tint?: [number, number] }) => {
    const count = points.length / 3;
    const matrices = new Float32Array(count * 16);
    const colors = new Float32Array(count * 4);
    const tilt = opts.tilt ?? 0;
    const [t0, t1] = opts.tint ?? [1, 1];
    for (let i = 0; i < count; i++) {
      const sc = opts.scale[0] + rand() * (opts.scale[1] - opts.scale[0]);
      e.set((rand() - 0.5) * tilt, rand() * Math.PI * 2, (rand() - 0.5) * tilt);
      q.setFromEuler(e);
      p.set(points[i * 3], points[i * 3 + 1] - (opts.sink ?? 0) * sc, points[i * 3 + 2]);
      m.compose(p, q, s.setScalar(sc));
      m.toArray(matrices, i * 16);
      const k = t0 + rand() * (t1 - t0);
      colors[i * 4] = k * (0.95 + rand() * 0.1);
      colors[i * 4 + 1] = k * (0.95 + rand() * 0.1);
      colors[i * 4 + 2] = k * (0.95 + rand() * 0.1);
      colors[i * 4 + 3] = 1;
    }
    return { matrices, colors };
  };

  const dry = WATER_LEVEL + 1.5;
  const layers: { name: string; spacing: number; accept: Density; opts: Parameters<typeof toInstances>[1] }[] = [
    {
      name: 'conifer',
      spacing: 4.5,
      accept: (_x, _z, h, up, f) => (h > 18 && h < 290 && up > 0.8 ? smoothstep(-0.05, 0.25, f) * (h > 50 ? 0.9 : 0.35) : 0),
      opts: { scale: [0.7, 1.35], sink: 0.3, tint: [0.8, 1.15] },
    },
    {
      name: 'broadleaf',
      spacing: 7,
      accept: (_x, _z, h, up, f) => (h > WATER_LEVEL + 3 && h < 90 && up > 0.86 ? smoothstep(-0.12, 0.18, f) * 0.8 : 0),
      opts: { scale: [0.75, 1.3], sink: 0.3, tint: [0.8, 1.2] },
    },
    {
      name: 'bush',
      spacing: 4.5,
      accept: (_x, _z, h, up, f) => (h > dry && h < 170 && up > 0.8 ? 0.06 + 0.5 * smoothstep(0.3, 0, Math.abs(f)) : 0),
      opts: { scale: [0.6, 1.6], sink: 0.15, tint: [0.8, 1.2] },
    },
    {
      name: 'boulder',
      spacing: 22,
      accept: (_x, _z, h, up) => (h > WATER_LEVEL - 2 ? (up < 0.85 ? 0.8 : h > 160 ? 0.45 : 0.05) : 0),
      opts: { scale: [2.5, 9], sink: 0.15, tilt: 0.3, tint: [0.85, 1.1] },
    },
    {
      name: 'stone',
      spacing: 6,
      accept: (_x, _z, h, up) => (h > WATER_LEVEL - 1 ? (up < 0.9 ? 0.55 : 0.08) : 0),
      opts: { scale: [0.4, 1.6], sink: 0.1, tilt: 0.5, tint: [0.85, 1.1] },
    },
    {
      name: 'grass',
      spacing: 3.2,
      accept: (_x, _z, h, up, f, md) => (h > dry && h < 200 && up > 0.88 && f < 0.15 ? smoothstep(-0.25, 0.15, md) * 0.85 : 0),
      opts: { scale: [0.8, 1.5], sink: 0.02, tilt: 0.15, tint: [0.75, 1.15] },
    },
    {
      name: 'grass2',
      spacing: 3.2,
      accept: (_x, _z, h, up, f, md) => (h > dry && h < 220 && up > 0.86 && f < 0.3 ? smoothstep(-0.3, 0.1, md) * 0.8 : 0),
      opts: { scale: [0.9, 1.7], sink: 0.02, tilt: 0.2, tint: [0.8, 1.25] },
    },
    {
      name: 'flowers',
      spacing: 3.5,
      accept: (_x, _z, h, up, f, md) => (h > dry && h < 120 && up > 0.92 && f < 0 ? smoothstep(0.05, 0.35, md) * 0.9 : 0),
      opts: { scale: [0.8, 1.3], sink: 0.02, tilt: 0.15, tint: [0.85, 1.1] },
    },
  ];

  // ---------------------------------------------------------------- materials
  const foliage = (roughness = 0.85) => new THREE.MeshStandardNodeMaterial({ roughness, side: THREE.DoubleSide });
  const terrainMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.95 });
  {
    const h = positionWorld.y;
    const up = vgWorldNormal.y;
    const large = mx_noise_float(positionWorld.mul(0.012)).mul(0.5).add(0.5);
    const small = mx_noise_float(positionWorld.mul(0.35)).mul(0.5).add(0.5);
    let c = mix(color(0x3d6527), color(0x6f8c37), large.mul(0.7).add(small.mul(0.3)));
    c = mix(c, color(0x6b5a40), tslSmoothstep(0.93, 0.82, up).mul(0.7));
    c = mix(c, color(0x787168), tslSmoothstep(0.8, 0.66, up));
    c = mix(color(0xc8b78a), c, tslSmoothstep(WATER_LEVEL + 0.5, WATER_LEVEL + 3.5, h));
    c = mix(c, color(0xf0f4f8), tslSmoothstep(230, 290, h).mul(tslSmoothstep(0.5, 0.72, up)));
    terrainMaterial.colorNode = c;
  }

  // ---------------------------------------------------------------- meshes
  const groups: Record<string, VirtualMesh[]> = { terrain: [] };
  // Small clutter stops at a draw distance (it fades out, then costs nothing). Trees and boulders: unlimited.
  const drawDistance: Record<string, number> = { grass: 450, grass2: 450, flowers: 300, stone: 1200, bush: 1500 };
  const byName: Record<string, VirtualMesh> = {};
  let instanceTotal = 0;
  const add = (name: string, group: string, material: THREE.NodeMaterial, inst: { matrices: Float32Array; colors?: Float32Array }) => {
    const mesh = new VirtualMesh(vg, assets[name], material, inst, { maxDrawDistance: drawDistance[name] });
    mesh.name = name;
    byName[name] = mesh;
    // small foliage skips shadows: every caster re-draws its meshlets in the shadow pass
    mesh.castShadow = !['bush', 'grass', 'grass2', 'flowers'].includes(name);
    mesh.receiveShadow = true;
    scene.add(mesh);
    (groups[group] ??= []).push(mesh);
    instanceTotal += inst.matrices.length / 16;
  };

  const identity = new THREE.Matrix4().toArray(new Float32Array(16));
  for (const name of Object.keys(assets).filter((n) => n.startsWith('terrain-'))) add(name, 'terrain', terrainMaterial, { matrices: identity });

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    app.progress(`Placing ${layer.name}…`, i / layers.length);
    await tick();
    const points = scatter(sp(layer.spacing), layer.accept);
    const material = layer.name === 'boulder' || layer.name === 'stone' ? new THREE.MeshStandardNodeMaterial({ roughness: 0.9 }) : foliage(layer.name.startsWith('grass') ? 0.95 : 0.85);
    add(layer.name, layer.name.startsWith('grass') ? 'grass' : layer.name, material, toInstances(points, layer.opts));
    console.log(layer.name, (points.length / 3).toLocaleString(), 'instances');
  }

  // ---------------------------------------------------------------- environment
  const sunDirection = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(60), THREE.MathUtils.degToRad(205));
  const sky = new SkyMesh();
  sky.scale.setScalar(40000);
  sky.turbidity.value = 4;
  sky.rayleigh.value = 1.2;
  sky.sunPosition.value.copy(sunDirection);
  scene.add(sky);
  scene.fog = new THREE.FogExp2(0xb4c8d8, 0.00022);

  scene.add(new THREE.HemisphereLight(0xcfe2ff, 0x4d4030, 2.4));
  const sun = new THREE.DirectionalLight(0xfff0dc, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  const shadowExtent = 220;
  Object.assign(sun.shadow.camera, { left: -shadowExtent, right: shadowExtent, top: shadowExtent, bottom: -shadowExtent, near: 1, far: 2400 });
  sun.shadow.bias = -0.0004;
  sun.shadow.intensity = 0.7;
  sun.shadow.normalBias = 0.6;
  scene.add(sun, sun.target);
  app.renderer.shadowMap.enabled = true;

  const water = new THREE.Mesh(
    new THREE.PlaneGeometry(WORLD_SIZE * 3, WORLD_SIZE * 3).rotateX(-Math.PI / 2),
    new THREE.MeshStandardNodeMaterial({ color: 0x2a5a70, roughness: 0.06, metalness: 0.2, transparent: true, opacity: 0.88 })
  );
  water.position.y = WATER_LEVEL;
  water.receiveShadow = true;
  scene.add(water);

  // ---------------------------------------------------------------- camera
  // Start on a meadow on the shore of the central lake, looking north across forests to the mountains.
  let start = new THREE.Vector2(150, 1350);
  for (let r = 0; r < 1200; r += 20) {
    const a = r * 0.37;
    const x = 150 + Math.cos(a) * r;
    const z = 1350 + Math.sin(a) * r;
    const h = height.sample(x, z);
    if (h > WATER_LEVEL + 6 && h < 70 && uprightness(height, x, z) > 0.96 && forest.sample(x, z) < 0) {
      start = new THREE.Vector2(x, z);
      break;
    }
  }
  camera.near = 0.2;
  camera.far = 14000;
  camera.updateProjectionMatrix();
  camera.position.set(start.x, height.sample(start.x, start.y) + 18, start.y);
  const lookZ = start.y - 600;
  controls.target.set(start.x + 120, Math.max(WATER_LEVEL, height.sample(start.x + 120, lookZ)) + 30, lookZ);
  controls.maxPolarAngle = Math.PI * 0.495;

  const focus = new THREE.Vector3();
  app.onFrame = () => {
    // keep the shadow frustum centered in front of the camera
    camera.getWorldDirection(focus).multiplyScalar(shadowExtent * 0.6).add(camera.position);
    sun.target.position.copy(focus);
    sun.position.copy(focus).addScaledVector(sunDirection, 1100);
  };

  // ---------------------------------------------------------------- UI
  const folder = app.gui.addFolder('World');
  const visibility = Object.fromEntries(Object.keys(groups).map((k) => [k, true]));
  for (const name of Object.keys(groups)) {
    folder.add(visibility, name).onChange((v: boolean) => groups[name].forEach((mesh) => (mesh.visible = v)));
  }
  const distances = folder.addFolder('Draw distance (m)');
  const distanceGroups: Record<string, string[]> = { grass: ['grass', 'grass2'], flowers: ['flowers'], stones: ['stone'], bushes: ['bush'] };
  const distanceSettings = { grass: 450, flowers: 300, stones: 1200, bushes: 1500 };
  for (const key of Object.keys(distanceGroups) as (keyof typeof distanceSettings)[]) {
    distances.add(distanceSettings, key, 50, 6000, 50).onChange((v: number) => distanceGroups[key].forEach((n) => (byName[n].maxDrawDistance = v)));
  }
  const env = { shadows: true };
  folder.add(env, 'shadows').onChange((v: boolean) => (sun.castShadow = v));

  const totalMeshlets = Object.values(assets).reduce((a, d) => a + d.meshletCount, 0);
  app.extraStats =
    `${instanceTotal.toLocaleString()} instances · quality ${quality}\n` +
    `assets: ${Object.keys(assets).length}, ${totalMeshlets} meshlets, DAG build ${(buildMs / 1000).toFixed(1)} s`;
}

/**
 * Showcase: `?scene=ruins`. An ancient sanctuary and the ruined city around it, built from a handful of very
 * dense procedural assets (see detailedAssets.ts) instanced thousands of times: billions of full-detail
 * triangles, where every stone block, flute, egg-and-dart moulding and paving stone is real geometry.
 *
 *  - centre: a peripteral Ionic temple on a stepped platform, an altar with carved relief panels, and a twisted
 *    bronze knot on a pedestal in a paved plaza, with four smaller temples, inside a ruined precinct wall;
 *  - a colonnaded avenue leading south, and a grid of ruined houses and cobbled streets all around;
 *  - hills with cliffs and boulders, olive trees, cypresses, bushes and grass.
 *
 * `?tour` (or the "fly-through" button) flies from a close-up of the carved bronze out to the whole site and back.
 */
import * as THREE from 'three/webgpu';
import { cameraViewMatrix, color, float, max as tslMax, min as tslMin, mix, mx_noise_float, normalize, normalView, positionView, positionWorld, smoothstep as tslSmoothstep, vec4 } from 'three/tsl';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { buildVirtualMeshCached, VirtualMesh, vgWorldNormal, type VirtualMeshBuildOptions, type VirtualMeshData, type VirtualMeshSource } from '../../src/index';
import type { DemoApp } from './app';
import { makeBroadleaf, makeBush, makeGrassClump } from './assets';
import { makeGroundTerrain, RUINS_ASSETS } from './detailedAssets';
import { Perlin, mulberry32, smoothstep } from './noise';

const TERRAIN_SIZE = 2400;
/** Half size of the flat, built-up part of the site. */
const CITY = 288;
const BLOCK = 36;
/** Main temple: column grid and the top of its platform. */
const TEMPLE = { x: 0, z: -38, nx: 6, nz: 13, spacing: 3.4, base: 1.2 };
/** Height of a full column (plinth to abacus, see makeColumn). */
const COLUMN_H = 8.745;
const PRECINCT = { x0: -50, x1: 50, z0: -66, z1: 82 };
const AVENUE = { z0: 82, z1: 262, colX: 9, spacing: 4.6 };
const HERO = { x: 0, z: 20 };
/**
 * Cached builds are looked up by asset name and this version instead of a hash of the generated mesh, so a
 * cached load skips generation entirely. Bump it whenever a generator, an asset parameter or the terrain changes.
 */
const ASSETS_VERSION = 1;

interface AssetSpec {
  name: string;
  group: string;
  make: () => VirtualMeshSource;
  build?: VirtualMeshBuildOptions;
}

type Key = { target: [number, number, number]; dist: number; yaw: number; pitch: number; time: number };

export async function createRuinsScene(app: DemoApp) {
  const params = new URLSearchParams(location.search);
  const { scene, camera, controls, vg } = app;
  const perlin = new Perlin(2024);
  const rand = mulberry32(77);
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const tStart = performance.now();

  // ---------------------------------------------------------------- layout helpers
  const inPrecinct = (x: number, z: number, m = 0) => x > PRECINCT.x0 - m && x < PRECINCT.x1 + m && z > PRECINCT.z0 - m && z < PRECINCT.z1 + m;
  const inAvenue = (x: number, z: number, m = 0) => Math.abs(x) < 14 + m && z > AVENUE.z0 - 2 && z < AVENUE.z1 + 8 + m;
  const onStreet = (x: number, z: number) => {
    if (Math.abs(x) > CITY + 3 || Math.abs(z) > CITY + 3) return false;
    const sx = Math.abs(((x + CITY + BLOCK / 2) % BLOCK) - BLOCK / 2);
    const sz = Math.abs(((z + CITY + BLOCK / 2) % BLOCK) - BLOCK / 2);
    return sx < 3 || sz < 3;
  };
  const isPaved = (x: number, z: number) =>
    (x > -48 && x < 48 && z > -15 && z < 80) ||
    (Math.abs(x - TEMPLE.x) < 10.7 && Math.abs(z - TEMPLE.z) < 22.6) ||
    (Math.abs(x) < 6 && z > AVENUE.z0 && z < AVENUE.z1) ||
    (onStreet(x, z) && !inPrecinct(x, z, 2) && !inAvenue(x, z));

  // ---------------------------------------------------------------- terrain shape
  /** Depth of a rectangle's interior: 0 at its border, growing inwards. */
  const inside = (x: number, z: number, x0: number, x1: number, z0: number, z1: number) => Math.min(x - x0, x1 - x, z - z0, z1 - z);
  /**
   * The ground dips a little under the paving, away from its edges: coarse LODs of a paving tile may sit a few
   * centimetres lower than its stones, and must not sink into the terrain from afar.
   */
  const pavingDip = (x: number, z: number) => {
    let d = smoothstep(0, 5, inside(x, z, -48, 48, -15, 80)) * 0.3;
    d = Math.max(d, smoothstep(0, 2.5, inside(x, z, -6, 6, AVENUE.z0, AVENUE.z1)) * 0.3);
    if (d === 0 && onStreet(x, z) && !inPrecinct(x, z, 2) && !inAvenue(x, z)) {
      const sx = Math.abs(((x + CITY + BLOCK / 2) % BLOCK) - BLOCK / 2);
      const sz = Math.abs(((z + CITY + BLOCK / 2) % BLOCK) - BLOCK / 2);
      d = smoothstep(1.8, 0, Math.min(sx, sz)) * 0.15;
    }
    return d;
  };
  const heightAt = (x: number, z: number) => {
    const edge = Math.max(Math.abs(x), Math.abs(z)) * 0.7 + Math.hypot(x, z) * 0.3;
    const rise = smoothstep(CITY + 40, CITY + 360, edge + 130 * perlin.fbm2(x / 420, z / 420, 3) + 25 * perlin.fbm2(x / 90, z / 90, 2));
    const flat = -0.08 + 0.03 * perlin.fbm2(x / 14, z / 14, 2);
    if (rise <= 0) return flat - pavingDip(x, z);
    const hills = (35 + 80 * (0.5 + 0.5 * perlin.fbm2(x / 520 + 3, z / 520 - 1, 4))) * rise;
    const ridges = perlin.ridged2(x / 300 + 5, z / 300, 5) * 90 * rise * rise;
    return flat + hills + ridges;
  };
  const slopeAt = (x: number, z: number) => {
    const e = 2;
    const dx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e);
    const dz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
    return 1 / Math.sqrt(1 + dx * dx + dz * dz); // normal.y
  };

  // ---------------------------------------------------------------- assets
  const groupOf: Record<string, string> = {
    knot: 'sculpture & reliefs',
    pedestal: 'sculpture & reliefs',
    relief: 'sculpture & reliefs',
    column: 'columns',
    'column-broken': 'columns',
    drum: 'columns',
    capital: 'columns',
    'wall-ashlar': 'walls & steps',
    'wall-ashlar-ruin': 'walls & steps',
    'wall-rubble': 'walls & steps',
    'wall-rubble-ruin': 'walls & steps',
    step: 'walls & steps',
    beam: 'walls & steps',
    flagstones: 'paving',
    cobbles: 'paving',
    boulder: 'rocks',
    cliff: 'rocks',
    cypress: 'vegetation',
  };
  // Library defaults: no build options needed, even for assets made of thousands of separate stones.
  const specs: AssetSpec[] = RUINS_ASSETS.map((a) => ({ name: a.name, group: groupOf[a.name], make: a.make }));
  specs.push(
    { name: 'terrain', group: 'terrain', make: () => makeGroundTerrain(TERRAIN_SIZE, 640, heightAt) },
    { name: 'olive', group: 'vegetation', make: () => makeBroadleaf(17) },
    { name: 'bush', group: 'vegetation', make: () => makeBush(5) },
    { name: 'grass', group: 'vegetation', make: () => makeGrassClump(11, 48), build: { prune: true } },
    { name: 'flowers', group: 'vegetation', make: () => makeGrassClump(13, 32, 8), build: { prune: true } }
  );

  const assets: Record<string, VirtualMeshData> = {};
  let genMs = 0;
  let buildMs = 0;
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    const label = `Building ${spec.name} (${i + 1}/${specs.length})`;
    app.progress(`${label}…`, i / specs.length);
    await tick();
    // Generated on first access, which only happens when the cache misses.
    let mesh: VirtualMeshSource | null = null;
    const get = () => {
      if (!mesh) {
        const t0 = performance.now();
        mesh = spec.make();
        genMs += performance.now() - t0;
      }
      return mesh;
    };
    const source: VirtualMeshSource = {
      get positions() {
        return get().positions;
      },
      get normals() {
        return get().normals;
      },
      get colors() {
        return get().colors;
      },
      get indices() {
        return get().indices;
      },
    };
    const t0 = performance.now();
    assets[spec.name] = await buildVirtualMeshCached(source, {
      ...spec.build,
      cache: { key: `ruins-v${ASSETS_VERSION}:${spec.name}:${JSON.stringify(spec.build ?? {})}` },
      onProgress: (f) => app.progress(`${label}: meshlet DAG…`, (i + f) / specs.length),
    });
    buildMs += performance.now() - t0;
  }
  buildMs -= genMs;

  // ---------------------------------------------------------------- placement
  app.progress('Placing the ruins…', 1);
  await tick();
  const placed: Record<string, { matrices: number[]; colors: number[] }> = {};
  const m4 = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  type PutOptions = { s?: number; sx?: number; sy?: number; sz?: number; pitch?: number; roll?: number; tint?: number; rgb?: [number, number, number] };
  const put = (name: string, x: number, y: number, z: number, yaw = 0, o: PutOptions = {}) => {
    euler.set(o.pitch ?? 0, yaw, o.roll ?? 0, 'YXZ');
    quat.setFromEuler(euler);
    const s = o.s ?? 1;
    scl.set((o.sx ?? 1) * s, (o.sy ?? 1) * s, (o.sz ?? 1) * s);
    m4.compose(pos.set(x, y, z), quat, scl);
    const p = (placed[name] ??= { matrices: [], colors: [] });
    for (const v of m4.elements) p.matrices.push(v);
    const k = o.tint ?? 0.93 + rand() * 0.14;
    const [r, g, b] = o.rgb ?? [1, 1, 1];
    p.colors.push(k * r, k * g * (0.98 + rand() * 0.04), k * b * (0.97 + rand() * 0.05), 1);
  };
  const pick = <T,>(items: T[]) => items[Math.floor(rand() * items.length)];

  /** A straight wall line from (x0, z0) to (x1, z1), in segments of ~8 m. */
  const wallLine = (x0: number, z0: number, x1: number, z1: number, base: number, types: string[], sink: [number, number], missing: number) => {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const n = Math.max(1, Math.round(len / 8));
    const sx = len / (n * 8);
    const yaw = Math.atan2(-(z1 - z0), x1 - x0);
    for (let k = 0; k < n; k++) {
      if (rand() < missing) continue;
      const t = (k + 0.5) / n;
      const flip = rand() < 0.5 ? Math.PI : 0;
      put(pick(types), lerp(x0, x1, t), base - (sink[0] + rand() * (sink[1] - sink[0])), lerp(z0, z1, t), yaw + flip, { sx });
    }
  };

  /** Lying drums and a capital where a column fell (direction `dir`). */
  const fallenColumn = (x: number, y: number, z: number, dir: number) => {
    let d = 1.5;
    const drums = 2 + Math.floor(rand() * 3);
    for (let k = 0; k < drums; k++) {
      const a = dir + (rand() - 0.5) * 0.5;
      put('drum', x + Math.sin(a) * d, y + 0.52, z + Math.cos(a) * d, a + Math.PI / 2 + (rand() - 0.5) * 0.6, { pitch: Math.PI / 2 });
      d += 2.2 + rand() * 1.2;
    }
    if (rand() < 0.8) {
      const a = dir + (rand() - 0.5) * 0.4;
      put('capital', x + Math.sin(a) * (d + 0.5), y + 0.28, z + Math.cos(a) * (d + 0.5), rand() * Math.PI * 2, { pitch: 0.5 + rand() * 0.9 });
    }
  };

  /** Beams over every run of 5 consecutive standing columns (columns given in order along one side). */
  const beams = (cols: { x: number; z: number; ok: boolean }[], base: number, spacing: number) => {
    let k = 0;
    while (k + 4 < cols.length) {
      if (cols.slice(k, k + 5).every((c) => c.ok)) {
        const a = cols[k];
        const b = cols[k + 4];
        const yaw = Math.atan2(-(b.z - a.z), b.x - a.x);
        put('beam', (a.x + b.x) / 2, base + COLUMN_H, (a.z + b.z) / 2, yaw, { sx: spacing / 3.4 });
        k += 4;
      } else k++;
    }
  };

  /** Peripteral temple: stepped platform, colonnade, cella walls, beams on standing runs. */
  const temple = (cx: number, cz: number, yaw: number, nx: number, nz: number, steps: number, intact: number, front = false) => {
    const s = TEMPLE.spacing;
    const base = steps * 0.4;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const at = (lx: number, lz: number): [number, number] => [cx + lx * cos + lz * sin, cz - lx * sin + lz * cos];
    const hx = ((nx - 1) / 2) * s + 1.2;
    const hz = ((nz - 1) / 2) * s + 1.2;
    // steps: each course 0.4 m high and 1.1 m deep, stepping back 0.5 m per level
    for (let k = 0; k < steps; k++) {
      const ex = hx + (steps - 1 - k) * 0.5;
      const ez = hz + (steps - 1 - k) * 0.5;
      for (const side of [-1, 1]) {
        const nX = Math.ceil((2 * ex) / 12);
        for (let i = 0; i < nX; i++) {
          const [x, z] = at(-ex + ((i + 0.5) * 2 * ex) / nX, side * (ez - 0.55));
          put('step', x, k * 0.4, z, yaw + (side < 0 ? Math.PI : 0), { sx: (2 * ex) / (nX * 12) });
        }
        const lenZ = 2 * (ez - 1.1);
        const nZ = Math.ceil(lenZ / 12);
        for (let i = 0; i < nZ; i++) {
          const [x, z] = at(side * (ex - 0.55), -lenZ / 2 + ((i + 0.5) * lenZ) / nZ);
          put('step', x, k * 0.4, z, yaw + side * (Math.PI / 2), { sx: lenZ / (nZ * 12) });
        }
      }
    }
    // platform paving inside the top course
    const px = hx - 1.1;
    const pz = hz - 1.1;
    const tx = Math.ceil((2 * px) / 6);
    const tz = Math.ceil((2 * pz) / 6);
    for (let i = 0; i < tx; i++) {
      for (let j = 0; j < tz; j++) {
        const [x, z] = at(-px + ((i + 0.5) * 2 * px) / tx, -pz + ((j + 0.5) * 2 * pz) / tz);
        put('flagstones', x, base - 0.045, z, yaw + (rand() < 0.5 ? Math.PI : 0), { sx: (2 * px) / (tx * 6), sz: (2 * pz) / (tz * 6) });
      }
    }
    // colonnade, walked around the perimeter so each side is an ordered run
    const sides: { x: number; z: number; ok: boolean }[][] = [[], [], [], []];
    const state = new Map<string, boolean>();
    const columnAt = (i: number, j: number) => {
      const key = `${i},${j}`;
      if (!state.has(key)) {
        const lx = (i - (nx - 1) / 2) * s;
        const lz = (j - (nz - 1) / 2) * s;
        const [x, z] = at(lx, lz);
        const r = rand();
        const keep = front && j === nz - 1 && i > 0 ? intact + 0.3 : intact;
        let ok = false;
        if (r < keep) {
          put('column', x, base - 0.02, z, yaw + Math.floor(rand() * 4) * (Math.PI / 2) + (rand() - 0.5) * 0.04);
          ok = true;
        } else if (r < keep + (1 - keep) * 0.6) put('column-broken', x, base - 0.02, z, yaw + rand() * Math.PI * 2);
        else fallenColumn(x, base, z, Math.atan2(lx, lz) + yaw + (rand() - 0.5));
        state.set(key, ok);
      }
      const lx = (i - (nx - 1) / 2) * s;
      const lz = (j - (nz - 1) / 2) * s;
      const [x, z] = at(lx, lz);
      return { x, z, ok: state.get(key)! };
    };
    for (let i = 0; i < nx; i++) sides[0].push(columnAt(i, nz - 1));
    for (let j = nz - 1; j >= 0; j--) sides[1].push(columnAt(nx - 1, j));
    for (let i = nx - 1; i >= 0; i--) sides[2].push(columnAt(i, 0));
    for (let j = 0; j < nz; j++) sides[3].push(columnAt(0, j));
    for (const side of sides) beams(side, base, s);
    // cella
    const cw = ((nx - 1) / 2) * s - 3;
    const cz0 = -((nz - 1) / 2) * s + 3.4;
    const cz1 = ((nz - 1) / 2) * s - 4.4;
    const wall = (x0: number, z0: number, x1: number, z1: number) => {
      const [ax, az] = at(x0, z0);
      const [bx, bz] = at(x1, z1);
      wallLine(ax, az, bx, bz, base, ['wall-ashlar', 'wall-ashlar-ruin', 'wall-ashlar-ruin'], [0.1, 2.6], 0.12);
    };
    wall(-cw, cz0, -cw, cz1);
    wall(cw, cz1, cw, cz0);
    wall(cw, cz0, -cw, cz0);
    wall(-cw, cz1, -1.6, cz1);
    wall(1.6, cz1, cw, cz1);
  };

  temple(TEMPLE.x, TEMPLE.z, 0, TEMPLE.nx, TEMPLE.nz, 3, 0.62, true);
  for (const [x, z, yaw] of [
    [-33, 10, Math.PI / 2],
    [33, 10, -Math.PI / 2],
    [-33, 54, Math.PI / 2],
    [33, 54, -Math.PI / 2],
  ]) {
    temple(x, z, yaw, 4, 6, 2, 0.45);
  }
  const smallTemple = (x: number, z: number) => [-33, 33].some((tx) => Math.abs(x - tx) < 11.5) && [10, 54].some((tz) => Math.abs(z - tz) < 7.5);

  // altar with carved relief panels, and the bronze knot on its pedestal
  const altarZ = -5;
  for (let k = 0; k < 3; k++) {
    for (const side of [-1, 1]) put('step', 0, k * 0.4, altarZ + side * 0.55, side < 0 ? Math.PI : 0, { sx: 0.52 });
  }
  for (let k = -1; k <= 1; k++) {
    put('relief', k * 2.02, 0.12, altarZ + 1.13, 0);
    put('relief', -k * 2.02, 0.12, altarZ - 1.13, Math.PI);
  }
  put('relief', 3.15, 0.12, altarZ, Math.PI / 2, { sx: 0.95 });
  put('relief', -3.15, 0.12, altarZ, -Math.PI / 2, { sx: 0.95 });
  put('pedestal', HERO.x, 0, HERO.z, 0.3);
  put('knot', HERO.x, 1.55, HERO.z, 0.35, { tint: 1 });
  // empty pedestals around the plaza (their statues are long gone)
  for (const [x, z] of [
    [-14, 34],
    [14, 34],
    [-14, 64],
    [14, 64],
    [-20, 2],
    [20, 2],
  ]) {
    put('pedestal', x, 0, z, rand() * 6, { s: 0.7 });
  }

  // plaza paving
  for (let x = -45; x < 48; x += 6) {
    for (let z = -12; z < 80; z += 6) {
      if (smallTemple(x, z) || Math.abs(x) < 3.5 && Math.abs(z - altarZ) < 2) continue;
      put('flagstones', x, -0.045, z, Math.floor(rand() * 4) * (Math.PI / 2));
    }
  }
  // precinct wall, with the gate to the avenue
  const P = PRECINCT;
  const precinctTypes = ['wall-ashlar-ruin', 'wall-ashlar-ruin', 'wall-ashlar'];
  wallLine(P.x0, P.z0, P.x1, P.z0, 0, precinctTypes, [0.4, 3.8], 0.15);
  wallLine(P.x1, P.z0, P.x1, P.z1, 0, precinctTypes, [0.4, 3.8], 0.15);
  wallLine(P.x0, P.z1, P.x0, P.z0, 0, precinctTypes, [0.4, 3.8], 0.15);
  wallLine(P.x1, P.z1, 8, P.z1, 0, precinctTypes, [0.4, 3.8], 0.1);
  wallLine(-8, P.z1, P.x0, P.z1, 0, precinctTypes, [0.4, 3.8], 0.1);

  // colonnaded avenue
  const avenueCols: { x: number; z: number; ok: boolean }[][] = [[], []];
  for (let z = AVENUE.z0 + 4; z < AVENUE.z1; z += AVENUE.spacing) {
    for (const [i, x] of [-AVENUE.colX, AVENUE.colX].entries()) {
      const r = rand();
      let ok = false;
      if (r < 0.5) {
        put('column', x, -0.02, z, Math.floor(rand() * 4) * (Math.PI / 2));
        ok = true;
      } else if (r < 0.8) put('column-broken', x, -0.02, z, rand() * 6);
      else fallenColumn(x, 0, z, (x < 0 ? -1 : 1) * (Math.PI / 2) + (rand() - 0.5) * 0.8);
      avenueCols[i].push({ x, z, ok });
    }
  }
  for (const side of avenueCols) beams(side, 0, AVENUE.spacing);
  for (let z = AVENUE.z0 + 3; z < AVENUE.z1; z += 6) for (const x of [-3, 3]) put('cobbles', x, -0.045, z, Math.floor(rand() * 4) * (Math.PI / 2));

  // the city: blocks of ruined houses between cobbled streets
  for (let bx = -CITY; bx < CITY; bx += BLOCK) {
    for (let bz = -CITY; bz < CITY; bz += BLOCK) {
      const x0 = bx + 3;
      const x1 = bx + BLOCK - 3;
      const z0 = bz + 3;
      const z1 = bz + BLOCK - 3;
      const cx = (x0 + x1) / 2;
      const cz = (z0 + z1) / 2;
      if (inPrecinct(cx, cz, 22) || inAvenue(cx, cz, 18)) continue;
      const kind = rand();
      if (kind < 0.08) {
        // an olive grove where the houses are gone
        for (let k = 0; k < 14; k++) put('olive', lerp(x0 + 2, x1 - 2, rand()), -0.1, lerp(z0 + 2, z1 - 2, rand()), rand() * 6, { s: 0.5 + rand() * 0.35, rgb: [0.95, 0.95, 0.75] });
        continue;
      }
      const civic = kind > 0.88;
      const types = civic ? ['wall-ashlar-ruin', 'wall-ashlar-ruin', 'wall-ashlar'] : ['wall-rubble', 'wall-rubble', 'wall-rubble-ruin'];
      const sink: [number, number] = civic ? [0.5, 3.5] : [0.05, 2.3];
      const missing = kind < 0.2 ? 0.55 : 0.2;
      wallLine(x0, z0, x1, z0, 0, types, sink, missing);
      wallLine(x1, z0, x1, z1, 0, types, sink, missing);
      wallLine(x1, z1, x0, z1, 0, types, sink, missing);
      wallLine(x0, z1, x0, z0, 0, types, sink, missing);
      if (civic) {
        // a hall: a row of columns down the middle instead of party walls
        for (let k = 0; k < 6; k++) {
          const z = lerp(z0 + 3, z1 - 3, k / 5);
          if (rand() < 0.45) put('column', cx, -0.02, z, Math.floor(rand() * 4) * (Math.PI / 2));
          else if (rand() < 0.6) put('column-broken', cx, -0.02, z, rand() * 6);
          else fallenColumn(cx, 0, z, rand() * 6);
        }
        continue;
      }
      wallLine(cx, z0, cx, z1, 0, types, sink, missing + 0.1);
      wallLine(x0, cz, x1, cz, 0, types, sink, missing + 0.1);
      if (rand() < 0.25) put('boulder', lerp(x0, x1, rand()), -0.2, lerp(z0, z1, rand()), rand() * 6, { s: 0.5 + rand() * 0.8 });
    }
  }
  // streets
  for (let a = -CITY; a <= CITY; a += BLOCK) {
    for (let b = -CITY - 3 + 3; b <= CITY; b += 6) {
      for (const [x, z] of [
        [b, a],
        [a, b],
      ]) {
        if (x === a && z === b && (b + CITY) % BLOCK === 0) continue; // crossing, laid by the other street
        if (inPrecinct(x, z, 3) || inAvenue(x, z)) continue;
        put('cobbles', x, -0.045, z, Math.floor(rand() * 4) * (Math.PI / 2));
      }
    }
  }

  // hills: cliffs and boulders
  const onHills = (count: number, minR: number, accept: (x: number, z: number, h: number, up: number) => boolean) => {
    const out: [number, number, number][] = [];
    for (let t = 0; t < count * 30 && out.length < count; t++) {
      const x = (rand() - 0.5) * TERRAIN_SIZE * 0.9;
      const z = (rand() - 0.5) * TERRAIN_SIZE * 0.9;
      if (Math.max(Math.abs(x), Math.abs(z)) < minR) continue;
      const h = heightAt(x, z);
      if (accept(x, z, h, slopeAt(x, z))) out.push([x, h, z]);
    }
    return out;
  };
  // Rocks sit as deep as the terrain drops under them, so none floats on a slope.
  const lowest = (x: number, z: number, r: number) => Math.min(heightAt(x, z), heightAt(x + r, z), heightAt(x - r, z), heightAt(x, z + r), heightAt(x, z - r));
  for (const [x, , z] of onHills(150, CITY + 60, (_x, _z, h, up) => h > 15 && up < 0.92 && up > 0.6)) {
    const sc = 5 + rand() * 11;
    put('cliff', x, lowest(x, z, sc * 1.4) - sc * 0.15, z, rand() * 6, { s: sc, roll: (rand() - 0.5) * 0.2, pitch: (rand() - 0.5) * 0.2 });
  }
  for (const [x, , z] of onHills(500, CITY + 20, (_x, _z, h, up) => h > 1 && (up < 0.95 || rand() < 0.3))) {
    const sc = 1.2 + rand() * 5;
    put('boulder', x, lowest(x, z, sc * 1.1) - sc * 0.1, z, rand() * 6, { s: sc, pitch: (rand() - 0.5) * 0.3 });
  }

  // vegetation
  const vegetation = (name: string, spacing: number, region: number, accept: (x: number, z: number, h: number, up: number) => number, scale: [number, number], o: PutOptions = {}) => {
    for (let x = -region; x < region; x += spacing) {
      for (let z = -region; z < region; z += spacing) {
        const px = x + rand() * spacing;
        const pz = z + rand() * spacing;
        const h = heightAt(px, pz);
        const p = accept(px, pz, h, h > 0.5 ? slopeAt(px, pz) : 1);
        if (p <= 0 || rand() > p) continue;
        put(name, px, h - 0.05, pz, rand() * 6, { ...o, s: scale[0] + rand() * (scale[1] - scale[0]) });
      }
    }
  };
  const meadow = (x: number, z: number) => perlin.fbm2(x / 60 + 9, z / 60 - 4, 3);
  const open = (x: number, z: number) => !isPaved(x, z) && !(Math.abs(x - TEMPLE.x) < 12 && Math.abs(z - TEMPLE.z) < 24);
  const dry: [number, number, number] = [1.45, 1.1, 0.42];
  vegetation('grass', 1.6, CITY + 40, (x, z) => (open(x, z) ? smoothstep(-0.3, 0.2, meadow(x, z)) * 0.9 : 0), [0.8, 1.6], { rgb: dry });
  vegetation('grass', 3.5, 700, (x, z, h, up) => (Math.max(Math.abs(x), Math.abs(z)) > CITY + 40 && up > 0.85 && h < 140 ? 0.7 * smoothstep(-0.4, 0.2, meadow(x, z)) : 0), [1, 1.8], { rgb: dry });
  vegetation('flowers', 3, CITY, (x, z) => (open(x, z) ? smoothstep(0.15, 0.45, meadow(x, z)) * 0.8 : 0), [0.8, 1.2]);
  vegetation('bush', 7, CITY + 300, (x, z, _h, up) => (open(x, z) && up > 0.8 ? 0.12 + 0.3 * smoothstep(0, 0.4, meadow(x, z)) : 0), [0.5, 1.4]);
  const silver: [number, number, number] = [0.95, 0.95, 0.75];
  vegetation('olive', 16, CITY, (x, z) => (open(x, z) && !onStreet(x, z) && !inPrecinct(x, z, 4) ? 0.22 : 0), [0.45, 0.75], { rgb: silver });
  vegetation('olive', 14, 900, (x, z, h, up) => (Math.max(Math.abs(x), Math.abs(z)) > CITY + 30 && up > 0.85 && h < 160 ? 0.35 * smoothstep(-0.2, 0.3, -meadow(x, z)) : 0), [0.6, 1.0], { rgb: silver });
  vegetation('cypress', 18, 900, (x, z, h, up) => (Math.max(Math.abs(x), Math.abs(z)) > CITY + 30 && up > 0.85 && h < 180 ? 0.25 * smoothstep(0, 0.4, -meadow(x, z)) : 0), [0.7, 1.2]);
  for (let z = AVENUE.z0 + 6; z < AVENUE.z1; z += 9) for (const x of [-15, 15]) put('cypress', x + (rand() - 0.5), -0.2, z, rand() * 6, { s: 0.85 + rand() * 0.3 });

  // ---------------------------------------------------------------- materials
  const stone = new THREE.MeshStandardNodeMaterial({ roughness: 0.88 });
  const bronze = new THREE.MeshStandardNodeMaterial({ roughness: 0.38, metalness: 0.75 });
  const foliage = (roughness = 0.85) => new THREE.MeshStandardNodeMaterial({ roughness, side: THREE.DoubleSide });
  const terrainMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.96 });
  {
    const h = positionWorld.y;
    const up = vgWorldNormal.y;
    const large = mx_noise_float(positionWorld.mul(0.015)).mul(0.5).add(0.5);
    const small = mx_noise_float(positionWorld.mul(0.4)).mul(0.5).add(0.5);
    let c = mix(color(0x8a7457), color(0xa48b67), large.mul(0.6).add(small.mul(0.4)));
    c = mix(c, color(0x7c7a4a), tslSmoothstep(0.3, 0.75, large).mul(tslSmoothstep(2, 12, h)).mul(0.8));
    c = mix(c, color(0x8d8576), tslSmoothstep(0.85, 0.65, up));
    // Under the paving the ground takes the paving's distant colour: from afar the coarse LODs of terrain and
    // paving tiles cross each other, and matching colours (and flat normals, see the paving material) hide it.
    const px = positionWorld.x;
    const pz = positionWorld.z;
    const rect = (x0: number, x1: number, z0: number, z1: number) =>
      tslSmoothstep(-0.5, 0.5, tslMin(tslMin(px.sub(x0), float(x1).sub(px)), tslMin(pz.sub(z0), float(z1).sub(pz))));
    const lane = (v: typeof px) => v.add(CITY + BLOCK / 2).mod(BLOCK).sub(BLOCK / 2).abs();
    const streets = tslSmoothstep(3.5, 2.5, tslMin(lane(px), lane(pz)))
      .mul(rect(-CITY - 3, CITY + 3, -CITY - 3, CITY + 3))
      .mul(float(1).sub(rect(PRECINCT.x0 - 3, PRECINCT.x1 + 3, PRECINCT.z0 - 3, PRECINCT.z1 + 3)))
      .mul(float(1).sub(rect(-14, 14, AVENUE.z0 - 2, AVENUE.z1 + 8)));
    const plaza = tslMax(rect(-48, 48, -15, 80), rect(-6, 6, AVENUE.z0, AVENUE.z1));
    const paved = tslMax(plaza, streets).mul(tslSmoothstep(1, 0.2, h));
    terrainMaterial.colorNode = mix(c, color(0xb7a27f), paved);
    terrainMaterial.normalNode = normalize(mix(normalView, cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz, paved));
  }
  // Paving seen from afar: coarse LODs span many stones but keep the steep normals of their bevels, which
  // shades as large blotches. Fade towards the ground normal with distance (what a normal map's mips would do).
  const paving = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 });
  const up = cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz;
  paving.normalNode = normalize(mix(normalView, up, tslSmoothstep(15, 110, positionView.z.negate())));
  const materials: Record<string, THREE.NodeMaterial> = {
    flagstones: paving,
    cobbles: paving,
    knot: bronze,
    terrain: terrainMaterial,
    olive: foliage(),
    cypress: foliage(),
    bush: foliage(),
    grass: foliage(0.95),
    flowers: foliage(0.9),
  };

  // ---------------------------------------------------------------- meshes
  /** Per-mesh draw capacity: enough for a close-up filling the screen, without reserving GPU memory for the worst case. */
  const capacity: Record<string, number> = {
    knot: 1_500_000,
    terrain: 1_000_000,
    flagstones: 1_200_000,
    cobbles: 1_200_000,
    boulder: 1_000_000,
    cliff: 1_000_000,
    column: 1_500_000,
    olive: 1_000_000,
    grass: 800_000,
  };
  const drawDistance: Record<string, number> = { grass: 320, flowers: 220, bush: 900 };
  const groups: Record<string, VirtualMesh[]> = {};
  let instances = 0;
  let fullDetail = 0;
  const identity = new THREE.Matrix4().toArray(new Float32Array(16));
  for (const spec of specs) {
    const data = assets[spec.name];
    const p = placed[spec.name];
    const matrices = spec.name === 'terrain' ? identity : p ? new Float32Array(p.matrices) : null;
    if (!matrices) continue;
    const mesh = new VirtualMesh(vg, data, materials[spec.name] ?? stone, { matrices, colors: p ? new Float32Array(p.colors) : undefined }, {
      maxDrawDistance: drawDistance[spec.name],
    });
    mesh.name = spec.name;
    mesh.castShadow = !['grass', 'flowers', 'flagstones', 'cobbles', 'terrain', 'bush'].includes(spec.name);
    mesh.receiveShadow = true;
    scene.add(mesh);
    (groups[spec.group] ??= []).push(mesh);
    instances += mesh.instanceCount;
    fullDetail += mesh.fullDetailTriangles;
  }

  // ---------------------------------------------------------------- environment
  const sunDirection = new THREE.Vector3();
  const makeSky = (scale: number) => {
    const sky = new SkyMesh();
    sky.scale.setScalar(scale);
    sky.turbidity.value = 5;
    sky.rayleigh.value = 1.4;
    sky.mieCoefficient.value = 0.005;
    sky.mieDirectionalG.value = 0.85;
    return sky;
  };
  const sky = makeSky(20000);
  scene.add(sky);
  scene.fog = new THREE.FogExp2(0xd6c3a2, 0.0006);
  // Image-based light from the same sky over warm ground: soft fill, and something for the bronze to reflect.
  const envScene = new THREE.Scene();
  const envSky = makeSky(400);
  envScene.add(envSky);
  const envGround = new THREE.Mesh(new THREE.CircleGeometry(300, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicNodeMaterial({ color: 0x4a3c2c }));
  envGround.position.y = -2;
  envScene.add(envGround);
  const pmrem = new THREE.PMREMGenerator(app.renderer);
  let envTarget: THREE.RenderTarget | null = null;
  const hemi = new THREE.HemisphereLight(0xd8d4cc, 0x7a6244, 0.55);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffd6a8, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0003;
  sun.shadow.intensity = 0.85;
  scene.add(sun, sun.target);
  app.renderer.shadowMap.enabled = true;
  const env = { shadows: true, sunElevation: 17, sunAzimuth: 250 };
  const setSun = () => {
    sunDirection.setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - env.sunElevation), THREE.MathUtils.degToRad(env.sunAzimuth));
    sky.sunPosition.value.copy(sunDirection);
    envSky.sunPosition.value.copy(sunDirection);
    envTarget = pmrem.fromScene(envScene, 0, 0.5, 1000, envTarget ? { renderTarget: envTarget } : {});
    scene.environment = envTarget.texture;
    scene.environmentIntensity = 0.14;
    const low = smoothstep(40, 4, env.sunElevation);
    sun.color.setRGB(1, 0.93 - 0.15 * low, 0.84 - 0.3 * low);
    sun.intensity = 3.2 * smoothstep(-2, 8, env.sunElevation);
  };
  setSun();

  // ---------------------------------------------------------------- camera and fly-through
  const ground = (x: number, z: number) => Math.max(0, heightAt(x, z));
  const knotTop = 1.55;
  const keys: Key[] = [
    // close-up of the bronze strands, temple behind
    { target: [HERO.x + 1.38, knotTop + 1.85, HERO.z - 0.45], dist: 1.33, yaw: 0.17, pitch: 0.11, time: 0 },
    { target: [HERO.x, knotTop + 1.9, HERO.z], dist: 7, yaw: 0.35, pitch: 0.08, time: 9 },
    // the altar reliefs
    { target: [1.5, 0.8, altarZ + 1.1], dist: 3, yaw: 0.85, pitch: 0.3, time: 17 },
    // up the temple front
    { target: [0, 6, TEMPLE.z + 20], dist: 26, yaw: 0.6, pitch: 0.2, time: 26 },
    // the whole sanctuary
    { target: [0, 0, 0], dist: 160, yaw: 1.2, pitch: 0.42, time: 37 },
    // the whole city and the hills
    { target: [0, 0, 0], dist: 620, yaw: 2.4, pitch: 0.55, time: 50 },
    // down the colonnaded avenue
    { target: [0, 3, 170], dist: 70, yaw: 3.4, pitch: 0.25, time: 64 },
    // between the colonnades, looking north to the sanctuary
    { target: [AVENUE.colX, 5, 140], dist: 9, yaw: 5.55, pitch: 0.12, time: 74 },
    // back through the plaza to the bronze
    { target: [HERO.x, knotTop + 1.9, HERO.z], dist: 9, yaw: Math.PI * 2 + 0.2, pitch: 0.1, time: 86 },
    { target: [HERO.x + 1.38, knotTop + 1.85, HERO.z - 0.45], dist: 1.33, yaw: Math.PI * 2 + 0.17, pitch: 0.11, time: 94 },
  ];
  const loopTime = keys[keys.length - 1].time;
  // Monotone cubic (Fritsch-Carlson) through the keys, per channel and in time: velocity stays continuous and
  // nothing overshoots, even between a 1 m close-up and a 600 m panorama. Distance is interpolated in log
  // space, so the pull-out reads as one steady zoom.
  const channels = keys.map((k) => [k.target[0], k.target[1], k.target[2], Math.log(k.dist), k.yaw, k.pitch]);
  const slopes = keys.map((_, i) =>
    channels[i].map((_, c) => {
      if (i === 0 || i === keys.length - 1) return 0;
      const d0 = (channels[i][c] - channels[i - 1][c]) / (keys[i].time - keys[i - 1].time);
      const d1 = (channels[i + 1][c] - channels[i][c]) / (keys[i + 1].time - keys[i].time);
      if (d0 * d1 <= 0) return 0;
      return Math.sign(d0) * Math.min(Math.abs(d0 + d1) / 2, 3 * Math.min(Math.abs(d0), Math.abs(d1)));
    })
  );
  const camTarget = new THREE.Vector3();
  const pose = new Array<number>(6);
  const poseAt = (time: number, target: THREE.Vector3, position: THREE.Vector3) => {
    const t = ((time % loopTime) + loopTime) % loopTime;
    let i = 0;
    while (i < keys.length - 2 && t >= keys[i + 1].time) i++;
    const h = keys[i + 1].time - keys[i].time;
    const u = (t - keys[i].time) / h;
    const u2 = u * u;
    const u3 = u2 * u;
    for (let c = 0; c < 6; c++) {
      pose[c] =
        (2 * u3 - 3 * u2 + 1) * channels[i][c] + (u3 - 2 * u2 + u) * h * slopes[i][c] + (-2 * u3 + 3 * u2) * channels[i + 1][c] + (u3 - u2) * h * slopes[i + 1][c];
    }
    const [tx, ty, tz, logDist, yaw, pitch] = pose;
    const dist = Math.exp(logDist);
    target.set(tx, ty, tz);
    position.set(tx + dist * Math.cos(pitch) * Math.sin(yaw), ty + dist * Math.sin(pitch), tz + dist * Math.cos(pitch) * Math.cos(yaw));
    position.y = Math.max(position.y, ground(position.x, position.z) + 0.4);
    return dist;
  };

  camera.far = 9000;
  poseAt(0, camTarget, camera.position);
  controls.target.copy(camTarget);
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 0.15;
  controls.maxDistance = 2500;
  camera.updateProjectionMatrix();

  const tour = { active: params.has('tour'), time: 0 };
  const stopTour = () => (tour.active = false);
  app.renderer.domElement.addEventListener('pointerdown', stopTour);
  app.renderer.domElement.addEventListener('wheel', stopTour, { passive: true });

  const focus = new THREE.Vector3();
  const toTarget = new THREE.Vector3();
  app.onFrame = (dt) => {
    if (tour.active) {
      tour.time += dt;
      poseAt(tour.time, controls.target, camera.position);
    }
    // Near plane follows the distance to what we look at: 2 cm for close-ups, metres for the panorama.
    const dist = camera.position.distanceTo(controls.target);
    const height = camera.position.y - ground(camera.position.x, camera.position.z);
    const near = THREE.MathUtils.clamp(Math.min(dist, height + 1) * 0.02, 0.02, 3);
    if (Math.abs(near - camera.near) > camera.near * 0.15) {
      camera.near = near;
      camera.updateProjectionMatrix();
    }
    // Shadow map follows the camera and covers more ground as it pulls back.
    const extent = THREE.MathUtils.clamp(dist * 1.6, 14, 320);
    const sc = sun.shadow.camera;
    if (Math.abs(sc.right - extent) > extent * 0.1) {
      Object.assign(sc, { left: -extent, right: extent, top: extent, bottom: -extent, near: 1, far: 3000 });
      sc.updateProjectionMatrix();
      sun.shadow.normalBias = extent * 0.0012;
    }
    camera.getWorldDirection(toTarget);
    focus.copy(camera.position).addScaledVector(toTarget, Math.min(dist, extent * 0.7));
    sun.target.position.copy(focus);
    sun.position.copy(focus).addScaledVector(sunDirection, 1200);
  };

  // ---------------------------------------------------------------- UI
  const folder = app.gui.addFolder('Ruins');
  const visibility = Object.fromEntries(Object.keys(groups).map((k) => [k, true]));
  for (const name of Object.keys(groups)) folder.add(visibility, name).onChange((v: boolean) => groups[name].forEach((mesh) => (mesh.visible = v)));
  folder.add(env, 'shadows').onChange((v: boolean) => (sun.castShadow = v));
  folder.add(env, 'sunElevation', 2, 70, 0.5).name('sun elevation').onChange(setSun);
  folder.add(env, 'sunAzimuth', 0, 360, 1).name('sun azimuth').onChange(setSun);
  folder.add({ 'fly-through': () => ((tour.active = !tour.active), tour.active && (tour.time = 0)) }, 'fly-through');

  const fmtB = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(0)}M`);
  const sourceTriangles = Object.values(assets).reduce((a, d) => a + d.stats.leafTriangles, 0);
  const loadSeconds = (performance.now() - tStart) / 1000;
  app.extraStats =
    `${instances.toLocaleString()} instances of ${Object.keys(assets).length} assets (${(sourceTriangles / 1e6).toFixed(1)}M source triangles)\n` +
    `${fmtB(fullDetail)} full-detail triangles in the scene\n` +
    `load ${loadSeconds.toFixed(1)} s: ` +
    (genMs > 0 ? `generate ${(genMs / 1000).toFixed(1)} s, DAG build ${(buildMs / 1000).toFixed(1)} s` : `from cache ${(buildMs / 1000).toFixed(1)} s`);
  console.log(app.extraStats);
  Object.assign(window, { ruins: { tour, keys, poseAt, assets, placed, loadSeconds, genMs, buildMs, fullDetail, instances } });
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

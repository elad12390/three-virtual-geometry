/**
 * Stress-test map: a 2 km heightfield with forests, rocks, bushes and
 * meadows, all procedurally generated and all rendered through VirtualMesh.
 * Open with `?scene=map` (default). `?quality=low` builds a lighter version.
 */
import * as THREE from 'three/webgpu';
import { color, mix, mx_noise_float, normalWorld, positionWorld, smoothstep as tslSmoothstep } from 'three/tsl';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { buildVirtualMeshCached, type VirtualMeshBuildOptions, type VirtualMeshData, type VirtualMeshSource } from '../../src/index';
import { VirtualMesh, vgWorldNormal } from '../../src/index';
import type { DemoApp } from './app';
import { makeBroadleaf, makeBush, makeConifer, makeGrassClump, makeRock, makeTerrain } from './assets';
import { Perlin, mulberry32, smoothstep } from './noise';

const MAP_SIZE = 2048;
const WATER_LEVEL = 8;

interface AssetSpec {
  name: string;
  make: () => VirtualMeshSource;
  build?: VirtualMeshBuildOptions;
}

export async function createMapScene(app: DemoApp) {
  const quality = new URLSearchParams(location.search).get('quality') === 'low' ? 'low' : 'high';
  const density = quality === 'low' ? 0.25 : 1;
  const { scene, camera, controls, vg } = app;
  const perlin = new Perlin(1337);

  // ---------------------------------------------------------------- terrain shape
  const lakeCenter = new THREE.Vector2(260, 220);
  const heightAt = (x: number, z: number) => {
    const hills = perlin.fbm2(x / 650, z / 650, 5) * 60;
    const detail = perlin.fbm2(x / 90, z / 90, 3) * 4;
    const north = smoothstep(-150, -850, z);
    const mountains = perlin.ridged2(x / 480 + 10, z / 480 - 4, 6) * 300 * north;
    const lake = smoothstep(420, 120, Math.hypot(x - lakeCenter.x, z - lakeCenter.y)) * -70;
    return 30 + hills + detail + mountains + lake;
  };
  const uprightness = (x: number, z: number) => {
    const e = 2;
    const dx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e);
    const dz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
    return 1 / Math.sqrt(1 + dx * dx + dz * dz); // = normal.y
  };
  const forest = (x: number, z: number) => perlin.fbm2(x / 380 + 50, z / 380 - 20, 4);

  // ---------------------------------------------------------------- assets
  const terrainSegments = quality === 'low' ? 256 : 512;
  const specs: AssetSpec[] = [
    { name: 'terrain', make: () => makeTerrain(MAP_SIZE, terrainSegments, heightAt) },
    { name: 'boulder', make: () => makeRock(11, quality === 'low' ? 48 : 80) },
    { name: 'stone', make: () => makeRock(23, quality === 'low' ? 32 : 56) },
    { name: 'conifer', make: () => makeConifer(5) },
    { name: 'broadleaf', make: () => makeBroadleaf(9) },
    { name: 'bush', make: () => makeBush(3) },
    { name: 'grass', make: () => makeGrassClump(7, 56), build: { prune: true } },
    { name: 'flowers', make: () => makeGrassClump(13, 36, 10), build: { prune: true } },
  ];

  const assets: Record<string, VirtualMeshData> = {};
  let buildMs = 0;
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i];
    const label = `Building ${spec.name} (${i + 1}/${specs.length})`;
    app.progress(`${label}: generating…`, i / specs.length);
    await new Promise((r) => setTimeout(r, 0));
    const source = spec.make();
    assets[spec.name] = await buildVirtualMeshCached(source, {
      ...spec.build,
      onProgress: (f) => app.progress(`${label}: meshlet DAG…`, (i + f) / specs.length),
    });
    buildMs += assets[spec.name].stats.buildMs;
    console.log(spec.name, assets[spec.name].stats, 'meshlets', assets[spec.name].meshletCount);
  }

  // ---------------------------------------------------------------- scatter
  const rand = mulberry32(42);
  const tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), e: new THREE.Euler(), p: new THREE.Vector3(), s: new THREE.Vector3() };
  type Placement = { x: number; z: number; h: number; up: number };
  const scatter = (
    count: number,
    accept: (pl: Placement) => number,
    region: { cx: number; cz: number; r: number } | null = null
  ): Placement[] => {
    const result: Placement[] = [];
    const maxTries = count * 40;
    for (let t = 0; t < maxTries && result.length < count; t++) {
      let x: number;
      let z: number;
      if (region) {
        const a = rand() * Math.PI * 2;
        const d = Math.sqrt(rand()) * region.r;
        x = region.cx + Math.cos(a) * d;
        z = region.cz + Math.sin(a) * d;
      } else {
        x = (rand() - 0.5) * MAP_SIZE * 0.98;
        z = (rand() - 0.5) * MAP_SIZE * 0.98;
      }
      const pl = { x, z, h: heightAt(x, z), up: uprightness(x, z) };
      if (rand() < accept(pl)) result.push(pl);
    }
    return result;
  };
  const toInstances = (
    placements: Placement[],
    opts: { scale: [number, number]; sink?: number; tilt?: number; tint?: [number, number] }
  ) => {
    const matrices = new Float32Array(placements.length * 16);
    const colors = new Float32Array(placements.length * 4);
    placements.forEach((pl, i) => {
      const s = opts.scale[0] + rand() * (opts.scale[1] - opts.scale[0]);
      const tilt = opts.tilt ?? 0;
      tmp.e.set((rand() - 0.5) * tilt, rand() * Math.PI * 2, (rand() - 0.5) * tilt);
      tmp.q.setFromEuler(tmp.e);
      tmp.p.set(pl.x, pl.h - (opts.sink ?? 0) * s, pl.z);
      tmp.m.compose(tmp.p, tmp.q, tmp.s.setScalar(s));
      tmp.m.toArray(matrices, i * 16);
      const [t0, t1] = opts.tint ?? [1, 1];
      const k = t0 + rand() * (t1 - t0);
      colors.set([k * (0.95 + rand() * 0.1), k * (0.95 + rand() * 0.1), k * (0.95 + rand() * 0.1), 1], i * 4);
    });
    return { matrices, colors };
  };

  const start = new THREE.Vector3(-120, 0, 520);
  const meadow = { cx: start.x + 120, cz: start.z - 260, r: 520 };
  const n = (c: number) => Math.round(c * density);
  const placements = {
    conifer: scatter(n(9000), (p) => (p.h > 18 && p.h < 260 && p.up > 0.8 ? smoothstep(-0.05, 0.25, forest(p.x, p.z)) * (p.h > 45 ? 1 : 0.35) : 0)),
    broadleaf: scatter(n(3500), (p) => (p.h > WATER_LEVEL + 3 && p.h < 80 && p.up > 0.86 ? smoothstep(-0.1, 0.2, forest(p.x, p.z)) : 0)),
    bush: scatter(n(9000), (p) => (p.h > WATER_LEVEL + 1.5 && p.h < 150 && p.up > 0.8 ? 0.15 + 0.85 * smoothstep(0.3, 0, Math.abs(forest(p.x, p.z))) : 0)),
    boulder: scatter(n(1400), (p) => (p.h > WATER_LEVEL - 2 ? (p.up < 0.85 ? 0.9 : p.h > 140 ? 0.5 : 0.08) : 0)),
    stone: scatter(n(3500), (p) => (p.h > WATER_LEVEL - 1 ? (p.up < 0.9 ? 0.7 : 0.15) : 0)),
    grass: scatter(n(110000), (p) => (p.h > WATER_LEVEL + 1 && p.up > 0.88 && forest(p.x, p.z) < 0.12 ? 0.9 : 0), meadow),
    flowers: scatter(n(16000), (p) => (p.h > WATER_LEVEL + 1.5 && p.up > 0.9 && forest(p.x, p.z) < 0 ? 0.8 : 0), meadow),
  };

  // ---------------------------------------------------------------- materials
  const foliage = (roughness = 0.85) => new THREE.MeshStandardNodeMaterial({ roughness, side: THREE.DoubleSide });
  const terrainMaterial = new THREE.MeshStandardNodeMaterial({ roughness: 0.95 });
  {
    const h = positionWorld.y;
    const up = vgWorldNormal.y;
    const large = mx_noise_float(positionWorld.mul(0.02)).mul(0.5).add(0.5);
    const small = mx_noise_float(positionWorld.mul(0.35)).mul(0.5).add(0.5);
    let c = mix(color(0x3d6527), color(0x6f8c37), large.mul(0.7).add(small.mul(0.3)));
    c = mix(c, color(0x6b5a40), tslSmoothstep(0.93, 0.82, up).mul(0.7));
    c = mix(c, color(0x787168), tslSmoothstep(0.8, 0.66, up));
    c = mix(color(0xc8b78a), c, tslSmoothstep(WATER_LEVEL + 0.5, WATER_LEVEL + 3.5, h));
    c = mix(c, color(0xf0f4f8), tslSmoothstep(190, 235, h).mul(tslSmoothstep(0.5, 0.72, up)));
    terrainMaterial.colorNode = c;
  }

  const meshes: Record<string, VirtualMesh> = {};
  const add = (name: string, material: THREE.NodeMaterial, inst: { matrices: Float32Array; colors?: Float32Array }) => {
    const mesh = new VirtualMesh(vg, assets[name], material, inst);
    mesh.name = name;
    // small foliage skips shadows: every caster re-draws its meshlets in the shadow pass
    mesh.castShadow = ['terrain', 'boulder', 'stone', 'conifer', 'broadleaf'].includes(name);
    mesh.receiveShadow = true;
    scene.add(mesh);
    meshes[name] = mesh;
  };
    add('terrain', terrainMaterial, { matrices: new THREE.Matrix4().toArray(new Float32Array(16)) });

  add('boulder', new THREE.MeshStandardNodeMaterial({ roughness: 0.9 }), toInstances(placements.boulder, { scale: [2.5, 9], sink: 0.15, tilt: 0.3, tint: [0.85, 1.1] }));
  add('stone', new THREE.MeshStandardNodeMaterial({ roughness: 0.9 }), toInstances(placements.stone, { scale: [0.4, 1.6], sink: 0.1, tilt: 0.5, tint: [0.85, 1.1] }));
  add('conifer', foliage(), toInstances(placements.conifer, { scale: [0.7, 1.35], sink: 0.3, tint: [0.8, 1.15] }));
  add('broadleaf', foliage(), toInstances(placements.broadleaf, { scale: [0.75, 1.3], sink: 0.3, tint: [0.8, 1.2] }));
  add('bush', foliage(), toInstances(placements.bush, { scale: [0.6, 1.6], sink: 0.15, tint: [0.8, 1.2] }));
  add('grass', foliage(0.95), toInstances(placements.grass, { scale: [0.8, 1.5], sink: 0.02, tilt: 0.15, tint: [0.75, 1.15] }));
  add('flowers', foliage(0.9), toInstances(placements.flowers, { scale: [0.8, 1.3], sink: 0.02, tilt: 0.15, tint: [0.85, 1.1] }));

  // ---------------------------------------------------------------- environment
  const sunDirection = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(58), THREE.MathUtils.degToRad(215));
  const sky = new SkyMesh();
  sky.scale.setScalar(20000);
  sky.turbidity.value = 4;
  sky.rayleigh.value = 1.2;
  sky.sunPosition.value.copy(sunDirection);
  scene.add(sky);
  scene.fog = new THREE.FogExp2(0xb4c8d8, 0.00045);

  scene.add(new THREE.HemisphereLight(0xcfe2ff, 0x4d4030, 2.4));
  const sun = new THREE.DirectionalLight(0xfff0dc, 3.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  const shadowExtent = 220;
  Object.assign(sun.shadow.camera, { left: -shadowExtent, right: shadowExtent, top: shadowExtent, bottom: -shadowExtent, near: 1, far: 2000 });
  sun.shadow.bias = -0.0004;
  sun.shadow.intensity = 0.7; // soften shadow density so shaded foliage keeps its color
  sun.shadow.normalBias = 0.6;
  scene.add(sun, sun.target);
  app.renderer.shadowMap.enabled = true;

  const water = new THREE.Mesh(
    new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE).rotateX(-Math.PI / 2),
    new THREE.MeshStandardNodeMaterial({ color: 0x2a5a70, roughness: 0.06, metalness: 0.2, transparent: true, opacity: 0.88 })
  );
  water.position.y = WATER_LEVEL;
  water.receiveShadow = true;
  scene.add(water);

  // ---------------------------------------------------------------- camera
  camera.near = 0.2;
  camera.far = 6000;
  camera.updateProjectionMatrix();
  start.y = heightAt(start.x, start.z) + 14;
  camera.position.copy(start);
  controls.target.set(start.x + 160, heightAt(start.x + 160, start.z - 420) + 25, start.z - 420);
  controls.maxPolarAngle = Math.PI * 0.495;

  const focus = new THREE.Vector3(); // scratch, reused every frame
  app.onFrame = () => {
    // keep the shadow frustum centered in front of the camera
    camera.getWorldDirection(focus).multiplyScalar(shadowExtent * 0.6).add(camera.position);
    sun.target.position.copy(focus);
    sun.position.copy(focus).addScaledVector(sunDirection, 900);
  };

  // ---------------------------------------------------------------- UI
  const folder = app.gui.addFolder('Map');
  const visibility = Object.fromEntries(Object.keys(meshes).map((k) => [k, true]));
  for (const name of Object.keys(meshes)) {
    folder.add(visibility, name).onChange((v: boolean) => (meshes[name].visible = v));
  }
  const env = { shadows: true };
  folder.add(env, 'shadows').onChange((v: boolean) => (sun.castShadow = v));

  const totalMeshlets = Object.values(assets).reduce((a, d) => a + d.meshletCount, 0);
  app.extraStats = `assets: ${Object.keys(assets).length}, ${totalMeshlets} meshlets, DAG build ${(buildMs / 1000).toFixed(1)} s (${quality})`;
}

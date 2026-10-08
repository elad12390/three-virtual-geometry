/**
 * Jungle island: procedural coconut palms, broadleaf trees, banana plants and ferns (every leaflet real geometry),
 * with CC0 shrubs and ground cover from Poly Haven, scattered by the thousand over hills that fall away to the sea,
 * all swaying in the wind.
 *
 * The models are not in the repository: `npm run demo:forest` downloads them (about 100 MB) and opens
 * `?scene=forest`. Options: `&trees=N` (canopy trees, default 1500), `&tour` (fly-through).
 */
import * as THREE from 'three/webgpu';
import { float, max, sin, time, vec3 } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import type { DemoApp } from './app';
import type { BenchPose } from './bench';
import { buildVirtualMeshCached, type VirtualMeshData } from '../../src/index';
import { mulberry32, Perlin, smoothstep } from './noise';
import { makeBanana, makeFern, makePalm } from './tropical';
import { makeBroadleaf } from './assets';
import { makeTour, type TourKey } from './tour';

const BASE = 'scans/forest/';
/** Half the side of the forested square, in metres. */
const EXTENT = 330;
/** The clearing the tour flies down into, and the path leading to it. */
const CLEARING = new THREE.Vector2(40, 90);
const CLEARING_RADIUS = 38;

interface Credit {
  id: string;
  name: string;
  role: 'tree' | 'ground' | 'terrain';
  file: string;
}

export async function createForestScene(app: DemoApp) {
  const { scene, camera, controls, vg, renderer } = app;
  const params = new URLSearchParams(location.search);
  const random = mulberry32(5);
  const perlin = new Perlin(11);

  // Keep detail: the engine may coarsen to at most 3 px to stay inside its draw buffers.
  vg.maxErrorThreshold = 3;
  const response = await fetch(`${BASE}credits.json`).catch(() => null);
  if (!response?.ok || !response.headers.get('content-type')?.includes('json')) {
    throw new Error('The forest models are not downloaded yet. Run "npm run demo:forest" in the repository: it downloads them (about 100 MB) and opens this scene.');
  }
  const credits = (await response.json()) as Credit[];

  // ---------------------------------------------------------------- terrain
  /** An island: rolling jungle hills, flattened in the clearing, falling away to beaches and the sea (y = 0). */
  const clearingWeight = (x: number, z: number) => smoothstep(CLEARING_RADIUS + 25, CLEARING_RADIUS - 5, Math.hypot(x - CLEARING.x, z - CLEARING.y));
  const land = (x: number, z: number) => smoothstep(EXTENT + 110, EXTENT - 30, Math.hypot(x, z) * (1 + 0.15 * perlin.fbm2(x / 200 + 3, z / 200, 2)));
  const heightAt = (x: number, z: number) => {
    const hills = 16 * (perlin.fbm2(x / 260, z / 260, 4) + 0.35) + 3 * perlin.fbm2(x / 60, z / 60, 3);
    const island = land(x, z);
    return (hills * (1 - 0.8 * clearingWeight(x, z)) + 2.5) * island - 6 * (1 - island);
  };
  const textureLoader = new THREE.TextureLoader();
  const groundTexture = (name: string, srgb = false) => {
    const t = textureLoader.load(`${BASE}textures/forrest_ground_01/forrest_ground_01_${name}_2k.jpg`);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(1600 / 2.5, 1600 / 2.5); // the scan covers 2 x 2 m; a little larger hides the tiling
    t.anisotropy = 16;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  const terrainGeometry = new THREE.PlaneGeometry(1600, 1600, 400, 400).rotateX(-Math.PI / 2);
  const position = terrainGeometry.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < position.count; i++) position.setY(i, heightAt(position.getX(i), position.getZ(i)));
  terrainGeometry.computeVertexNormals();
  const arm = groundTexture('arm');
  const terrain = new THREE.Mesh(
    terrainGeometry,
    new THREE.MeshStandardNodeMaterial({ map: groundTexture('diff', true), normalMap: groundTexture('nor_gl'), roughnessMap: arm, aoMap: arm, color: 0xa9b48e })
  );
  terrain.receiveShadow = true;
  scene.add(terrain);
  const sea = new THREE.Mesh(new THREE.PlaneGeometry(20000, 20000).rotateX(-Math.PI / 2), new THREE.MeshStandardNodeMaterial({ color: 0x1f6f8c, roughness: 0.08, metalness: 0.1 }));
  scene.add(sea);

  // ---------------------------------------------------------------- light and air
  const sunDirection = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 38), THREE.MathUtils.degToRad(300));
  const makeSky = (scale: number) => {
    const sky = new SkyMesh();
    sky.scale.setScalar(scale);
    sky.turbidity.value = 6;
    sky.rayleigh.value = 1.6;
    sky.mieCoefficient.value = 0.006;
    sky.mieDirectionalG.value = 0.86;
    sky.sunPosition.value.copy(sunDirection);
    return sky;
  };
  scene.add(makeSky(20000));
  const envScene = new THREE.Scene();
  envScene.add(makeSky(400));
  const envGround = new THREE.Mesh(new THREE.CircleGeometry(300, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicNodeMaterial({ color: 0x3a4a2c }));
  envGround.position.y = -2;
  envScene.add(envGround);
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(envScene, 0, 0.5, 1000).texture;
  scene.environmentIntensity = 0.35;
  scene.fog = new THREE.FogExp2(0xc4d2d6, 0.0026); // tropical haze: each row of trees a little paler than the last
  scene.add(new THREE.HemisphereLight(0xb8c8d8, 0x4a3c28, 0.75));
  const sun = new THREE.DirectionalLight(0xffd7a8, 3.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0003;
  sun.shadow.intensity = 0.9;
  scene.add(sun, sun.target);
  renderer.shadowMap.enabled = true;

  // ---------------------------------------------------------------- models
  app.progress('Loading the tree and forest-floor models…', 0);
  const loader = new GLTFLoader();
  /** Each file holds several variants side by side (three firs, 17 grass clumps, ...): one model per variant. */
  const variants = new Map<string, string[]>();
  const models = new Map<string, THREE.Object3D>();
  let loaded = 0;
  await Promise.all(
    credits
      .filter((c) => c.role !== 'terrain')
      .map(async (credit) => {
        const gltf = await loader.loadAsync(BASE + credit.file);
        prepareModel(gltf.scene);
        const keys: string[] = [];
        for (const [i, node] of [...gltf.scene.children].entries()) {
          node.position.set(0, node.position.y, 0); // variants are laid out along x: bring each to the origin
          const key = `${credit.id}#${i}`;
          models.set(key, node);
          keys.push(key);
        }
        variants.set(credit.id, keys);
        app.progress('Loading the tree and forest-floor models…', ++loaded / (credits.length - 1));
      })
  );

  // ---------------------------------------------------------------- planting
  const forest = new THREE.Group();
  const placements = new Map<string, THREE.Matrix4[]>();
  const plant = (model: string, x: number, z: number, scale: number, yaw = random() * Math.PI * 2, sink = 0.15) => {
    const keys = variants.get(model);
    if (!keys?.length) return;
    const id = keys[Math.floor(random() * keys.length)];
    const list = placements.get(id) ?? [];
    placements.set(id, list);
    list.push(new THREE.Matrix4().compose(new THREE.Vector3(x, heightAt(x, z) - sink, z), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw), new THREE.Vector3(scale, scale, scale)));
  };
  const pathWeight = (x: number, z: number) => smoothstep(9, 3, Math.abs(x - CLEARING.x * ((z + EXTENT) / (CLEARING.y + EXTENT))) ) * (z < CLEARING.y ? 1 : 0);
  /** 0 in the open (clearing, path), 1 in deep forest; patchy so the canopy has gaps and dense stands. */
  const density = (x: number, z: number) => (1 - clearingWeight(x, z)) * (1 - pathWeight(x, z)) * smoothstep(-0.35, 0.25, perlin.fbm2(x / 90 + 7, z / 90, 3));

  /** Procedural plants (palms, bananas, ferns): instance matrices per variant, turned into virtual meshes below. */
  const procedural = { palm: [[], [], [], [], [], []], broadleaf: [[], [], [], []], banana: [[], [], [], [], []], fern: [[], [], [], []] } as Record<string, number[][]>;
  const plantProcedural = (kind: string, x: number, z: number, scale: number, sink = 0.1) => {
    const list = procedural[kind][Math.floor(random() * procedural[kind].length)];
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, heightAt(x, z) - sink, z), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), random() * Math.PI * 2), new THREE.Vector3(scale, scale, scale));
    list.push(...m.elements);
  };
  const onLand = (x: number, z: number) => land(x, z) > 0.55 && heightAt(x, z) > 0.8;

  // Canopy: leafy broadleaf trees (scaled up to 11 to 20 m) and coconut palms, by rejection sampling thinned by
  // density. Palms also line the coast.
  const canopyCount = Math.max(0, Number(params.get('trees') ?? 2200));
  for (let i = 0, tries = 0; i < canopyCount && tries < canopyCount * 30; tries++) {
    const x = (random() * 2 - 1) * (EXTENT + 60);
    const z = (random() * 2 - 1) * (EXTENT + 60);
    if (!onLand(x, z)) continue;
    const coast = 1 - smoothstep(0.55, 0.9, land(x, z));
    if (random() > Math.max(density(x, z), coast * 0.8)) continue;
    i++;
    if (random() < 0.6 || coast > 0.3) plantProcedural('palm', x, z, 0.9 + random() * 0.35, 0.2);
    else plantProcedural('broadleaf', x, z, 0.9 + random() * 0.6, 0.2);
  }
  // Understory: banana clumps, jungle shrubs and young trees, thickest under and around the canopy.
  for (let i = 0; i < canopyCount * 2.5; i++) {
    const x = (random() * 2 - 1) * EXTENT;
    const z = (random() * 2 - 1) * EXTENT;
    if (!onLand(x, z)) continue;
    const d = density(x, z);
    if (random() > d * 0.9 + 0.1) continue;
    const r = random();
    if (r < 0.5) {
      for (let k = 0, n = 2 + Math.floor(random() * 4); k < n; k++) plantProcedural('banana', x + (random() - 0.5) * 3, z + (random() - 0.5) * 3, 0.85 + random() * 0.4);
    } else if (r < 0.8) plant('shrub_02', x, z, 1.4 + random() * 1.2);
    else plant('shrub_01', x, z, 2 + random() * 2.5, undefined, 0.05);
  }
  // Jungle floor: ferns, calathea, low shrubs and nettles under the trees, grass in the open.
  const groundCount = Number(params.get('ground') ?? 50000);
  for (let i = 0; i < groundCount; i++) {
    const x = (random() * 2 - 1) * EXTENT;
    const z = (random() * 2 - 1) * EXTENT;
    if (!onLand(x, z)) continue;
    const d = density(x, z);
    const r = random();
    if (r < 0.3) {
      if (random() < 0.3 + d * 0.7) plantProcedural('fern', x, z, 0.8 + random() * 0.9, 0.05);
    } else if (r < 0.45) {
      if (random() < d) plant('calathea_orbifolia_01', x, z, 1.6 + random() * 1.8, undefined, 0.02);
    } else if (r < 0.58) {
      plant(random() < 0.5 ? 'shrub_03' : 'nettle_plant', x, z, 2 + random() * 3, undefined, 0.02);
    } else if (r < 0.66) {
      plant('shrub_04', x, z, 3 + random() * 3, undefined, 0.02);
    } else if (r < 0.9) {
      if (random() < 1 - d * 0.7) plant('grass_medium_01', x, z, 1.5 + random() * 2.5, undefined, 0.02);
    } else if (r < 0.95) {
      if (random() < d) plant('root_cluster_02', x, z, 1 + random() * 1.2, undefined, 0.05);
    } else if (random() < 0.5) {
      plant('rock_moss_set_01', x, z, 0.6 + random() * 1.4, undefined, 0.3);
    }
  }
  // One InstancedMesh per part of each model: vg.add turns each into an instanced virtual mesh.
  const GROUND_IDS = new Set([...credits.filter((c) => c.role === 'ground').map((c) => c.id), 'calathea_orbifolia_01', 'shrub_03', 'shrub_04', 'nettle_plant']);
  for (const [key, matrices] of placements) {
    const model = models.get(key)!;
    const id = key.split('#')[0];
    model.updateMatrixWorld(true);
    model.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const instanced = new THREE.InstancedMesh(mesh.geometry, mesh.material, matrices.length);
      const m = new THREE.Matrix4();
      matrices.forEach((placement, i) => instanced.setMatrixAt(i, m.multiplyMatrices(placement, mesh.matrixWorld)));
      instanced.name = `${id}/${mesh.name}`;
      instanced.userData.foliage =
        /twig|leaves|leaf|branch/i.test(mesh.name + (mesh.material as THREE.Material).name) ||
        ['grass_medium_01', 'celandine_01', 'calathea_orbifolia_01', 'shrub_02', 'shrub_03', 'shrub_04', 'nettle_plant'].includes(id);
      instanced.castShadow = !GROUND_IDS.has(id) || id === 'rock_moss_set_01';
      instanced.receiveShadow = true;
      instanced.userData.ground = GROUND_IDS.has(id);
      forest.add(instanced);
    });
  }
  scene.add(forest);

  const t0 = performance.now();
  const imported = await vg.add(forest, {
    // Foliage here is alpha-cutout cards: coarser levels should drop whole cards (prune) rather than merge them
    // into stretched triangles, and voxel stand-ins would lose the leaf textures. Keep every build cached: the
    // trees alone are several hundred MB of built data.
    build: { prune: true, voxelLods: false, cache: { maxBytes: 4 * 1024 ** 3 } },
    // Small forest-floor detail stops being drawn at a distance (it shrinks away, so nothing pops).
    mesh: (group) => {
      const source = group.instances[0]?.object.userData ?? {};
      // Trees: every part bends from the base (bark included, so the whole tree moves together), and leaves and
      // twigs flutter on top. Grass and flowers only flutter.
      if (source.ground) return { maxDrawDistance: 90, deform: source.foliage ? wind(0, 0.05) : undefined };
      return { deform: wind(1, source.foliage ? 0.09 : 0) };
    },
    onProgress: async (f) => {
      app.progress('Building the trees (first visit only, then cached)…', f);
      await new Promise((r) => setTimeout(r, 0));
    },
  });
  console.log('forest import', imported.stats, `${((performance.now() - t0) / 1000).toFixed(1)} s`);

  // Procedural palms, bananas and ferns: a few variants each, every leaflet real geometry, instanced by the thousand.
  const leafy = new THREE.MeshStandardNodeMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.7 });
  const makers: Record<string, (seed: number) => Parameters<typeof buildVirtualMeshCached>[0]> = { palm: makePalm, broadleaf: makeBroadleaf, banana: makeBanana, fern: makeFern };
  const motion: Record<string, ReturnType<typeof wind>> = { palm: wind(1, 0.07), broadleaf: wind(0.8, 0.06), banana: wind(0.5, 0.05), fern: wind(0, 0.03) };
  let proceduralTriangles = 0;
  let proceduralInstances = 0;
  for (const [kind, lists] of Object.entries(procedural)) {
    for (const [v, matrices] of lists.entries()) {
      if (!matrices.length) continue;
      app.progress(`Building ${kind}s…`, v / lists.length);
      const data: VirtualMeshData = await buildVirtualMeshCached(makers[kind](101 + v * 17), { cache: { maxBytes: 4 * 1024 ** 3 } });
      const mesh = vg.createMesh(data, leafy, { matrices: new Float32Array(matrices) }, { deform: motion[kind], maxDrawDistance: kind === 'fern' ? 90 : undefined });
      mesh.name = `${kind} ${v}`;
      mesh.castShadow = kind !== 'fern';
      mesh.receiveShadow = true;
      scene.add(mesh);
      proceduralTriangles += mesh.fullDetailTriangles;
      proceduralInstances += mesh.instanceCount;
    }
  }

  // ---------------------------------------------------------------- camera
  // Fly-through: low among ferns and trunks, up through the canopy, out over the forest, down into the clearing.
  const ground = (x: number, z: number) => heightAt(x, z) + 1.6;
  const h = (x: number, z: number) => heightAt(x, z);
  const keys = [
    { target: [-20, h(-20, -120) + 1.2, -120], dist: 4, yaw: Math.PI + 0.4, pitch: 0.12, time: 0 },
    { target: [-10, h(-10, -80) + 2.5, -80], dist: 10, yaw: Math.PI + 0.1, pitch: 0.1, time: 12 },
    { target: [5, h(5, -40) + 8, -40], dist: 18, yaw: Math.PI - 0.2, pitch: 0.25, time: 24 },
    { target: [10, h(10, 0) + 22, 0], dist: 30, yaw: Math.PI - 0.1, pitch: 0.12, time: 36 },
    { target: [20, 10, 60], dist: 220, yaw: Math.PI + 0.6, pitch: 0.28, time: 50 },
    { target: [CLEARING.x, h(CLEARING.x, CLEARING.y) + 3, CLEARING.y], dist: 70, yaw: 2.6, pitch: 0.32, time: 64 },
    { target: [CLEARING.x, h(CLEARING.x, CLEARING.y) + 2, CLEARING.y], dist: 22, yaw: 1.4, pitch: 0.12, time: 76 },
    { target: [0, 0, 0], dist: 480, yaw: Math.PI + 1.1, pitch: 0.3, time: 92 },
    { target: [-20, h(-20, -120) + 1.2, -120], dist: 4, yaw: Math.PI + 0.4, pitch: 0.12, time: 106 },
  ] satisfies TourKey[];
  const recordSeconds = 95;
  const { poseAt, duration: tourSeconds } = makeTour(keys, ground);
  app.benchPoses = [
    { name: 'forest floor', position: [-18, h(-18, -124) + 1.8, -124], target: [-20, h(-20, -120) + 1.2, -120] },
    { name: 'among the trees', position: [-12, h(-12, -90) + 3, -90], target: [-10, h(-10, -80) + 2.5, -80] },
    { name: 'above the canopy', position: [-180, 110, -150], target: [20, 10, 60] },
    { name: 'clearing', position: [CLEARING.x + 20, h(CLEARING.x, CLEARING.y) + 5, CLEARING.y + 8], target: [CLEARING.x, h(CLEARING.x, CLEARING.y) + 2, CLEARING.y] },
  ] satisfies BenchPose[];

  camera.far = 5000;
  poseAt(0, controls.target, camera.position);
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 0.3;
  camera.updateProjectionMatrix();

  const tour = { active: params.has('tour'), time: 0 };
  const stopTour = () => (tour.active = false);
  renderer.domElement.addEventListener('pointerdown', stopTour);
  renderer.domElement.addEventListener('wheel', stopTour, { passive: true });
  const focus = new THREE.Vector3();
  const look = new THREE.Vector3();
  app.onFrame = (dt) => {
    if (tour.active) {
      tour.time = (tour.time + dt) % tourSeconds;
      poseAt(tour.time, controls.target, camera.position);
    }
    const dist = camera.position.distanceTo(controls.target);
    const above = camera.position.y - heightAt(camera.position.x, camera.position.z);
    const near = THREE.MathUtils.clamp(Math.min(dist, above + 1) * 0.02, 0.03, 1.5);
    if (Math.abs(near - camera.near) > camera.near * 0.15) {
      camera.near = near;
      camera.updateProjectionMatrix();
    }
    // The shadow map follows the camera and covers more ground as it rises.
    const extent = THREE.MathUtils.clamp(Math.max(dist, above) * 1.4, 30, 300);
    const sc = sun.shadow.camera;
    if (Math.abs(sc.right - extent) > extent * 0.1) {
      Object.assign(sc, { left: -extent, right: extent, top: extent, bottom: -extent, near: 1, far: 3000 });
      sc.updateProjectionMatrix();
      sun.shadow.normalBias = extent * 0.0012;
    }
    camera.getWorldDirection(look);
    focus.copy(camera.position).addScaledVector(look, Math.min(dist, extent * 0.7));
    sun.target.position.copy(focus);
    sun.position.copy(focus).addScaledVector(sunDirection, 1200);
  };

  Object.assign(window, { forest: { tour, tourSeconds, recordSeconds } }); // for recording the fly-through
  const folder = app.gui.addFolder('Forest');
  folder.add({ 'fly-through': () => ((tour.active = !tour.active), tour.active && (tour.time = 0)) }, 'fly-through');
  folder.add(sun, 'castShadow').name('shadows');

  const importedTriangles = imported.meshes.reduce((a, m) => a + m.fullDetailTriangles, 0);
  app.extraStats =
    `jungle: ${(imported.stats.instances + proceduralInstances).toLocaleString()} plants, ` +
    `${((importedTriangles + proceduralTriangles) / 1e9).toFixed(1)}B triangles (procedural palms, trees, bananas and ferns; Poly Haven CC0 shrubs and ground cover)`;
  console.log('jungle scene', { importedTriangles, proceduralTriangles, proceduralInstances });
}

/**
 * Wind, in the vertex shader. `sway` bends a whole tree away from the wind, more toward the top (with the square of
 * the height above its base), in slow gusts that roll across the forest so neighbouring trees move a little out of
 * step. `flutter` adds faster, smaller motion to leaves, twigs and grass.
 */
function wind(sway: number, flutter: number) {
  /* eslint-disable @typescript-eslint/no-explicit-any -- TSL nodes */
  return (p: any, { instanceOrigin: origin }: { instanceOrigin: any }) => {
    const phase = origin.x.mul(0.045).add(origin.z.mul(0.03));
    const gust = sin(time.mul(0.45).add(phase)).mul(0.5).add(0.5); // 0..1, rolling across the forest
    let offset: any = vec3(0, 0, 0);
    if (sway > 0) {
      const height = max(p.y.sub(origin.y), 0).div(16); // 1 at the top of a canopy tree
      const bend = height.mul(height).mul(gust.mul(0.55).add(0.15).add(sin(time.mul(1.3).add(phase.mul(3))).mul(0.06)));
      offset = vec3(0.9, 0, 0.45).mul(bend).mul(float(sway));
    }
    if (flutter > 0) {
      const local = p.x.mul(0.9).add(p.z.mul(0.7)).add(p.y.mul(0.5));
      const shake = vec3(sin(time.mul(3.1).add(local)), sin(time.mul(4.3).add(local.mul(1.3))).mul(0.4), sin(time.mul(2.7).add(local.mul(0.8))));
      offset = offset.add(shake.mul(gust.mul(0.6).add(0.4)).mul(float(flutter)));
    }
    return p.add(offset);
  };
  /* eslint-enable @typescript-eslint/no-explicit-any */
}

/**
 * Foliage in these models is alpha-blended (transparent): switch it to alpha cutout, the standard for foliage. Cutout
 * leaves sort correctly, cast proper shadows and hide what is behind them. Every part casts and receives shadows.
 */
function prepareModel(object: THREE.Object3D) {
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (material.transparent) {
        material.transparent = false;
        material.alphaTest = 0.4;
        material.depthWrite = true;
      }
    }
  });
  return object;
}

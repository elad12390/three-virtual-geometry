/**
 * A museum of real scans: 3D scans and photogrammetry, the kind of data virtual geometry was made for, shown as
 * exhibits in a grand museum (see museum.ts): statues in the galleries and the rotunda (scans of plaster casts of
 * classical sculpture, SMK, National Gallery of Denmark, public domain, 1 to 4 million triangles each), monumental
 * copies on the hero bases, and photoscanned coastal rocks and cliffs (Poly Haven, CC0, 1 to 2.9 million triangles
 * each) on the pedestals of the Great Court.
 *
 * The scans are not in the repository: `npm run demo:scans` downloads them (about 2 GB), bakes the statues to .vgeo
 * and opens `?scene=scans`. `&tour` plays the fly-through.
 */
import * as THREE from 'three/webgpu';
import { color, float, mix, mx_noise_float, positionWorld, smoothstep } from 'three/tsl';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadVirtualMesh, type VirtualMeshData } from '../../src/index';
import type { DemoApp } from './app';
import { buildMuseum, type ExhibitSlot } from './museum';
import { mulberry32 } from './noise';
import { makeTour, type TourKey } from './tour';

const BASE = 'scans/';

interface Credit {
  id: string;
  name: string;
  file: string;
  triangles: number;
  source: string;
  license: string;
}

/**
 * Turn (radians) that brings a scan's front to -z after `uprightFit`. The SMK scans face +z there (+y in their own
 * axes), except Thalia. Checked one by one with `&lineup`.
 */
const FRONTS: Record<string, number> = { KAS35: 0 };
const FRONT_TURN = (id: string) => FRONTS[id] ?? Math.PI;

/** Statue scans shown monumental on the hero bases, in order of preference. */
const HEROES = ['KAS224', 'KAS1026', 'KAS35', 'KAS255', 'KAS357'];

interface Statue {
  id: string;
  name: string;
  data: VirtualMeshData;
  /** Stands the scan upright on y = 0, centred, front facing -z. */
  fit: THREE.Matrix4;
  /** Height and largest footprint, in the scan's units. */
  height: number;
  footprint: number;
  /** Reclining or group pieces, much wider than tall: they go on the long plinths. */
  long: boolean;
}

export async function createScansScene(app: DemoApp) {
  const { scene, camera, controls, vg, renderer } = app;
  const params = new URLSearchParams(location.search);
  const random = mulberry32(11);

  const response = await fetchJson(`${BASE}credits.json`);
  if (!response) {
    throw new Error('The scans are not downloaded yet. Run "npm run demo:scans" in the repository: it downloads them (about 2 GB) and opens this scene.');
  }
  const credits = (await response.json()) as Credit[];

  // ---------------------------------------------------------------- the scans
  const statues: Statue[] = [];
  const statueCredits = credits.filter((c) => c.file.endsWith('.stl'));
  for (let i = 0; i < statueCredits.length; i++) {
    const credit = statueCredits[i];
    app.progress(`Loading statue scans: ${credit.name}…`, i / statueCredits.length);
    const dir = `${BASE}baked/${credit.id.replace(/\W+/g, '_')}/`;
    const found = await fetchJson(`${dir}manifest.json`);
    if (!found) {
      console.warn(`scans: ${credit.name} is not baked yet (run npm run demo:scans), skipping it`);
      continue;
    }
    const manifest = await found.json();
    const data = await loadVirtualMesh(dir + manifest.meshes[0].file);
    statues.push({ id: credit.id, name: credit.name, data, ...uprightFit(data) });
  }
  if (!statues.length) throw new Error('No statue scan is baked yet: run "npm run demo:scans".');

  if (params.has('lineup')) return lineup(app, statues);
  const museum = await buildMuseum(app);

  app.progress('Loading the rock scans…', 0);
  const loader = new GLTFLoader();
  const rockCredits = credits.filter((c) => c.file.endsWith('.gltf'));
  const rocks = (await Promise.all(rockCredits.map(async (c) => normalized((await loader.loadAsync(BASE + c.file)).scene, c.id)))).sort((a, b) =>
    a.name.localeCompare(b.name)
  );

  // ---------------------------------------------------------------- fill the slots
  // Weathered bronze, the patina varying across the surface (world-space noise: no two statues alike).
  const bronze = new THREE.MeshStandardNodeMaterial();
  const weathering = mx_noise_float(positionWorld.mul(0.8)).mul(0.5).add(0.5).add(mx_noise_float(positionWorld.mul(5)).mul(0.12));
  const patina = smoothstep(0.52, 0.8, weathering);
  bronze.colorNode = mix(color(0x8a5f36), color(0x3f6a58), patina);
  bronze.metalnessNode = mix(float(0.9), float(0.2), patina);
  bronze.roughnessNode = mix(float(0.34), float(0.75), patina);

  const matrices = new Map<Statue, number[]>();
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  /** Places `statue` on `slot`, as large as the slot allows (times `fill`). Scans face -z; slots give the front's yaw. */
  const place = (statue: Statue, slot: ExhibitSlot, fill: number) => {
    const scale = Math.min((slot.maxHeight * fill) / statue.height, (slot.maxFootprint * fill) / statue.footprint);
    q.setFromAxisAngle(up, slot.yaw + Math.PI + FRONT_TURN(statue.id) + (slot.kind === 'statue' ? (random() - 0.5) * 0.25 : 0));
    m.compose(slot.position, q, s.setScalar(scale)).multiply(statue.fit);
    if (!matrices.has(statue)) matrices.set(statue, []);
    matrices.get(statue)!.push(...m.elements);
  };
  const standing = statues.filter((statue) => !statue.long);
  const reclining = statues.filter((statue) => statue.long);
  const heroes = HEROES.map((id) => statues.find((statue) => statue.id === id)).filter((statue): statue is Statue => !!statue);
  let regular = 0;
  let wide = 0;
  let hero = 0;
  for (const slot of museum.slots) {
    if (slot.kind === 'hero') place(heroes[hero++ % heroes.length] ?? standing[0] ?? statues[0], slot, 0.92);
    else if (slot.kind === 'statue' && slot.maxFootprint > 2.5 && reclining.length) place(reclining[wide++ % reclining.length], slot, 0.92);
    else if (slot.kind === 'statue') {
      // A different scan than its neighbours: step through the list by a stride coprime with its length.
      const pool = standing.length ? standing : statues;
      const i = regular++;
      place(pool[(i * 5 + Math.floor(i / pool.length)) % pool.length], slot, 0.95);
    }
  }
  let statueTriangles = 0;
  for (const [statue, list] of matrices) {
    const mesh = vg.createMesh(statue.data, bronze, { matrices: new Float32Array(list) });
    mesh.name = statue.name;
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
    statueTriangles += mesh.fullDetailTriangles;
  }

  // Rock and cliff scans on the specimen pedestals of the Great Court, each fitted to its pedestal.
  const specimens = new THREE.Group();
  let featured: ExhibitSlot | null = null;
  let featuredShape = 0;
  museum.slots
    .filter((slot) => slot.kind === 'specimen')
    .forEach((slot, i) => {
      const source = rocks[i % rocks.length];
      if (!source) return;
      // The tour circles the chunkiest rock: most of these scans are wide, low patches of shoreline.
      const shape = source.userData.height / source.userData.width;
      if (shape > featuredShape) [featured, featuredShape] = [slot, shape];
      const rock = source.clone();
      rock.scale.setScalar(Math.min((slot.maxFootprint * 0.9) / source.userData.width, slot.maxHeight / source.userData.height));
      rock.rotation.y = random() * Math.PI * 2;
      rock.position.copy(slot.position);
      specimens.add(rock);
    });
  specimens.traverse((o) => (o.castShadow = o.receiveShadow = true));
  const specimenCount = specimens.children.length;
  scene.add(specimens);
  const specimenImport = await vg.add(specimens, {
    onProgress: async (f) => {
      app.progress('Building the rock scans (first visit only, then cached)…', f);
      await new Promise((r) => setTimeout(r, 0));
    },
  });

  // ---------------------------------------------------------------- fly-through
  const keys = exhibitTour(museum.slots, featured);
  const { poseAt, duration: tourSeconds } = makeTour(keys, () => 1.0);
  poseAt(0, controls.target, camera.position);
  controls.minDistance = 0.3;
  app.benchPoses = museum.benchPoses;
  const tour = { active: params.has('tour'), time: 0 };
  const stopTour = () => (tour.active = false);
  renderer.domElement.addEventListener('pointerdown', stopTour);
  renderer.domElement.addEventListener('wheel', stopTour, { passive: true });
  app.onFrame = (dt) => {
    if (tour.active) {
      tour.time = (tour.time + dt) % tourSeconds;
      poseAt(tour.time, controls.target, camera.position);
    }
    museum.onFrame(dt);
  };
  Object.assign(window, { scans: { tour, tourSeconds, recordSeconds: tourSeconds }, museum }); // one full loop is recorded
  const folder = app.gui.addFolder('Museum');
  folder.add({ 'fly-through': () => ((tour.active = !tour.active), tour.active && (tour.time = 0)) }, 'fly-through');

  const rockTriangles = specimenImport.meshes.reduce((a, mesh) => a + mesh.fullDetailTriangles, 0);
  const statueCount = [...matrices.values()].reduce((a, list) => a + list.length / 16, 0);
  app.extraStats =
    `museum: ${statueCount} statues from ${statues.length} scans (SMK, public domain), ${(statueTriangles / 1e9).toFixed(2)}B triangles; ` +
    `${specimenCount} rock and cliff specimens from ${rocks.length} scans (Poly Haven, CC0), ${(rockTriangles / 1e6).toFixed(0)}M triangles`;
  console.log('scans museum', { statues: statues.map((statue) => statue.name), statueTriangles, rockTriangles });
}

/**
 * The fly-through: a visit at the exhibits' own height. It looks at a statue's face up close, then arcs around its
 * front at chest height to show the whole figure; walks the gallery at eye level to a second statue; spirals all the
 * way around the giant in the rotunda from its base to its face; arcs past a statue in the vestibule; circles a rock
 * specimen in the Great Court; sweeps around the court's giant; and walks back to the first statue, easing into the
 * opening shot so the tour (and the recording) loops seamlessly. Built from the exhibit slots, so it follows the
 * layout. Yaw: the camera's side of the target (0 = on its +z side); a slot's yaw is the side its front faces.
 */
function exhibitTour(slots: ExhibitSlot[], featuredSpecimen: ExhibitSlot | null): TourKey[] {
  const nearest = (kind: ExhibitSlot['kind'], x: number, z: number) =>
    slots.filter((slot) => slot.kind === kind).sort((a, b) => Math.hypot(a.position.x - x, a.position.z - z) - Math.hypot(b.position.x - x, b.position.z - z))[0];
  const keys: TourKey[] = [];
  let lastYaw: number | null = null;
  /** Adds a key; yaw is unwrapped to the turn nearest the previous one, so the camera never spins the long way. */
  const key = (time: number, target: [number, number, number], dist: number, yaw: number, pitch: number) => {
    if (lastYaw !== null) yaw += Math.round((lastYaw - yaw) / (Math.PI * 2)) * Math.PI * 2;
    lastYaw = yaw;
    keys.push({ time, target, dist, yaw, pitch });
  };
  const at = (slot: ExhibitSlot, height: number): [number, number, number] => [slot.position.x, slot.position.y + height, slot.position.z];
  const first = nearest('statue', -70, 4);
  const second = nearest('statue', -44.4, -4);
  const vestibule = nearest('statue', 44.4, 4);
  const specimen = featuredSpecimen ?? nearest('specimen', 100, 0);
  const rotunda = nearest('hero', 0, 0);
  const court = nearest('hero', 160, 0);
  const statueTop = 3.1; // a statue fitted to a 3.4 m slot, above its plinth

  // 1. The first statue: its face up close, then an arc around its front showing the whole figure.
  key(0, at(first, statueTop - 0.25), 1.9, first.yaw - 0.35, 0.02);
  key(7, at(first, statueTop - 0.7), 2.5, first.yaw + 0.2, 0.0);
  key(15, at(first, statueTop / 2), 3.7, first.yaw + 0.85, -0.02);
  // 2. Down the gallery at eye level to the second statue, and around its front.
  key(23, [-55, 2.6, 0], 7, -Math.PI / 2, 0.04);
  key(31, at(second, statueTop / 2), 3.7, second.yaw - 0.8, 0.0);
  key(38, at(second, statueTop - 0.5), 2.4, second.yaw + 0.55, 0.04);
  // 3. Into the rotunda, and a full spiral around the giant, from its base up to its face.
  key(46, [-14, 4.5, 0], 12, -Math.PI / 2, 0.05);
  const giantCentre = at(rotunda, 4.4);
  for (let i = 0; i <= 8; i++) {
    const height = 3.5 + (i / 8) * 6; // camera height above the floor: base, rising to the face
    const dist = 11 - Math.sin((i / 8) * Math.PI) * 1.5;
    key(52 + i * 3.6, giantCentre, dist, -Math.PI / 2 + (i / 8) * Math.PI * 2.5, Math.asin(THREE.MathUtils.clamp((height - giantCentre[1]) / dist, -0.9, 0.9)));
  }
  // 4. East into the vestibule, and around a statue there.
  key(88, [40, 2.6, 0], 8, -Math.PI / 2, 0.04);
  key(95, at(vestibule, statueTop / 2), 3.7, vestibule.yaw + 0.7, 0.0);
  key(102, at(vestibule, statueTop - 0.5), 2.5, vestibule.yaw - 0.45, 0.04);
  // 5. The Great Court: past the specimens at eye level, and halfway around one of them.
  key(111, [specimen.position.x - 20, 2.6, specimen.position.z], 14, -Math.PI / 2, 0.05);
  const rock = at(specimen, 0.8);
  for (let i = 0; i <= 4; i++) key(119 + i * 4, rock, 5, -Math.PI / 2 + (i / 4) * Math.PI, 0.22);
  // 6. Around the front of the court's giant.
  const courtCentre = at(court, 3.6);
  key(143, courtCentre, 10, court.yaw - 0.9, 0.0);
  key(151, courtCentre, 8, court.yaw, -0.08);
  key(159, courtCentre, 9, court.yaw + 0.9, 0.08);
  // 7. Back west at eye level along the centre lines (the statue rows are either side), swinging out only in the
  //    rotunda to pass beside the giant, to the first statue's face: the loop closes.
  key(169, [90, 2.8, 0], 20, Math.PI / 2, 0.05);
  key(180, [22, 3, 0], 20, Math.PI / 2, 0.05);
  key(188, [-10, 3, 9], 20, Math.PI / 2, 0.05);
  key(196, [-44, 3, 0], 20, Math.PI / 2, 0.05);
  key(205, [-62, 3, 1], 9, Math.PI / 2 + 0.3, 0.04);
  key(214, at(first, statueTop - 0.25), 1.9, first.yaw - 0.35, 0.02);
  return keys;
}

/**
 * Debug view (`&lineup`): every statue scan in a row, each turned as the museum turns it to face a visitor (front to
 * +z, towards the camera), labelled in the console in order from left to right. Used to check each scan's front.
 */
function lineup(app: DemoApp, statues: Statue[]) {
  const { scene, camera, controls, vg } = app;
  const params = new URLSearchParams(location.search);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.5));
  const material = new THREE.MeshStandardNodeMaterial({ color: 0xb08050, roughness: 0.5, metalness: 0.3 });
  // `&lineup=4` shows statues 4 to 7: each twice, as the museum turns it (left) and turned around (right).
  const first = Number(params.get('lineup') || 0);
  const shown = statues.slice(first, first + 4);
  const m = new THREE.Matrix4();
  shown.forEach((statue, i) => {
    for (const [k, extra] of [[0, 0], [1, Math.PI]]) {
      const position = new THREE.Vector3((i - 1.5) * 5.2 + (k - 0.5) * 2.3, 0, 0);
      const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI + extra + FRONT_TURN(statue.id));
      m.compose(position, turn, new THREE.Vector3().setScalar(2 / statue.height)).multiply(statue.fit);
      scene.add(vg.createMesh(statue.data, material, { matrices: new Float32Array(m.elements) }));
    }
  });
  console.log('lineup, left to right:', shown.map((statue) => statue.id).join(' '));
  camera.position.set(0, 1.2, 14);
  controls.target.set(0, 1, 0);
  controls.update();
}

/**
 * The response for a JSON file, or null when it doesn't exist. Dev servers answer a missing file with the app's
 * index.html and status 200, so the content type is checked too.
 */
async function fetchJson(url: string) {
  const response = await fetch(url).catch(() => null);
  return response?.ok && response.headers.get('content-type')?.includes('json') ? response : null;
}

/** Wraps a loaded scan so its base sits at y = 0, centred, and records its size. Poly Haven scans are Y-up in metres. */
function normalized(object: THREE.Object3D, name: string) {
  object.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(object);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  object.position.sub(new THREE.Vector3(center.x, bounds.min.y, center.z));
  const wrapper = new THREE.Group();
  wrapper.name = name;
  wrapper.add(object);
  wrapper.userData.width = Math.max(size.x, size.z);
  wrapper.userData.height = Math.max(size.y, 1e-3);
  return wrapper;
}

/**
 * Stands a statue scan upright on y = 0, centred, front facing -z, and measures it. The SMK scans are Z-up and
 * normalised to about 130 units tall, so Z is taken as up rather than guessed from the shape.
 */
function uprightFit(data: VirtualMeshData) {
  const bounds = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < data.positions.length; i += 3) bounds.expandByPoint(v.fromArray(data.positions, i));
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const fit = new THREE.Matrix4().makeRotationX(-Math.PI / 2).multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -bounds.min.z));
  const footprint = Math.max(size.x, size.y);
  return { fit, height: size.z, footprint, long: size.x > size.z * 1.05 };
}

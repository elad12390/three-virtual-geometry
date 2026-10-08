/**
 * The museum building: a neoclassical sculpture museum for the virtual-geometry demo. It builds the architecture,
 * the plinths at every exhibit slot, the daylight and the fly-through, and hands back the slots; it places no exhibits.
 *
 * Plan (metres; the origin is the centre of the rotunda, y is up, north is -z, east +x):
 *
 *   Rotunda: 60 m across under a coffered dome with an oculus, a giant Corinthian order against its wall, the hero at
 *   the centre. From it four arms run out along the axes:
 *   - North, south and west galleries, 16 bays of 6.4 m each: a coffered barrel-vaulted nave (16 m wide, crown skylights)
 *     between two colonnades, two painted aisles (red, green, ochre) with their own skylights, ending in an apse with
 *     a hero under a half dome with an oculus. Statues stand in four rows per side of the nave.
 *   - East: a 5-bay vestibule into the Great Court, 60 x 90 m under a glazed iron vault, with 40 specimens on round
 *     pedestals in a 5 x 8 grid, statues along both walls and a hero in its apse.
 *   Sun comes in through the openings (no shadow from the glass): pools of light in the naves, a beam from the oculus.
 */
import * as THREE from 'three/webgpu';
import { buildVirtualMeshCached, fromBufferGeometry } from '../../src/index';
import type { DemoApp } from './app';
import type { BenchPose } from './bench';
import { MeshBuilder, type V3 } from './museumKit';
import { atan, positionLocal, positionWorld } from 'three/tsl';
import {
  APSE,
  COURT,
  ROT,
  buildApseFloor,
  buildApseStone,
  buildApseWall,
  buildArchRing,
  buildArmGapFloor,
  buildArmReveal,
  buildCourtGlass,
  buildCourtPurlins,
  buildCourtRib,
  buildGable,
  buildHeroBase,
  buildRotundaFloor,
  buildRotundaStone,
  buildRotundaWall,
  buildSpecimenPedestal,
} from './museumHalls';
import { buildAisleBay, buildColumn, buildNaveBay, buildPilaster, buildWallBay, D } from './museumModules';
import { armFloorMaterial, darkStoneMaterial, marbleMaterial, paintedWallMaterial, panelMaterial, rotundaFloorMaterial, stoneMaterial, worldFloorMaterial } from './museumMaterials';
import { buildShafts, circleOpening, createSky, rectOpening, shaftMaterial, type Opening } from './museumLight';
import type { TourKey } from './tour';

export interface ExhibitSlot {
  /** Centre of the top surface of its plinth/pedestal: the exhibit's base sits exactly here. */
  position: THREE.Vector3;
  /** Direction the exhibit's front should face, radians around +Y (0 = front faces +z). */
  yaw: number;
  kind: 'statue' | 'specimen' | 'hero';
  /** Limits for what may stand here, in metres. */
  maxHeight: number;
  maxFootprint: number;
  /** Which hall/room it belongs to (for grouping). */
  hall: string;
}

export interface Museum {
  slots: ExhibitSlot[];
  /** Cinematic fly-through keys for makeTour, and the second at which a recording should stop (before looping). */
  tour: TourKey[];
  recordSeconds: number;
  benchPoses: BenchPose[];
  /** Per-frame hook (e.g. shadow camera following the view). */
  onFrame(dt: number): void;
}

/** Distance from the rotunda centre to the plane where a gallery's first bay starts. */
const ARM_START = 30;
const TAU = Math.PI * 2;
const COURT_BAYS = 14;
const COURT_LENGTH = COURT_BAYS * D.bay;

interface ArmSpec {
  name: string;
  hall: string;
  /** Rotation about Y taking arm-local +z to the arm's direction in the world. */
  yaw: number;
  bays: number;
  wall: number;
  panel: number;
  /** Colour of the recessed ceiling panels. */
  panelTone: number;
  end: 'apse' | 'court';
}

const ARMS: ArmSpec[] = [
  { name: 'north', hall: 'North Gallery', yaw: Math.PI, bays: 16, wall: 0x5e1c1a, panel: 0x7a2623, panelTone: 0x8a3a30, end: 'apse' },
  { name: 'south', hall: 'South Gallery', yaw: 0, bays: 16, wall: 0x1c3a2e, panel: 0x254b3b, panelTone: 0x3f6a55, end: 'apse' },
  { name: 'west', hall: 'West Gallery', yaw: -Math.PI / 2, bays: 16, wall: 0x6e4f2a, panel: 0x8a6838, panelTone: 0xa07c4e, end: 'apse' },
  { name: 'east', hall: 'East Vestibule', yaw: Math.PI / 2, bays: 5, wall: 0x4a3326, panel: 0x5d4130, panelTone: 0x8a6a50, end: 'court' },
];

/** A matrix from position and rotation about Y. */
const placeMatrix = (x: number, y: number, z: number, rotY = 0) => new THREE.Matrix4().makeRotationY(rotY).setPosition(x, y, z);

/** A local frame (an arm, the court, an apse) with a group for its plain meshes. */
interface Frame {
  name: string;
  matrix: THREE.Matrix4;
  yaw: number;
  group: THREE.Group;
}

export async function buildMuseum(app: DemoApp): Promise<Museum> {
  const { scene, renderer, vg, camera, controls } = app;
  const params = new URLSearchParams(location.search);
  const slots: ExhibitSlot[] = [];
  const armsFilter = params.get('arms')?.split(',');

  // ------------------------------------------------------------------ light: sun through the openings, soft sky fill
  const sunDirection = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 68), THREE.MathUtils.degToRad(35));
  scene.add(createSky(sunDirection));
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(galleryEnvironment(), 0.03).texture;
  scene.environmentIntensity = 0.28;
  scene.add(new THREE.HemisphereLight(0xe4ebf5, 0x8c7660, 0.16));
  const sun = new THREE.DirectionalLight(0xffe0b8, 9.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0002;
  sun.shadow.intensity = 1;
  scene.add(sun, sun.target);
  renderer.shadowMap.enabled = true;
  renderer.toneMappingExposure = 1.0;
  // A paved plaza round the building, fading into haze at the horizon.
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000).rotateX(-Math.PI / 2), new THREE.MeshStandardNodeMaterial({ color: 0x8c7f6a, roughness: 0.95 }));
  ground.position.y = -0.35;
  ground.receiveShadow = true;
  scene.add(ground);
  scene.fog = new THREE.Fog(0xd9d2c4, 250, 1800);

  // ------------------------------------------------------------------ materials
  const stone = stoneMaterial();
  const roofStone = stoneMaterial(0xdacdb4, 0xb7a78a, true); // for shells that are also seen from outside
  const marble = marbleMaterial(); // taken over by the columns' virtual meshes: plain meshes need their own
  const heroMarble = marbleMaterial();
  const plinthStone = darkStoneMaterial();
  const armFloor = armFloorMaterial();
  const courtFloor = worldFloorMaterial();
  const glassMaterial = new THREE.MeshStandardNodeMaterial({ color: 0xe6eeea, transparent: true, opacity: 0.5, roughness: 0.12, metalness: 0, side: THREE.DoubleSide, depthWrite: false });
  const rotundaFloor = rotundaFloorMaterial();
  const quadrant = (atan(positionWorld.x, positionWorld.z) as N).add(TAU).mod(Math.PI / 2);
  const rotundaWall = paintedWallMaterial(0xc4b190, 0x46594a, {
    along: quadrant.sub((6 * Math.PI) / 180).mul(ARM_START),
    period: ARM_START * ((13 * Math.PI) / 180),
    y0: 2.6,
    y1: 15,
  });
  const apseAlong = (atan(positionLocal.z, positionLocal.x) as N).mul(APSE.radius);
  const courtWall = paintedWallMaterial(0x9a8566, 0xb09a78);
  const gableWall = paintedWallMaterial(0x9a8566, 0xb09a78, { along: positionLocal.x.add(3.2), y0: 1.75, y1: 9 });

  // Virtual-mesh instances, by module.
  const modules = new Map<string, { builder: MeshBuilder; material: THREE.NodeMaterial; matrices: number[]; centre: V3; panels?: MeshBuilder; panelMaterial?: THREE.NodeMaterial }>();
  const built = new Map<string, MeshBuilder | { stone: MeshBuilder; panels: MeshBuilder }>();
  /**
   * Instances a module. A module built with `panels` (recessed coffer panels) is two meshes: its stone and its painted
   * panels, `panelMaterial`; both share one centre so they stay registered.
   */
  const instance = (
    frame: Frame,
    key: string,
    builder: () => MeshBuilder | { stone: MeshBuilder; panels: MeshBuilder },
    material: THREE.NodeMaterial,
    matrix: THREE.Matrix4,
    panelMaterial?: THREE.NodeMaterial
  ) => {
    const chunk = `${key}@${frame.name}`;
    let m = modules.get(chunk);
    if (!m) {
      let b = built.get(key);
      if (!b) built.set(key, (b = builder()));
      const parts = b instanceof MeshBuilder ? [b] : [b.stone, b.panels];
      // Modules are built where they stand (a wall piece at x = 16); instance them around their bounding-box centre.
      const centre = centreOf(parts[0]);
      modules.set(chunk, (m = { builder: parts[0], material, matrices: [], centre, panels: parts[1], panelMaterial }));
    }
    m.matrices.push(...matrix.clone().multiply(new THREE.Matrix4().makeTranslation(m.centre[0], m.centre[1], m.centre[2])).elements);
  };
  const plain = (parent: THREE.Object3D, b: MeshBuilder | THREE.BufferGeometry, material: THREE.Material, shadows = true) => {
    const mesh = new THREE.Mesh(b instanceof MeshBuilder ? b.geometry() : b, material);
    mesh.castShadow = shadows;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  };
  const frameAt = (name: string, x: number, z: number, yaw: number, parent: THREE.Object3D = scene): Frame => {
    const matrix = placeMatrix(x, 0, z, yaw);
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    group.matrix.copy(matrix);
    parent.add(group);
    return { name, matrix, yaw, group };
  };
  const world = (frame: Frame, local: THREE.Matrix4) => frame.matrix.clone().multiply(local);
  const flipped = (z: number, x = 0) => placeMatrix(x, 0, z, Math.PI);

  const statuePlinth = () => plinthBlock(1.8, 1.0, 1.8);
  const longPlinth = () => plinthBlock(3.9, 1.0, 1.8);
  /** A statue slot on a plinth at local (lx, lz), facing `faceYaw` (local). */
  const addStatue = (frame: Frame, lx: number, lz: number, faceYaw: number, hall: string, long = false) => {
    const position = new THREE.Vector3(lx, 1.0, lz).applyMatrix4(frame.matrix);
    slots.push({ position, yaw: faceYaw + frame.yaw, kind: 'statue', maxHeight: 3.4, maxFootprint: long ? 3.6 : 1.8, hall });
    if (long) instance(frame, 'plinthLong', longPlinth, plinthStone, world(frame, placeMatrix(lx, 0, lz, faceYaw)));
    else instance(frame, 'plinth', statuePlinth, plinthStone, world(frame, placeMatrix(lx, 0, lz, 0)));
  };
  /** A hero slot on a stepped round base (a plain mesh), local (lx, lz). */
  const addHero = (frame: Frame, lx: number, lz: number, faceYaw: number, hall: string, radius: number, top: number, maxHeight: number, stepWidth: number) => {
    const mesh = plain(frame.group, buildHeroBase(radius, top, stepWidth), heroMarble);
    mesh.position.set(lx, 0, lz);
    mesh.matrixAutoUpdate = true;
    slots.push({ position: new THREE.Vector3(lx, top, lz).applyMatrix4(frame.matrix), yaw: faceYaw + frame.yaw, kind: 'hero', maxHeight, maxFootprint: radius * 0.78 * 2 * 0.95, hall });
  };

  const openings: Opening[] = [];

  // ------------------------------------------------------------------ rotunda
  {
    const rot = frameAt('rotunda', 0, 0, 0);
    plain(rot.group, buildRotundaFloor(), rotundaFloor).castShadow = false;
    plain(rot.group, buildRotundaWall(ARMS.map((a) => a.yaw)), rotundaWall);
    const dome = buildRotundaStone();
    plain(rot.group, dome.stone, roofStone);
    plain(rot.group, dome.panels, panelMaterial(0xa88060));
    // Giant order against the wall: between the gallery openings.
    for (let q = 0; q < 4; q++) {
      for (const deg of [19, 32, 45, 58, 71]) {
        const w = (q * Math.PI) / 2 + (deg * Math.PI) / 180;
        const m = placeMatrix(ROT.colRadius * Math.sin(w), 0, ROT.colRadius * Math.cos(w), w);
        m.multiply(new THREE.Matrix4().makeScale(ROT.colScale, ROT.colScale, ROT.colScale));
        instance(rot, 'giantColumn', buildColumn, marble, m);
      }
    }
    addHero(rot, 0, 0, 0, 'Rotunda', 3.0, 3.0, 10, 0.85);
    openings.push(circleOpening(0, ROT.drumTop + ROT.domeRise + 0.6, 0, ROT.oculus, 28));
    // Windows of the drum on the sunny side.
    const sunAzimuth = Math.atan2(sunDirection.x, sunDirection.z);
    for (let j = 0; j < ROT.windows; j++) {
      const w = ((j + 0.5) * TAU) / ROT.windows;
      if (Math.cos(w - sunAzimuth) < 0.35) continue;
      const hw = ROT.windowHalf / ROT.radius;
      const at = (a: number, y: number): [number, number, number] => [ROT.radius * Math.sin(a), y, ROT.radius * Math.cos(a)];
      openings.push([at(w - hw, ROT.windowSill), at(w + hw, ROT.windowSill), at(w + hw, ROT.windowHead), at(w - hw, ROT.windowHead)]);
    }
    // Statues in front of the wall between the columns, facing the centre; and a ring round the hero facing out.
    for (let q = 0; q < 4; q++) {
      for (const deg of [25.5, 38.5, 51.5, 64.5]) {
        const w = (q * Math.PI) / 2 + (deg * Math.PI) / 180;
        addStatue(rot, 25.4 * Math.sin(w), 25.4 * Math.cos(w), w + Math.PI, 'Rotunda');
      }
      for (const deg of [22, 45, 68]) {
        const w = (q * Math.PI) / 2 + (deg * Math.PI) / 180;
        addStatue(rot, 15 * Math.sin(w), 15 * Math.cos(w), w, 'Rotunda');
      }
    }
  }

  // ------------------------------------------------------------------ the apse that ends a gallery or the court
  const buildApse = (host: Frame, z: number, arm: { hall: string; wall: number; panel: number; panelTone: number }) => {
    const apse = frameAt(`${host.name}-apse`, 0, 0, host.yaw, host.group);
    apse.group.matrixAutoUpdate = true;
    apse.group.matrix.identity();
    apse.group.position.set(0, 0, z);
    apse.matrix = host.matrix.clone().multiply(placeMatrix(0, 0, z));
    const wallMat = paintedWallMaterial(arm.wall, arm.panel, { along: apseAlong, period: (Math.PI * APSE.radius) / 4, y0: 1.75, y1: 8 });
    plain(apse.group, buildApseWall(), wallMat);
    plain(apse.group, buildApseFloor(), armFloor).castShadow = false;
    const half = buildApseStone();
    plain(apse.group, half.stone, roofStone);
    plain(apse.group, half.panels, panelMaterial(arm.panelTone));
    for (const deg of [45, 90, 135]) {
      const psi = (deg * Math.PI) / 180;
      instance(apse, 'pilaster', buildPilaster, stone, apse.matrix.clone().multiply(placeMatrix(0, 0, 0, -psi)).multiply(placeMatrix(-APSE.radius, 0, 0)));
    }
    addHero(apse, 0, 1.6, Math.PI, arm.hall, 2.6, 2.4, 8, 0.5);
    openings.push(circleOpeningIn(apse.matrix, D.spring + D.rise + 0.5, APSE.oculus));
    for (const deg of [22.5, 67.5, 112.5, 157.5]) {
      const psi = (deg * Math.PI) / 180;
      addStatue(apse, 6.3 * Math.cos(psi), 6.3 * Math.sin(psi), -Math.PI / 2 - psi, arm.hall);
    }
    return apse;
  };

  // ------------------------------------------------------------------ galleries
  for (const arm of ARMS) {
    if (armsFilter && !armsFilter.includes(arm.name)) continue;
    const dir = new THREE.Vector3(Math.sin(arm.yaw), 0, Math.cos(arm.yaw));
    const frame = frameAt(arm.name, dir.x * ARM_START, dir.z * ARM_START, arm.yaw);
    const L = arm.bays * D.bay;
    const armPanel = panelMaterial(arm.panelTone);
    for (let i = 0; i < arm.bays; i++) {
      const zc = D.bay * (i + 0.5);
      instance(frame, 'nave', buildNaveBay, roofStone, world(frame, placeMatrix(0, 0, zc)), armPanel);
      instance(frame, 'aisle', buildAisleBay, roofStone, world(frame, placeMatrix(0, 0, zc)), armPanel);
      instance(frame, 'aisle', buildAisleBay, roofStone, world(frame, flipped(zc)), armPanel);
      for (const dz of [-1.6, 1.6]) {
        openings.push(rectOpening(frame.matrix, 0, zc + dz, 2.0, 1.15, D.spring + D.rise + 1.3));
        for (const side of [-1, 1]) openings.push(rectOpening(frame.matrix, side * 12.45, zc + dz, 1.4, 1.28, D.aisleCeiling + 1.2));
      }
    }
    for (let i = 0; i <= arm.bays; i++) {
      const zb = D.bay * i;
      for (const side of [-1, 1]) {
        instance(frame, 'column', buildColumn, marble, world(frame, placeMatrix(side * D.naveHalf, 0, zb)));
        instance(frame, 'pilaster', buildPilaster, stone, world(frame, side > 0 ? placeMatrix(0, 0, zb) : flipped(zb)));
      }
    }

    // Plain surfaces in the arm's frame: floor, painted walls, the arch to the rotunda, the end.
    const g = frame.group;
    plain(g, planeXZ(-D.wallHalf, D.wallHalf, 0, L), armFloor).castShadow = false;
    plain(g, buildArmGapFloor(), armFloor).castShadow = false;
    plain(g, buildArmReveal(), stone);
    const wallMat = paintedWallMaterial(arm.wall, arm.panel);
    for (const side of [-1, 1]) {
      plain(g, planeYZ(side * D.wallHalf, 0, L, 0, D.aisleCeiling, -side), wallMat);
      // The aisles end against the rotunda's mass, and (galleries) against the end wall.
      plain(g, planeXY(side * D.naveHalf, side * D.wallHalf, 0, D.aisleCeiling, 0, 1), wallMat);
      if (arm.end === 'apse') plain(g, planeXY(side * D.naveHalf, side * D.wallHalf, 0, D.aisleCeiling, L, -1), wallMat);
    }
    if (arm.end === 'apse') {
      buildApse(frame, L, arm);
    }

    // Slots. Statues face the centre line; plinths are instanced. Every fifth bay on the wall rows holds a long plinth
    // for a reclining figure instead of two standing ones.
    for (let i = 0; i < arm.bays; i++) {
      const zc = D.bay * (i + 0.5);
      const reclining = arm.end === 'apse' && i % 5 === 3;
      for (const side of [-1, 1]) {
        const face = side > 0 ? -Math.PI / 2 : Math.PI / 2;
        if (reclining) addStatue(frame, side * 14.6, zc, face, arm.hall, true);
        for (const dz of [-1.6, 1.6]) {
          if (!reclining) addStatue(frame, side * 14.6, zc + dz, face, arm.hall);
          addStatue(frame, side * 4.0, zc + dz, face, arm.hall);
        }
        addStatue(frame, side * 10.9, zc - 1.6, face, arm.hall);
        addStatue(frame, side * 10.9, zc + 1.6, face, arm.hall);
      }
    }

    if (arm.end === 'court') buildCourt(frame, L);
  }

  /** The glazed court beyond the east vestibule: specimens on round pedestals, statues along the walls, a hero in the apse. */
  function buildCourt(host: Frame, offset: number) {
    const origin = new THREE.Vector3(0, 0, offset).applyMatrix4(host.matrix);
    const court = frameAt('court', origin.x, origin.z, host.yaw);
    const g = court.group;
    const Lc = COURT_LENGTH;
    const hall = 'Great Court';
    const hw = COURT.halfWidth;
    plain(g, planeXZ(-hw, hw, 0, Lc), courtFloor).castShadow = false;
    for (const side of [-1, 1]) {
      plain(g, planeYZ(side * hw, 0, Lc, 0, COURT.springY, -side), courtWall);
    }
    const arch = (x: number) => D.spring + D.rise * Math.sqrt(Math.max(0, 1 - (x / D.naveHalf) ** 2));
    plain(g, buildGable(0, 1, (x) => (Math.abs(x) < D.naveHalf ? arch(x) : Math.abs(x) < D.wallHalf ? D.aisleCeiling : 0)), gableWall);
    plain(g, buildGable(Lc, -1, (x) => (Math.abs(x) < D.naveHalf ? arch(x) : 0)), gableWall);
    for (let i = 0; i < COURT_BAYS; i++) {
      const zc = D.bay * (i + 0.5);
      for (const side of [-1, 1]) {
        const shift = side * (hw - D.wallHalf);
        instance(court, 'wallbay', buildWallBay, stone, world(court, side > 0 ? placeMatrix(shift, 0, zc) : flipped(zc, shift)));
        for (const dz of [-1.6, 1.6]) addStatue(court, side * 28.2, zc + dz, side > 0 ? -Math.PI / 2 : Math.PI / 2, hall);
      }
    }
    for (let i = 0; i <= COURT_BAYS; i++) {
      const zb = D.bay * i;
      instance(court, 'courtRib', buildCourtRib, stone, world(court, placeMatrix(0, 0, zb)));
      for (const side of [-1, 1]) {
        const shift = side * (hw - D.wallHalf);
        instance(court, 'pilaster', buildPilaster, stone, world(court, side > 0 ? placeMatrix(shift, 0, zb) : flipped(zb, shift)));
      }
    }
    instance(court, 'courtPurlins', () => buildCourtPurlins(Lc), stone, world(court, placeMatrix(0, 0, 0)));
    // Frosted glass over the roof ribs: the court is bright and diffuse, and the buildings beyond are only a shadow.
    const glass = plain(g, buildCourtGlass(Lc), glassMaterial, false);
    glass.castShadow = false;
    glass.receiveShadow = false;
    // The gables: an archivolt round each opening, bays of wall dressing and pilasters beside them.
    plain(g, buildArchRing(0, 1), stone);
    plain(g, buildArchRing(Lc, -1), stone);
    for (const [gableZ, facing, edge, gap] of [
      [0, 1, 16, 6.4],
      [Lc, -1, 8, 6.4],
    ] as const) {
      const rot = facing > 0 ? Math.PI / 2 : -Math.PI / 2;
      for (const side of [-1, 1]) {
        for (let k = 0; edge + gap * (k + 1) <= hw; k++) {
          const tx = side * (edge + gap * (k + 0.5));
          instance(court, 'gableBay', buildWallBay, stone, world(court, placeMatrix(tx, 0, gableZ + (facing > 0 ? 16 : -16), rot)));
        }
        for (let k = 0; edge + gap * k <= hw - 1.2; k++) {
          instance(court, 'gablePilaster', buildPilaster, stone, world(court, placeMatrix(side * (edge + gap * k), 0, gableZ + (facing > 0 ? 16 : -16), rot)));
        }
      }
    }
    buildApse(court, Lc, { hall, wall: 0x9a8566, panel: 0xb09a78, panelTone: 0xa8886a });
    // Specimens on round pedestals: 5 rows across, 8 along.
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 8; col++) {
        const lx = (row - 2) * 10;
        const lz = Lc / 2 + (col - 3.5) * 10.5;
        slots.push({ position: new THREE.Vector3(lx, 0.9, lz).applyMatrix4(court.matrix), yaw: (row + col) * 0.9, kind: 'specimen', maxHeight: 2, maxFootprint: 4.4, hall });
        instance(court, 'pedestal', () => buildSpecimenPedestal(2.3, 0.9), plinthStone, world(court, placeMatrix(lx, 0, lz)));
      }
    }
  }

  // Build and add the instanced modules.
  // The fluted columns (9k triangles each, a few hundred of them) are virtual meshes: that is what the library is for.
  // Everything else is plain instanced three.js geometry: plinths and pedestals are a few dozen triangles, and thin open
  // shells (vaults, mouldings, pilasters) show LOD artefacts under the current simplifier (`?vgarch` forces everything
  // through virtual geometry, to reproduce that).
  const asVirtual = (chunk: string) => params.has('vgarch') || /^(column|giantColumn)@/.test(chunk);
  // Virtual modules are grouped across halls into one mesh per module; plain ones stay per hall (culled on their own).
  type Group = { builder: MeshBuilder; material: THREE.NodeMaterial; matrices: number[]; centre: V3 };
  const virtualGroups = new Map<string, { part: Group }>();
  const plainParts: [string, Group][] = [];
  for (const [chunk, m] of modules) {
    const parts: Group[] = [{ builder: m.builder, material: m.material, matrices: m.matrices, centre: m.centre }];
    if (m.panels && m.panelMaterial) parts.push({ builder: m.panels, material: m.panelMaterial, matrices: m.matrices, centre: m.centre });
    parts.forEach((part, k) => {
      if (!asVirtual(chunk)) return plainParts.push([`${chunk}${k ? '-panels' : ''}`, part]);
      const key = `${chunk.split('@')[0]}${k ? '-panels' : ''}`;
      const group = virtualGroups.get(key);
      if (group) group.part.matrices = group.part.matrices.concat(part.matrices);
      else virtualGroups.set(key, { part: { ...part, matrices: part.matrices.slice() } });
    });
  }
  let step = 0;
  const total = virtualGroups.size + plainParts.length;
  for (const [key, group] of virtualGroups) {
    app.progress(`Building the architecture: ${key}…`, step++ / total);
    const { builder, material, matrices, centre } = group.part;
    const data = await buildVirtualMeshCached(fromBufferGeometry(shifted(builder, centre), { normals: 'keep', uvs: false }), { voxelLods: false });
    const mesh = vg.createMesh(data, material, { matrices: new Float32Array(matrices) });
    mesh.name = `museum-${key}`;
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
  for (const [name, part] of plainParts) {
    const mesh = new THREE.InstancedMesh(shifted(part.builder, part.centre), part.material, part.matrices.length / 16);
    mesh.instanceMatrix.array.set(part.matrices);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.name = `museum-${name}`;
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }

  if (openings.length) {
    const shafts = new THREE.Mesh(buildShafts(openings, sunDirection), shaftMaterial(0.045));
    shafts.frustumCulled = false;
    shafts.renderOrder = 10;
    scene.add(shafts);
  }

  // ------------------------------------------------------------------ sun follows the view
  const focus = new THREE.Vector3();
  const look = new THREE.Vector3();
  const onFrame = () => {
    const dist = camera.position.distanceTo(controls.target);
    const near = THREE.MathUtils.clamp(dist * 0.02, 0.03, 0.5);
    if (Math.abs(near - camera.near) > camera.near * 0.15) {
      camera.near = near;
      camera.updateProjectionMatrix();
    }
    const extent = THREE.MathUtils.clamp(dist * 1.5, 24, 90);
    const sc = sun.shadow.camera;
    if (Math.abs(sc.right - extent) > extent * 0.1) {
      Object.assign(sc, { left: -extent, right: extent, top: extent, bottom: -extent, near: 1, far: 700 });
      sc.updateProjectionMatrix();
      sun.shadow.normalBias = extent * 0.0008;
    }
    camera.getWorldDirection(look);
    focus.copy(camera.position).addScaledVector(look, Math.min(dist, extent * 0.6));
    sun.target.position.copy(focus);
    sun.position.copy(focus).addScaledVector(sunDirection, 300);
  };

  camera.far = 3000;
  camera.updateProjectionMatrix();
  // ------------------------------------------------------------------ the fly-through
  // Camera/target pairs become the (target, distance, yaw, pitch) keys makeTour wants; yaw is unwrapped so it never spins.
  // The nave centre lines are clear of plinths, so the walk stays on them, at head height of the statues or above.
  const keys: TourKey[] = [];
  let lastYaw = 0;
  const shot = (time: number, cam: V3, target: V3) => {
    const dx = cam[0] - target[0];
    const dy = cam[1] - target[1];
    const dz = cam[2] - target[2];
    const dist = Math.hypot(dx, dy, dz);
    let yaw = Math.atan2(dx, dz);
    while (yaw - lastYaw > Math.PI) yaw -= TAU;
    while (yaw - lastYaw < -Math.PI) yaw += TAU;
    lastYaw = yaw;
    keys.push({ target, dist, yaw, pitch: Math.asin(dy / dist), time });
  };
  // Opening: the face of a statue in the west gallery, then the gallery opens out and runs towards the rotunda.
  shot(0, [-70.4, 3.7, 0.9], [-70, 3.6, 4]);
  shot(7, [-74, 4.5, -1], [-70, 3.2, 4]);
  shot(13, [-78, 5, 0], [-45, 6, 0]);
  shot(24, [-52, 5.4, 0], [-10, 8, 0]);
  shot(31, [-35, 6, 0], [0, 12, 0]);
  // Through the arch into the rotunda: up to the dome and the oculus, then past the hero.
  shot(37, [-24, 8, 0], [2, 22, 0]);
  shot(45, [-14, 12, 4], [0, 34, 0]);
  shot(52, [-4, 15, 9], [6, 12, 0]);
  shot(58, [10, 12, 10], [30, 8, 0]);
  // East through the vestibule into the glazed court, over the specimens, and a turn for the wide view back.
  shot(64, [20, 8, 4], [50, 7, 0]);
  shot(70, [34, 6, 1], [70, 8, 2]);
  shot(77, [52, 5.6, 1], [90, 8, 4]);
  shot(83, [68, 7.5, 3], [105, 8, 5]);
  shot(90, [92, 12, 5], [125, 8, 5]);
  shot(96, [112, 14, 6], [130, 9, 2]);
  shot(102, [126, 13, 8], [110, 8, 0]);
  shot(110, [140, 15, 10], [100, 8, 0]);
  // The way back, unhurried: west high over the statues, through the rotunda, and down the gallery to the first
  // statue, easing into the opening shot so the whole tour loops seamlessly (the recording is one full loop).
  shot(120, [118, 16, 4], [80, 8, 0]);
  shot(130, [84, 11, 2], [50, 7, 0]);
  shot(139, [50, 7, 0], [10, 10, 0]);
  shot(148, [20, 14, 6], [-20, 12, 4]);
  shot(157, [-24, 12, 6], [-60, 7, 0]);
  shot(165, [-60, 6, -1], [-70, 3.8, 3]);
  shot(172, [-70.4, 3.7, 0.9], [-70, 3.6, 4]);
  const tour = keys;
  const benchPoses: BenchPose[] = [
    { name: 'down the north gallery', position: [0, 4.4, -44], target: [0, 7.5, -112] },
    { name: 'rotunda dome', position: [-14, 3.5, 12], target: [0, 22, -2] },
    { name: 'the arch and the hero', position: [0, 5.5, -58], target: [0, 9, -8] },
    { name: 'the great court', position: [62, 6, 2], target: [100, 8, 4] },
    { name: 'apse hero', position: [0, 5, -112], target: [0, 6.5, -134] },
  ];
  return { slots, tour, recordSeconds: 172, benchPoses, onFrame };
}

// ---------------------------------------------------------------------------------------------- helpers

/** A circle in the horizontal plane at local height `y` around the origin of `matrix`, as a world-space opening. */
function circleOpeningIn(matrix: THREE.Matrix4, y: number, radius: number): Opening {
  const c = new THREE.Vector3(0, y, 0).applyMatrix4(matrix);
  return circleOpening(c.x, c.y, c.z, radius, 12);
}

/** The centre of a builder's bounding box. */
function centreOf(b: MeshBuilder): V3 {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < b.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], b.positions[i + k]);
      hi[k] = Math.max(hi[k], b.positions[i + k]);
    }
  }
  return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
}

function shifted(b: MeshBuilder, c: V3) {
  const g = b.geometry();
  g.translate(-c[0], -c[1], -c[2]);
  return g;
}

/** Plinth: a base course, a die and a cap, `w` x `h` x `d`, its top at y = h. */
function plinthBlock(w: number, h: number, d: number): MeshBuilder {
  const b = new MeshBuilder();
  const x = w / 2;
  const z = d / 2;
  b.box([-x, 0, -z], [x, 0.14, z], [true, true, false, false, true, true]);
  b.box([-x + 0.05, 0.14, -z + 0.05], [x - 0.05, 0.2, z - 0.05], [true, true, false, false, true, true]);
  b.box([-x + 0.1, 0.2, -z + 0.1], [x - 0.1, h - 0.2, z - 0.1]);
  b.box([-x + 0.05, h - 0.2, -z + 0.05], [x - 0.05, h - 0.14, z - 0.05], [true, true, false, false, true, true]);
  b.box([-x, h - 0.14, -z], [x, h, z], [true, true, true, false, true, true]);
  return b;
}

/** Horizontal quad (facing up) over [x0,x1] x [z0,z1] at y = 0. */
function planeXZ(x0: number, x1: number, z0: number, z1: number) {
  const b = new MeshBuilder();
  b.quad([x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1], [0, 1, 0]);
  return b.geometry();
}

/** Vertical quad on the plane x = `x` over [z0,z1] x [y0,y1], facing `facing` (+1 or -1 along x). */
function planeYZ(x: number, z0: number, z1: number, y0: number, y1: number, facing: number) {
  const b = new MeshBuilder();
  b.quad([x, y0, z0], [x, y0, z1], [x, y1, z1], [x, y1, z0], [facing, 0, 0]);
  return b.geometry();
}

/** Vertical quad on the plane z = `z` over [x0,x1] x [y0,y1], facing `facing` (+1 or -1 along z). */
function planeXY(x0: number, x1: number, y0: number, y1: number, z: number, facing: number) {
  const b = new MeshBuilder();
  b.quad([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [0, 0, facing]);
  return b.geometry();
}

/**
 * Light for reflections and soft fill: a warm dark room lit from above by bright skylights, so metal and polished
 * stone pick up a gallery's top light and nothing from below.
 */
function galleryEnvironment() {
  const env = new THREE.Scene();
  const room = new THREE.Mesh(new THREE.BoxGeometry(60, 30, 60), new THREE.MeshBasicMaterial({ color: 0x5a4a3a, side: THREE.BackSide }));
  room.position.y = 15;
  env.add(room);
  const light = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xfff1da).multiplyScalar(10), side: THREE.DoubleSide });
  for (const [x, z] of [
    [-12, -12],
    [12, -12],
    [-12, 12],
    [12, 12],
    [0, 0],
  ]) {
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), light);
    panel.rotation.x = Math.PI / 2;
    panel.position.set(x, 29.5, z);
    env.add(panel);
  }
  return env;
}

type N = any; // eslint-disable-line @typescript-eslint/no-explicit-any

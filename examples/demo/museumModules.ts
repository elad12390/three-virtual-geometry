/**
 * Geometry modules of the museum: a fluted column, a wall pilaster, an aisle bay, a vaulted nave bay and the large
 * domed spaces. Dimensions are in metres. Modules are built once and instanced as virtual meshes.
 *
 * Gallery arm frame ("arm-local"): z runs from the rotunda wall outwards, x is across (nave on x = 0), y up.
 * Wall modules are built for the +x side (wall at x = 16, room towards -x); the -x side is the same module rotated
 * by PI about Y.
 */
import { MeshBuilder, type V3, type Surface } from './museumKit';

export const D = {
  bay: 6.4,
  naveHalf: 8,
  wallHalf: 16,
  /** Top of the column capital. */
  colTop: 9.1,
  /** Top of the nave entablature = vault springing. */
  spring: 11.7,
  rise: 6.5,
  aisleCeiling: 10,
  wallOrderTop: 8.2,
  dado: 1.3,
  /** Half-width (x) of the open crown cells of the nave vault. */
  crown: 2.3,
};

const TAU = Math.PI * 2;
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Revolves a (radius, y) outline (counter-clockwise: material towards the axis) about the Y axis. */
export function lathe(b: MeshBuilder, profile: [number, number][], segments = 48, centre: V3 = [0, 0, 0]) {
  const at: V3[] = [];
  const rights: V3[] = [];
  const ups: V3[] = [];
  for (let k = 0; k <= segments; k++) {
    const phi = (k / segments) * TAU;
    at.push(centre);
    rights.push([Math.cos(phi), 0, Math.sin(phi)]);
    ups.push([0, 1, 0]);
  }
  b.sweep(profile, at, rights, ups);
}

/** A cornice outline (u outward from the wall, v up; counter-clockwise), `k` scales it. About 1.8 m tall at k = 1. */
export function entablatureProfile(k: number, shift = 0, back = 0): [number, number][] {
  const p: [number, number][] = [
    [0, 0],
    [0.13, 0],
    [0.13, 0.16],
    [0.2, 0.16],
    [0.2, 0.32],
    [0.27, 0.32],
    [0.27, 0.5],
    [0.3, 0.5],
    [0.3, 0.54],
    [0.2, 0.54],
    [0.2, 1.05],
    [0.26, 1.05],
    [0.26, 1.12],
    [0.34, 1.12],
    [0.34, 1.2],
    [0.64, 1.2],
    [0.64, 1.42],
    [0.7, 1.44],
    [0.7, 1.5],
    [0.62, 1.56],
    [0.6, 1.66],
    [0.52, 1.72],
    [0.52, 1.8],
    [0, 1.8],
  ];
  return p.map(([u, v], i) => [i === p.length - 1 ? back : u * k + shift, v * k]);
}

/** A row of dentil blocks (small cubes under a cornice). `dirX`: +1 / -1 is the side the room is on. */
function dentils(b: MeshBuilder, planeX: number, dirX: number, y: number, z0: number, z1: number, k: number, shift = 0) {
  const w = 0.12 * k;
  const pitch = 0.24 * k;
  const n = Math.floor((z1 - z0) / pitch);
  for (let i = 0; i < n; i++) {
    const zc = z0 + (i + 0.5) * pitch + ((z1 - z0) - n * pitch) / 2;
    const xa = planeX + dirX * (0.26 * k + shift);
    const xb = planeX + dirX * (0.58 * k + shift);
    b.box([Math.min(xa, xb), y + 1.05 * k, zc - w / 2], [Math.max(xa, xb), y + 1.2 * k, zc + w / 2]);
  }
}

// ------------------------------------------------------------------------------------------------ column

export function buildColumn(): MeshBuilder {
  const b = new MeshBuilder();
  // Plinth block and moulded base.
  b.box([-0.8, 0, -0.8], [0.8, 0.3, 0.8], [true, true, true, false, true, true]);
  lathe(
    b,
    [
      [0.78, 0.3],
      [0.78, 0.36],
      [0.81, 0.4],
      [0.81, 0.46],
      [0.77, 0.51],
      [0.7, 0.52],
      [0.67, 0.54],
      [0.65, 0.58],
      [0.65, 0.62],
      [0.7, 0.64],
      [0.73, 0.69],
      [0.7, 0.75],
      [0.64, 0.78],
      [0.63, 0.78],
      [0.63, 0.82],
    ],
    40
  );
  // Fluted, slightly swelling shaft: 20 flutes.
  const flutes = 20;
  const base = 0.82;
  const top = 7.6;
  const radius = (t: number) => 0.63 - 0.11 * t + 0.015 * Math.sin(Math.PI * t);
  b.grid(
    (u, v) => {
      const phi = u * TAU;
      const f = (u * flutes) % 1;
      const c = Math.abs(2 * f - 1);
      const r = radius(v) - 0.055 * Math.sqrt(Math.max(0, 1 - c * c));
      return [Math.cos(phi) * r, base + (top - base) * v, Math.sin(phi) * r];
    },
    flutes * 6,
    8,
    (p) => [p[0] * 10, p[1], p[2] * 10],
    { wrapU: true }
  );
  // Astragal and neck.
  lathe(
    b,
    [
      [0.5, 7.6],
      [0.55, 7.62],
      [0.58, 7.68],
      [0.55, 7.76],
      [0.5, 7.8],
    ],
    40
  );
  // Corinthian-style bell: two rows of leaves and corner volutes as radial relief.
  b.grid(
    (u, v) => {
      const phi = u * TAU;
      const t = v;
      const y = 7.8 + 1.15 * t;
      let r = 0.5 + 0.3 * Math.pow(t, 1.25);
      const row1 = Math.sin(Math.PI * Math.min(1, t / 0.62)) * (0.5 + 0.5 * Math.cos(16 * phi));
      const row2 = Math.sin(Math.PI * Math.min(1, Math.max(0, (t - 0.25) / 0.6))) * (0.5 + 0.5 * Math.cos(16 * phi + Math.PI));
      const volute = Math.pow(0.5 + 0.5 * Math.cos(4 * phi - Math.PI / 4), 3) * smooth(0.55, 0.85, t) * (1 - 0.3 * smooth(0.9, 1, t));
      r += 0.075 * row1 + 0.075 * row2 + 0.13 * volute;
      return [Math.cos(phi) * r, y, Math.sin(phi) * r];
    },
    96,
    14,
    (p) => [p[0] * 10, p[1], p[2] * 10],
    { wrapU: true }
  );
  // Abacus with a rim.
  b.box([-0.86, 8.94, -0.86], [0.86, 9.1, 0.86]);
  b.box([-0.8, 8.9, -0.8], [0.8, 8.94, 0.8], [true, true, false, true, true, true]);
  return b;
}

// ------------------------------------------------------------------------------------------------ wall dressing

/** A fluted pilaster at the +x wall (x = 16), centred on z = 0. */
export function buildPilaster(): MeshBuilder {
  const b = new MeshBuilder();
  const wall = D.wallHalf;
  const flat = (halfWidth: number, proj: number, y0: number, y1: number) => b.box([wall - proj, y0, -halfWidth], [wall, y1, halfWidth], [true, true, true, false, true, true]);
  // Pedestal and its cap.
  flat(0.68, 0.5, 0, D.dado - 0.12);
  flat(0.74, 0.56, D.dado - 0.12, D.dado);
  // Fluted shaft: profile in (z, depth), extruded up.
  const proj = 0.4;
  const profile: [number, number][] = [[0.5, 0]];
  profile.push([0.5, proj]);
  const n = 5;
  const gw = 0.1;
  const gap = (1 - n * gw) / (n + 1);
  let z = 0.5 - gap;
  for (let i = 0; i < n; i++) {
    profile.push([z, proj]);
    profile.push([z - 0.01, proj - 0.055]);
    profile.push([z - gw + 0.01, proj - 0.055]);
    profile.push([z - gw, proj]);
    z -= gw + gap;
  }
  profile.push([-0.5, proj], [-0.5, 0]);
  b.extrude(profile, [wall, D.dado, 0], [wall, 7.35, 0], [0, 0, 1], [-1, 0, 0], { smoothAngle: 0.2 });
  // Capital: stacked blocks with leaf ribs, then the abacus.
  flat(0.54, 0.44, 7.35, 7.5);
  flat(0.5, 0.42, 7.5, 7.95);
  for (const zc of [-0.3, -0.1, 0.1, 0.3]) b.box([wall - 0.5, 7.5, zc - 0.045], [wall - 0.42, 7.9, zc + 0.045]);
  flat(0.68, 0.52, 7.95, 8.2);
  return b;
}

/** Wall entablature with dentils and the dado of one bay on the +x wall (x = 16), z in [-3.2, 3.2]. */
export function buildWallBay(): MeshBuilder {
  const b = new MeshBuilder();
  const half = D.bay / 2;
  const wall = D.wallHalf;
  // Wall cornice (room towards -x).
  b.extrude(entablatureProfile(1), [wall, D.wallOrderTop, -half], [wall, D.wallOrderTop, half], [-1, 0, 0], [0, 1, 0]);
  dentils(b, wall, -1, D.wallOrderTop, -half, half, 1);
  // Dado.
  b.extrude(
    [
      [0.14, 0],
      [0.14, 1.1],
      [0.2, 1.12],
      [0.2, 1.2],
      [0.14, 1.24],
      [0.14, D.dado],
      [0, D.dado],
    ],
    [wall, 0, -half],
    [wall, 0, half],
    [-1, 0, 0],
    [0, 1, 0]
  );
  return b;
}

/** Coffered aisle ceiling of one bay (x in [8.9, 16], z in [-3.2, 3.2]) with two skylights. */
export function buildAisleCeiling(panels?: MeshBuilder): MeshBuilder {
  const b = new MeshBuilder();
  const half = D.bay / 2;
  const ceiling: Surface = (x, z, d) => [x, D.aisleCeiling + d, z];
  b.coffers(ceiling, [8.9, 10.7, 14.2, 16], [-half, 0, half], [12.4, 2, 0], {
    rib: [0.1, 0.1],
    depth: 0.45,
    open: (i) => i === 1,
    curb: 1.2,
    steps: [1, 1],
    panels,
  });
  return b;
}

/** One aisle bay on the +x side: wall dressing and ceiling; the recessed ceiling panels come back separately. */
export function buildAisleBay(): { stone: MeshBuilder; panels: MeshBuilder } {
  const panels = new MeshBuilder();
  const stone = buildWallBay();
  stone.append(buildAisleCeiling(panels));
  return { stone, panels };
}

// ------------------------------------------------------------------------------------------------ nave bay

/** Elliptical vault cross-section: x = -a cos(theta), y = spring + b sin(theta); d offsets along the outward normal. */
export const vaultSurface = (a: number, bb: number, spring: number): Surface => (theta, z, d) => {
  const nx = -Math.cos(theta) / a;
  const ny = Math.sin(theta) / bb;
  const l = Math.hypot(nx, ny);
  return [-a * Math.cos(theta) + (d * nx) / l, spring + bb * Math.sin(theta) + (d * ny) / l, z];
};

/** Nave bay, z in [-3.2, 3.2]: coffered barrel vault with crown skylights, transverse rib halves and the entablatures. */
export function buildNaveBay(): { stone: MeshBuilder; panels: MeshBuilder } {
  const b = new MeshBuilder();
  const panels = new MeshBuilder();
  const half = D.bay / 2;
  const a = D.naveHalf;
  const S = vaultSurface(a, D.rise, D.spring);
  const crownAngle = Math.asin(D.crown / a);
  const left = Math.PI / 2 - crownAngle;
  const cuts: number[] = [];
  const perSide = 4;
  for (let i = 0; i <= perSide; i++) cuts.push((left * i) / perSide);
  for (let i = perSide; i >= 0; i--) cuts.push(Math.PI - (left * i) / perSide);
  const rib = 0.35; // half width of the transverse rib
  const zCuts = [-half, 0, half];
  b.coffers(S, cuts, zCuts, (p) => [0, 3, p[2]], {
    rib: [0.09, (rib + 0.1) / half],
    depth: 0.5,
    open: (i) => i === perSide,
    curb: 1.4,
    steps: [3, 1],
    panels,
  });
  // Transverse ribs: half of the rib on each end of the bay, hanging 0.5 m below the vault.
  const drop = -0.55;
  for (const sign of [-1, 1]) {
    const zin = sign * (half - rib);
    const zout = sign * half;
    b.grid((u, v) => S(u * Math.PI, zin + (zout - zin) * v, drop), 24, 1, [0, 0, 0]);
    b.grid((u, v) => S(u * Math.PI, zin, drop * (1 - v)), 24, 1, (p) => [0, p[1], 0]);
  }
  // Entablatures on both colonnades, resting on the capitals: the architrave underside starts behind the column axis.
  for (const side of [-1, 1]) {
    const x = side * a;
    const dirX = -side;
    const shiftU = 0.8;
    const prof = entablatureProfile(1.45, shiftU);
    prof.unshift([-0.9, 0]);
    b.extrude(prof, [x, D.colTop, -half], [x, D.colTop, half], [dirX, 0, 0], [0, 1, 0]);
    b.cap(prof, [x, D.colTop, -half], [dirX, 0, 0], [0, 1, 0], [0, 0, -1]);
    b.cap(prof, [x, D.colTop, half], [dirX, 0, 0], [0, 1, 0], [0, 0, 1]);
    dentils(b, x, dirX, D.colTop, -half, half, 1.45, shiftU);
    // Aisle side of the architrave, visible between the columns and below the aisle ceiling.
    b.quad([x - dirX * 0.9, D.colTop, -half], [x - dirX * 0.9, D.colTop, half], [x - dirX * 0.9, D.aisleCeiling, half], [x - dirX * 0.9, D.aisleCeiling, -half], [-dirX, 0, 0]);
  }
  return { stone: b, panels };
}

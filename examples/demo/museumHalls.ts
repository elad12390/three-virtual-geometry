/**
 * Geometry of the large spaces: the domed rotunda, the apses that end the galleries, and the glazed court.
 * Rotunda pieces are built in world space; apse and court pieces in the local frame of their host (see museum.ts).
 */
import { MeshBuilder, type V3, type Surface } from './museumKit';
import { D, entablatureProfile, lathe } from './museumModules';

const TAU = Math.PI * 2;

export const ROT = {
  radius: 30,
  /** Giant order: the standard column scaled up. */
  colScale: 1.95,
  colRadius: 29.2,
  colTop: 9.1 * 1.95,
  ringK: 1.45 * 1.95,
  ringTop: 9.1 * 1.95 + 1.8 * 1.45 * 1.95,
  drumTop: 27,
  domeRise: 14,
  oculus: 6,
  windows: 24,
  windowHalf: 1.75,
  windowSill: 23.7,
  windowHead: 26.5,
  revealDepth: 2.4,
};

/** A point at world angle `w` (direction (sin w, cos w)), radius `r`, height `y`. */
const polar = (w: number, r: number, y: number): V3 => [r * Math.sin(w), y, r * Math.cos(w)];
/** Unit tangent d/dw of the direction. */
const tangent = (w: number): V3 => [Math.cos(w), 0, -Math.sin(w)];

/** Elliptical dome surface of horizontal radius a and height b about `c`: u = elevation, v = azimuth, d along the normal. */
export const ellipsoid =
  (a: number, b: number, c: V3): Surface =>
  (eps, psi, d) => {
    const ch = Math.cos(eps);
    const sh = Math.sin(eps);
    const hx = Math.cos(psi);
    const hz = Math.sin(psi);
    const nx = (ch * hx) / a;
    const ny = sh / b;
    const nz = (ch * hz) / a;
    const l = Math.hypot(nx, ny, nz);
    return [c[0] + a * ch * hx + (d * nx) / l, c[1] + b * sh + (d * ny) / l, c[2] + a * ch * hz + (d * nz) / l];
  };

/**
 * A coffered dome (or half dome): `columns` cells over [psi0, psi1], `rows` over the elevation, an open oculus of
 * radius `oculus` at the top with a short curb. Rows shrink towards the top like the cells do.
 */
export function buildDome(a: number, b: number, c: V3, psi0: number, psi1: number, columns: number, rows: number, oculus: number, depth: number, panels?: MeshBuilder): MeshBuilder {
  const mb = new MeshBuilder();
  const S = ellipsoid(a, b, c);
  const epsMax = Math.acos(oculus / a);
  const eps: number[] = [];
  for (let i = 0; i <= rows; i++) eps.push(epsMax * Math.sin((Math.PI / 2) * (i / rows)));
  const psi: number[] = [];
  for (let i = 0; i <= columns; i++) psi.push(psi0 + ((psi1 - psi0) * i) / columns);
  mb.coffers(S, eps, psi, c, { rib: [0.1, 0.12], depth, steps: [2, 3], panels });
  // Oculus curb: a short ring rising from the opening.
  const top: V3 = [c[0], c[1] + b + 1, c[2]];
  mb.grid((u, v) => S(epsMax, psi0 + (psi1 - psi0) * u, 1.0 * v), columns * 3, 1, top);
  return mb;
}

/** Sweeps an entablature round an arc (angles in the world direction convention), rights pointing to the centre. */
function entablatureArc(mb: MeshBuilder, centre: V3, radius: number, w0: number, w1: number, y: number, k: number, shift: number, back: number, behind: number, steps: number) {
  const profile = entablatureProfile(k, shift, back);
  profile.unshift([-behind, 0]);
  const at: V3[] = [];
  const rights: V3[] = [];
  const ups: V3[] = [];
  for (let i = 0; i <= steps; i++) {
    const w = w0 + ((w1 - w0) * i) / steps;
    const p = polar(w, radius, y);
    at.push([centre[0] + p[0], p[1], centre[2] + p[2]]);
    rights.push([-Math.sin(w), 0, -Math.cos(w)]);
    ups.push([0, 1, 0]);
  }
  mb.sweep(profile, at, rights, ups);
}

// ------------------------------------------------------------------------------------------------ rotunda

/** The rotunda's painted wall: arched openings to the galleries (`arms`: their world angles), drum with windows. */
export function buildRotundaWall(arms: number[]): MeshBuilder {
  const b = new MeshBuilder();
  const R = ROT.radius;
  const inward = (p: V3): V3 => [0, p[1], 0];
  const alpha = Math.asin(D.naveHalf / R);
  const sorted = arms.map((w) => ((w % TAU) + TAU) % TAU).sort((x, y) => x - y);
  const wall = (w0: number, w1: number, yb: (w: number) => number, yt: number, nw: number) =>
    b.grid(
      (u, v) => {
        const w = w0 + (w1 - w0) * u;
        const y0 = yb(w);
        return polar(w, R, y0 + (yt - y0) * v);
      },
      nw,
      1,
      inward
    );
  const flat = () => 0;
  sorted.forEach((w, i) => {
    const next = i + 1 < sorted.length ? sorted[i + 1] : sorted[0] + TAU;
    // Above the arch of this gallery.
    wall(
      w - alpha,
      w + alpha,
      (a) => D.spring + D.rise * Math.sqrt(Math.max(0, 1 - Math.pow((R * Math.sin(a - w)) / D.naveHalf, 2))),
      ROT.ringTop,
      24
    );
    // Between this gallery and the next.
    wall(w + alpha, next - alpha, flat, ROT.ringTop, 40);
  });
  // Drum with windows, between the order and the dome.
  const n = ROT.windows;
  const hw = ROT.windowHalf / R;
  const out = R + ROT.revealDepth;
  for (let j = 0; j < n; j++) {
    const wc = ((j + 0.5) * TAU) / n;
    const wn = ((j + 1.5) * TAU) / n;
    wall(wc + hw, wn - hw, () => ROT.ringTop, ROT.drumTop, 6); // pier
    // Sill and lintel under and over the opening.
    b.grid((u, v) => polar(wc - hw + 2 * hw * u, R, ROT.ringTop + (ROT.windowSill - ROT.ringTop) * v), 2, 1, inward);
    b.grid((u, v) => polar(wc - hw + 2 * hw * u, R, ROT.windowHead + (ROT.drumTop - ROT.windowHead) * v), 2, 1, inward);
    // Reveals.
    const t = tangent(wc);
    for (const side of [-1, 1]) {
      const w = wc + side * hw;
      b.quad(polar(w, R, ROT.windowSill), polar(w, out, ROT.windowSill), polar(w, out, ROT.windowHead), polar(w, R, ROT.windowHead), [-side * t[0], 0, -side * t[2]]);
    }
    b.quad(polar(wc - hw, R, ROT.windowSill), polar(wc + hw, R, ROT.windowSill), polar(wc + hw, out, ROT.windowSill), polar(wc - hw, out, ROT.windowSill), [0, 1, 0]);
    b.quad(polar(wc - hw, R, ROT.windowHead), polar(wc + hw, R, ROT.windowHead), polar(wc + hw, out, ROT.windowHead), polar(wc - hw, out, ROT.windowHead), [0, -1, 0]);
  }
  return b;
}

/** Stone of the rotunda: the entablature ring on the giant order, the drum cornice and the coffered dome. */
export function buildRotundaStone(): { stone: MeshBuilder; panels: MeshBuilder } {
  const b = new MeshBuilder();
  const panels = new MeshBuilder();
  const s = ROT.colScale;
  entablatureArc(b, [0, 0, 0], ROT.colRadius, 0, TAU, ROT.colTop, ROT.ringK, 0.8 * s, 0, 0.9 * s, 180);
  entablatureArc(b, [0, 0, 0], ROT.radius, 0, TAU, ROT.drumTop - 1.2, 0.7, 0, -0.01, 0, 180);
  b.append(buildDome(ROT.radius, ROT.domeRise, [0, ROT.drumTop, 0], 0, TAU, 32, 8, ROT.oculus, 0.7, panels));
  return { stone: b, panels };
}

/** Floor of the rotunda: a disc in the arm floor's plane. */
export function buildRotundaFloor(): MeshBuilder {
  const b = new MeshBuilder();
  const r = ROT.radius + 0.06; // a hair under the wall: the polygon's chords sag inside the circle
  b.grid((u, v) => [r * v * Math.sin(u * TAU), 0, r * v * Math.cos(u * TAU)], 96, 6, [0, 10, 0], { wrapU: true });
  return b;
}

/** In an arm's local frame: the reveal of the arch through the rotunda wall (cylinder to the arm's first plane). */
export function buildArmReveal(): MeshBuilder {
  const b = new MeshBuilder();
  const R = ROT.radius;
  const a = D.naveHalf;
  const zOf = (x: number) => Math.sqrt(R * R - x * x) - R;
  b.grid((u, v) => {
    const th = u * Math.PI;
    const x = -a * Math.cos(th);
    return [x, D.spring + D.rise * Math.sin(th), zOf(x) * (1 - v)];
  }, 32, 1, [0, 6, -1]);
  for (const side of [-1, 1]) {
    const x = side * a;
    b.quad([x, 0, zOf(x)], [x, 0, 0], [x, D.spring, 0], [x, D.spring, zOf(x)], [-side, 0, 0]);
  }
  return b;
}

/** In an arm's local frame: floor between the rotunda disc and the arm's first plane, across the nave. */
export function buildArmGapFloor(): MeshBuilder {
  const b = new MeshBuilder();
  const R = ROT.radius;
  const a = D.naveHalf;
  const zOf = (x: number) => Math.sqrt(R * R - x * x) - R;
  const n = 32;
  for (let i = 0; i < n; i++) {
    const x0 = -a + (2 * a * i) / n;
    const x1 = -a + (2 * a * (i + 1)) / n;
    b.quad([x0, 0, zOf(x0)], [x1, 0, zOf(x1)], [x1, 0, 0], [x0, 0, 0], [0, 1, 0]);
  }
  return b;
}

/** Stepped round base for a hero: concentric courses up to a moulded die; its top is at y = `top`. */
export function buildHeroBase(radius: number, top: number, stepWidth = 0.85): MeshBuilder {
  const b = new MeshBuilder();
  const steps = 3;
  const stepH = Math.min(0.42, top * 0.15);
  const profile: [number, number][] = [];
  for (let i = 0; i < steps; i++) {
    const r = radius + (steps - i) * stepWidth;
    profile.push([r, i * stepH], [r, (i + 1) * stepH]);
  }
  const dieR = radius * 0.78;
  const y0 = steps * stepH;
  profile.push(
    [radius + 0.1, y0],
    [dieR + 0.25, y0],
    [dieR + 0.25, y0 + 0.2],
    [dieR + 0.1, y0 + 0.3],
    [dieR, y0 + 0.4],
    [dieR, top - 0.55],
    [dieR + 0.12, top - 0.45],
    [dieR + 0.3, top - 0.35],
    [dieR + 0.3, top - 0.12],
    [dieR + 0.14, top - 0.06],
    [dieR + 0.14, top],
    [0, top]
  );
  lathe(b, profile, 64);
  return b;
}

/** A round display pedestal for a specimen: moulded drum, top at y = `top`. */
export function buildSpecimenPedestal(radius: number, top: number): MeshBuilder {
  const b = new MeshBuilder();
  lathe(
    b,
    [
      [radius + 0.1, 0],
      [radius + 0.1, 0.1],
      [radius, 0.18],
      [radius - 0.12, 0.2],
      [radius - 0.12, top - 0.2],
      [radius, top - 0.14],
      [radius + 0.08, top - 0.1],
      [radius + 0.08, top],
      [0, top],
    ],
    64
  );
  return b;
}

// ------------------------------------------------------------------------------------------------ apse

export const APSE = { radius: 8, oculus: 1.9, columns: 6, rows: 5 };

/** Apse in the arm-local frame with its arch on the plane z = 0 and the niche towards +z. Wall, floor and stone separately. */
export function buildApseWall(): MeshBuilder {
  const b = new MeshBuilder();
  const r = APSE.radius;
  b.grid((u, v) => [r * Math.cos(u * Math.PI), D.spring * v, r * Math.sin(u * Math.PI)], 48, 1, (p) => [0, p[1], 0]);
  return b;
}

export function buildApseFloor(): MeshBuilder {
  const b = new MeshBuilder();
  b.grid((u, v) => [APSE.radius * v * Math.cos(u * Math.PI), 0, APSE.radius * v * Math.sin(u * Math.PI)], 48, 4, [0, 5, 0]);
  return b;
}

export function buildApseStone(): { stone: MeshBuilder; panels: MeshBuilder } {
  const b = new MeshBuilder();
  const panels = new MeshBuilder();
  b.append(buildDome(APSE.radius, D.rise, [0, D.spring, 0], 0, Math.PI, APSE.columns, APSE.rows, APSE.oculus, 0.4, panels));
  entablatureArc(b, [0, 0, 0], APSE.radius, -Math.PI / 2, Math.PI / 2, D.colTop, 1.45, 0.8, 0, 0.9, 48);
  return { stone: b, panels };
}

// ------------------------------------------------------------------------------------------------ court

export const COURT = {
  halfWidth: 30,
  springY: D.aisleCeiling,
  rise: 13,
};
const arcR = (COURT.halfWidth * COURT.halfWidth + COURT.rise * COURT.rise) / (2 * COURT.rise);
/** Height of the court's glazed vault above the walls' top at cross-position x. */
export const courtRoofY = (x: number) => COURT.springY + COURT.rise - arcR + Math.sqrt(arcR * arcR - x * x);

/** Roof arch of the court at z = 0 (court-local), spanning x in [-30, 30]: a deep I-section rib. */
export function buildCourtRib(): MeshBuilder {
  const b = new MeshBuilder();
  const n = 28;
  const at: V3[] = [];
  const rights: V3[] = [];
  const ups: V3[] = [];
  for (let i = 0; i <= n; i++) {
    const x = -COURT.halfWidth + (2 * COURT.halfWidth * i) / n;
    const y = courtRoofY(x);
    at.push([x, y, 0]);
    rights.push([0, 0, 1]);
    const l = Math.hypot(x, y - (COURT.springY + COURT.rise - arcR));
    ups.push([x / l, (y - (COURT.springY + COURT.rise - arcR)) / l, 0]);
  }
  // Outer flange, web, inner flange (a built-up girder), profile counter-clockwise with the material on the left.
  const girder: [number, number][] = [
    [-0.28, -0.05],
    [0.28, -0.05],
    [0.28, 0.08],
    [0.06, 0.08],
    [0.06, 0.82],
    [0.28, 0.82],
    [0.28, 0.95],
    [-0.28, 0.95],
    [-0.28, 0.82],
    [-0.06, 0.82],
    [-0.06, 0.08],
    [-0.28, 0.08],
  ];
  b.sweep(girder, at, rights, ups, { closed: true });
  return b;
}

/** Purlins of the court roof along its length (court-local z in [0, length]) at several arch positions. */
export function buildCourtPurlins(length: number): MeshBuilder {
  const b = new MeshBuilder();
  for (const x of [-24, -18, -12, -6, 0, 6, 12, 18, 24]) {
    const y = courtRoofY(x);
    b.box([x - 0.09, y + 0.95, 0], [x + 0.09, y + 1.2, length]);
  }
  return b;
}

/**
 * A gable wall in the court's local frame, on the plane z = `z` over x in [-30, 30], from the floor (or an opening's
 * outline `bottom(x)`) up to the glazed vault; facing `facing` (+1: towards +z).
 */
export function buildGable(z: number, facing: number, bottom: (x: number) => number): MeshBuilder {
  const b = new MeshBuilder();
  const step = 0.5;
  const n = Math.round((2 * COURT.halfWidth) / step);
  for (let i = 0; i < n; i++) {
    const x0 = -COURT.halfWidth + i * step;
    const x1 = x0 + step;
    const yb0 = bottom(x0);
    const yb1 = bottom(x1);
    const yt0 = courtRoofY(x0);
    const yt1 = courtRoofY(x1);
    if (yt0 - yb0 < 0.01 && yt1 - yb1 < 0.01) continue;
    b.quad([x0, yb0, z], [x1, yb1, z], [x1, yt1, z], [x0, yt0, z], [0, 0, facing]);
  }
  return b;
}

/**
 * Stepped archivolt round a nave-wide arch (x in [-8, 8], springing at the vault's springing height) on the plane
 * z = `z`, standing proud of the wall towards `facing` (+1: towards +z), with jambs down to the floor.
 */
export function buildArchRing(z: number, facing: number): MeshBuilder {
  const b = new MeshBuilder();
  const a = D.naveHalf;
  const W = 1.3;
  const profile: [number, number][] = [
    [W, 0],
    [W, 0.12],
    [W - 0.22, 0.12],
    [W - 0.22, 0.24],
    [W - 0.5, 0.24],
    [W - 0.5, 0.34],
    [0, 0.34],
    [0, 0],
  ];
  const at: V3[] = [];
  const rights: V3[] = [];
  const ups: V3[] = [];
  const n = 40;
  const up: V3 = [0, 0, facing];
  // Left jamb (rising), the arch, right jamb (falling): one continuous sweep.
  const jamb = 6;
  for (let i = 0; i <= jamb; i++) {
    at.push([-a, (D.spring * i) / jamb, z]);
    rights.push([-1, 0, 0]);
    ups.push(up);
  }
  for (let i = 1; i < n; i++) {
    const t = (i / n) * Math.PI;
    const nx = -Math.cos(t) / a;
    const ny = Math.sin(t) / D.rise;
    const l = Math.hypot(nx, ny);
    at.push([-a * Math.cos(t), D.spring + D.rise * Math.sin(t), z]);
    rights.push([nx / l, ny / l, 0]);
    ups.push(up);
  }
  for (let i = 0; i <= jamb; i++) {
    at.push([a, D.spring - (D.spring * i) / jamb, z]);
    rights.push([1, 0, 0]);
    ups.push(up);
  }
  b.sweep(profile, at, rights, ups, { smoothAngle: 0.2 });
  return b;
}

/** The glazing of the court roof: one surface over the ribs, court-local. */
export function buildCourtGlass(length: number): MeshBuilder {
  const b = new MeshBuilder();
  b.grid((u, v) => {
    const x = -COURT.halfWidth + 2 * COURT.halfWidth * u;
    return [x, courtRoofY(x) + 0.9, length * v];
  }, 48, 4, [0, 80, 0]);
  return b;
}

/**
 * Small procedural-geometry kit for the museum architecture: an indexed mesh builder with flat quads, boxes,
 * parametric surface grids (numeric normals), profile sweeps (mouldings), coffered surface patches and lathes.
 * Everything is plain arrays so modules (a column, a vaulted bay) can be built once and instanced.
 */
import * as THREE from 'three/webgpu';

export type V3 = [number, number, number];

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** A surface of two parameters and an offset along its normal. */
export type Surface = (u: number, v: number, d: number) => V3;

export class MeshBuilder {
  positions: number[] = [];
  normals: number[] = [];
  indices: number[] = [];

  get triangleCount() {
    return this.indices.length / 3;
  }

  vertex(p: V3, n: V3): number {
    this.positions.push(p[0], p[1], p[2]);
    this.normals.push(n[0], n[1], n[2]);
    return this.positions.length / 3 - 1;
  }

  /** A flat quad p0 p1 p2 p3 (any winding); the normal `n` (default: from the points) decides the facing. */
  quad(p0: V3, p1: V3, p2: V3, p3: V3, n?: V3) {
    const fn = norm(cross(sub(p1, p0), sub(p3, p0)));
    let normal = n ? norm(n) : fn;
    const flip = dot(fn, normal) < 0;
    if (!n && flip) normal = scale(normal, -1);
    const a = this.vertex(p0, normal);
    const b = this.vertex(p1, normal);
    const c = this.vertex(p2, normal);
    const d = this.vertex(p3, normal);
    if (flip) this.indices.push(a, d, c, a, c, b);
    else this.indices.push(a, b, c, a, c, d);
  }

  /** A flat triangle, facing along `n`. */
  tri(p0: V3, p1: V3, p2: V3, n: V3) {
    const fn = cross(sub(p1, p0), sub(p2, p0));
    const a = this.vertex(p0, n);
    const b = this.vertex(p1, n);
    const c = this.vertex(p2, n);
    if (dot(fn, n) >= 0) this.indices.push(a, b, c);
    else this.indices.push(a, c, b);
  }

  /** Axis-aligned box. `faces`: which of +x -x +y -y +z -z to emit (default all). */
  box(min: V3, max: V3, faces: [boolean, boolean, boolean, boolean, boolean, boolean] = [true, true, true, true, true, true]) {
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    if (faces[0]) this.quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], [1, 0, 0]);
    if (faces[1]) this.quad([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1], [-1, 0, 0]);
    if (faces[2]) this.quad([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1], [0, 1, 0]);
    if (faces[3]) this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0]);
    if (faces[4]) this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1]);
    if (faces[5]) this.quad([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [0, 0, -1]);
  }

  /**
   * A parametric surface sampled on an (nu x nv) grid, u and v in [0,1], with smooth normals from the grid
   * neighbours. `facing`: a point the surface should face towards (or a function of the patch centre), which
   * also fixes the winding. `wrapU`: u closes on itself (lathes) so normals have no seam.
   */
  grid(S: (u: number, v: number) => V3, nu: number, nv: number, facing: V3 | ((p: V3) => V3), opts: { wrapU?: boolean; flip?: boolean } = {}) {
    const rows = nu + 1;
    const cols = nv + 1;
    const P: V3[] = new Array(rows * cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) P[i * cols + j] = S(i / nu, j / nv);
    const at = (i: number, j: number) => {
      if (opts.wrapU) i = ((i % nu) + nu) % nu;
      else i = Math.max(0, Math.min(nu, i));
      j = Math.max(0, Math.min(nv, j));
      return P[i * cols + j];
    };
    const N: V3[] = new Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const du = sub(at(i + 1, j), at(i - 1, j));
        const dv = sub(at(i, j + 1), at(i, j - 1));
        N[i * cols + j] = norm(cross(du, dv));
      }
    }
    const centre = P[Math.floor(rows / 2) * cols + Math.floor(cols / 2)];
    const target = typeof facing === 'function' ? facing(centre) : facing;
    const toward = sub(target, centre);
    const mid = N[Math.floor(rows / 2) * cols + Math.floor(cols / 2)];
    let flip = dot(mid, toward) < 0;
    if (opts.flip) flip = !flip;
    const base = this.positions.length / 3;
    for (let k = 0; k < P.length; k++) this.vertex(P[k], flip ? scale(N[k], -1) : N[k]);
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const a = base + i * cols + j;
        const b = base + (i + 1) * cols + j;
        const c = base + (i + 1) * cols + j + 1;
        const d = base + i * cols + j + 1;
        if (flip) this.indices.push(a, c, b, a, d, c);
        else this.indices.push(a, b, c, a, c, d);
      }
    }
  }

  /**
   * Sweeps a 2D profile (u along `rights`, v along `ups`, material on the left of travel, so the normal is the
   * right-hand perpendicular) along stations `at`. Sharp corners get separate vertices.
   */
  sweep(profile: [number, number][], at: V3[], rights: V3[], ups: V3[], opts: { flip?: boolean; smoothAngle?: number; closed?: boolean } = {}) {
    const n = profile.length;
    const edges = opts.closed ? n : n - 1;
    const cosSmooth = Math.cos(opts.smoothAngle ?? 0.6);
    const edgeNormal2: [number, number][] = [];
    for (let e = 0; e < edges; e++) {
      const a = profile[e];
      const b = profile[(e + 1) % n];
      const du = b[0] - a[0];
      const dv = b[1] - a[1];
      const l = Math.hypot(du, dv) || 1;
      edgeNormal2.push(opts.flip ? [-dv / l, du / l] : [dv / l, -du / l]);
    }
    const vertexNormal = (e: number, end: 0 | 1): [number, number] => {
      // Normal at the start (end=0) or end (end=1) of edge e, averaged with the neighbour when the corner is smooth.
      const own = edgeNormal2[e];
      const otherIndex = end === 0 ? e - 1 : e + 1;
      let other: [number, number] | undefined;
      if (opts.closed) other = edgeNormal2[(otherIndex + edges) % edges];
      else other = edgeNormal2[otherIndex];
      if (!other) return own;
      if (own[0] * other[0] + own[1] * other[1] < cosSmooth) return own;
      const x = own[0] + other[0];
      const y = own[1] + other[1];
      const l = Math.hypot(x, y) || 1;
      return [x / l, y / l];
    };
    const stations = at.length;
    const place = (k: number, uv: [number, number]): V3 => [
      at[k][0] + rights[k][0] * uv[0] + ups[k][0] * uv[1],
      at[k][1] + rights[k][1] * uv[0] + ups[k][1] * uv[1],
      at[k][2] + rights[k][2] * uv[0] + ups[k][2] * uv[1],
    ];
    const world = (k: number, nn: [number, number]): V3 => norm([rights[k][0] * nn[0] + ups[k][0] * nn[1], rights[k][1] * nn[0] + ups[k][1] * nn[1], rights[k][2] * nn[0] + ups[k][2] * nn[1]]);
    for (let e = 0; e < edges; e++) {
      const pa = profile[e];
      const pb = profile[(e + 1) % n];
      const na = vertexNormal(e, 0);
      const nb = vertexNormal(e, 1);
      for (let k = 0; k < stations - 1; k++) {
        const v00 = this.vertex(place(k, pa), world(k, na));
        const v01 = this.vertex(place(k, pb), world(k, nb));
        const v10 = this.vertex(place(k + 1, pa), world(k + 1, na));
        const v11 = this.vertex(place(k + 1, pb), world(k + 1, nb));
        // Winding from the geometric normal vs the intended one.
        const fn = cross(sub(place(k, pb), place(k, pa)), sub(place(k + 1, pa), place(k, pa)));
        const intended = world(k, edgeNormal2[e]);
        if (dot(fn, intended) >= 0) this.indices.push(v00, v01, v11, v00, v11, v10);
        else this.indices.push(v00, v11, v01, v00, v10, v11);
      }
    }
  }

  /**
   * Closes the end of a swept profile: the polygon (its last point joined to its first) at station `at`, facing
   * `normal` (the way the cap looks out of the solid).
   */
  cap(profile: [number, number][], at: V3, right: V3, up: V3, normal: V3) {
    const contour = profile.map(([u, v]) => new THREE.Vector2(u, v));
    const triangles = THREE.ShapeUtils.triangulateShape(contour, []);
    const place = (i: number): V3 => [at[0] + right[0] * profile[i][0] + up[0] * profile[i][1], at[1] + right[1] * profile[i][0] + up[1] * profile[i][1], at[2] + right[2] * profile[i][0] + up[2] * profile[i][1]];
    for (const [i, j, k] of triangles) this.tri(place(i), place(j), place(k), normal);
  }

  /** Sweeps `profile` along the straight line a -> b. `right` and `up` orient the profile. */
  extrude(profile: [number, number][], a: V3, b: V3, right: V3, up: V3, opts: { flip?: boolean; closed?: boolean; smoothAngle?: number } = {}) {
    this.sweep(profile, [a, b], [right, right], [up, up], opts);
  }

  /**
   * A lattice over breakpoints: S is evaluated at (us[i], vs[j]); neighbouring lattices that share breakpoints along
   * their common edge share vertices there, so coffered surfaces have no cracks.
   */
  lattice(S: Surface, us: number[], vs: number[], d: number, facing: V3 | ((p: V3) => V3)) {
    this.grid((u, v) => S(us[Math.round(u * (us.length - 1))], vs[Math.round(v * (vs.length - 1))], d), us.length - 1, vs.length - 1, facing);
  }

  /**
   * Coffers on a parametric surface S(u, v, d): d > 0 recesses away from the room. Cells come from the u and v
   * cut lists; each gets a frame of ribs at d = 0 (rib width: fraction `rib` of the cell on every side), bevelled
   * sides and a recessed panel. Cells in `open` have no panel: the sides rise as a curb of height `curb`.
   * `steps` subdivides the recessed part of a cell (for curved surfaces); ribs are one segment wide. The recessed
   * panels go to `panels` when given, so they can be painted differently from the ribs.
   */
  coffers(
    S: Surface,
    uCuts: number[],
    vCuts: number[],
    facing: V3 | ((p: V3) => V3),
    opts: { rib: [number, number]; depth: number; bevel?: number; open?: (i: number, j: number) => boolean; curb?: number; steps?: [number, number]; panels?: MeshBuilder }
  ) {
    const bevel = opts.bevel ?? 0.25;
    const [su, sv] = opts.steps ?? [3, 3];
    const span = (a: number, b: number, n: number) => Array.from({ length: n + 1 }, (_, k) => a + ((b - a) * k) / n);
    for (let i = 0; i < uCuts.length - 1; i++) {
      for (let j = 0; j < vCuts.length - 1; j++) {
        const u0 = uCuts[i];
        const u1 = uCuts[i + 1];
        const v0 = vCuts[j];
        const v1 = vCuts[j + 1];
        const iu0 = u0 + (u1 - u0) * opts.rib[0];
        const iu1 = u1 - (u1 - u0) * opts.rib[0];
        const iv0 = v0 + (v1 - v0) * opts.rib[1];
        const iv1 = v1 - (v1 - v0) * opts.rib[1];
        // Breakpoints: rib, recessed part (subdivided), rib.
        const U = [u0, ...span(iu0, iu1, su), u1];
        const V = [v0, ...span(iv0, iv1, sv), v1];
        const mu = U.length - 1; // index of u1
        const mv = V.length - 1;
        // Frame at d = 0: four strips of the lattice around the inner rectangle.
        this.lattice(S, U, V.slice(0, 2), 0, facing);
        this.lattice(S, U, V.slice(mv - 1), 0, facing);
        this.lattice(S, U.slice(0, 2), V.slice(1, mv), 0, facing);
        this.lattice(S, U.slice(mu - 1), V.slice(1, mv), 0, facing);
        const rimU = U.slice(1, mu);
        const rimV = V.slice(1, mv);
        if (opts.open?.(i, j)) {
          // The curb is a short shaft rising away from the room; its walls face the shaft axis.
          const c = opts.curb ?? 0.8;
          const axis = S((iu0 + iu1) / 2, (iv0 + iv1) / 2, c * 0.5);
          const wall = (fixedU: number | null, along: number[], fixed: number) =>
            this.grid(
              (u, v) => (fixedU !== null ? S(fixed, along[Math.round(u * (along.length - 1))], c * v) : S(along[Math.round(u * (along.length - 1))], fixed, c * v)),
              along.length - 1,
              1,
              axis
            );
          wall(0, rimV, iu0);
          wall(0, rimV, iu1);
          wall(null, rimU, iv0);
          wall(null, rimU, iv1);
          continue;
        }
        // Bevelled sides and the recessed panel, on the same breakpoints along the rim.
        const pu0 = iu0 + (iu1 - iu0) * bevel;
        const pu1 = iu1 - (iu1 - iu0) * bevel;
        const pv0 = iv0 + (iv1 - iv0) * bevel;
        const pv1 = iv1 - (iv1 - iv0) * bevel;
        const dep = opts.depth;
        const toPu = (u: number) => pu0 + ((u - iu0) / (iu1 - iu0)) * (pu1 - pu0);
        const toPv = (v: number) => pv0 + ((v - iv0) / (iv1 - iv0)) * (pv1 - pv0);
        const panelU = rimU.map(toPu);
        const panelV = rimV.map(toPv);
        (opts.panels ?? this).lattice(S, panelU, panelV, dep, facing);
        const side = (rim: (k: number) => V3, panel: (k: number) => V3, n: number) =>
          this.grid(
            (u, v) => {
              const k = Math.round(u * n);
              const a = rim(k);
              const b = panel(k);
              return [a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v, a[2] + (b[2] - a[2]) * v];
            },
            n,
            1,
            facing
          );
        side((k) => S(rimU[k], iv0, 0), (k) => S(panelU[k], pv0, dep), rimU.length - 1);
        side((k) => S(rimU[k], iv1, 0), (k) => S(panelU[k], pv1, dep), rimU.length - 1);
        side((k) => S(iu0, rimV[k], 0), (k) => S(pu0, panelV[k], dep), rimV.length - 1);
        side((k) => S(iu1, rimV[k], 0), (k) => S(pu1, panelV[k], dep), rimV.length - 1);
      }
    }
  }

  append(other: MeshBuilder, matrix?: THREE.Matrix4) {
    const base = this.positions.length / 3;
    if (!matrix) {
      for (const x of other.positions) this.positions.push(x);
      for (const x of other.normals) this.normals.push(x);
    } else {
      const v = new THREE.Vector3();
      const normalMatrix = new THREE.Matrix3().getNormalMatrix(matrix);
      for (let i = 0; i < other.positions.length; i += 3) {
        v.set(other.positions[i], other.positions[i + 1], other.positions[i + 2]).applyMatrix4(matrix);
        this.positions.push(v.x, v.y, v.z);
        v.set(other.normals[i], other.normals[i + 1], other.normals[i + 2]).applyMatrix3(normalMatrix).normalize();
        this.normals.push(v.x, v.y, v.z);
      }
    }
    for (const i of other.indices) this.indices.push(i + base);
  }

  geometry(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.normals, 3));
    g.setIndex(this.indices);
    return g;
  }
}

/** Frames (rights, ups) for a straight sweep along +z, +x ... helpers for the common cases. */
export const frames = (n: number, right: V3, up: V3) => ({ rights: Array.from({ length: n }, () => right), ups: Array.from({ length: n }, () => up) });

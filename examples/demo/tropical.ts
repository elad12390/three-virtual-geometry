/**
 * Procedural tropical plants for the jungle demo: coconut palms, banana plants and ferns. Stylized but dense: every
 * leaflet is real, folded geometry (no alpha cards), with vertex colors, so a palm is tens of thousands of triangles
 * and the virtual geometry hierarchy does the reduction.
 */
import * as THREE from 'three/webgpu';
import type { VirtualMeshSource } from '../../src/index';
import { mulberry32 } from './noise';

/** Accumulates triangles with per-vertex colors; normals are computed at the end. */
class Builder {
  positions: number[] = [];
  colors: number[] = [];
  indices: number[] = [];
  vertex(p: THREE.Vector3, c: THREE.Color) {
    this.positions.push(p.x, p.y, p.z);
    this.colors.push(c.r, c.g, c.b);
    return this.positions.length / 3 - 1;
  }
  quad(a: number, b: number, c: number, d: number) {
    this.indices.push(a, b, c, a, c, d);
  }
  /** A tube along `points`, radius per point, `sides` around. */
  tube(points: THREE.Vector3[], radius: (i: number) => number, sides: number, color: (i: number, side: number) => THREE.Color) {
    const up = new THREE.Vector3(0, 1, 0);
    const t = new THREE.Vector3();
    const n1 = new THREE.Vector3();
    const n2 = new THREE.Vector3();
    const p = new THREE.Vector3();
    const first = this.positions.length / 3;
    points.forEach((point, i) => {
      t.subVectors(points[Math.min(i + 1, points.length - 1)], points[Math.max(i - 1, 0)]).normalize();
      n1.crossVectors(t, Math.abs(t.y) > 0.95 ? new THREE.Vector3(1, 0, 0) : up).normalize();
      n2.crossVectors(t, n1).normalize();
      for (let s = 0; s < sides; s++) {
        const a = (s / sides) * Math.PI * 2;
        p.copy(point).addScaledVector(n1, Math.cos(a) * radius(i)).addScaledVector(n2, Math.sin(a) * radius(i));
        this.vertex(p, color(i, s));
      }
    });
    for (let i = 0; i < points.length - 1; i++) {
      for (let s = 0; s < sides; s++) {
        const a = first + i * sides + s;
        const b = first + i * sides + ((s + 1) % sides);
        this.quad(a, b, b + sides, a + sides);
      }
    }
  }
  toSource(): VirtualMeshSource {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.positions, 3));
    geometry.setIndex(this.indices);
    geometry.computeVertexNormals();
    return {
      positions: new Float32Array(this.positions),
      normals: new Float32Array(geometry.attributes.normal.array),
      colors: new Float32Array(this.colors),
      indices: new Uint32Array(this.indices),
    };
  }
}

interface FrondOptions {
  length: number;
  /** Leaflets per side. */
  leaflets: number;
  leafletLength: number;
  leafletWidth: number;
  /** How far the frond arches down toward the tip, relative to its length. */
  droop: number;
  /** Upward angle of the leaflets from the rachis (a V seen end-on), radians. */
  vee: number;
  base: THREE.Color;
  tip: THREE.Color;
  rand: () => number;
}

/**
 * A pinnate frond (palm or fern): an arching rachis with folded leaflets on both sides, built along +x and then placed
 * with `transform`. Leaflets are short at the base and tip, droop along their length, and fold along their midline.
 */
function frond(b: Builder, o: FrondOptions, transform: THREE.Matrix4) {
  const point = (t: number) => new THREE.Vector3(o.length * t, o.length * (0.35 * t - o.droop * t * t), 0).applyMatrix4(transform);
  const rachis: THREE.Vector3[] = [];
  for (let i = 0; i <= 24; i++) rachis.push(point(i / 24));
  const stalk = o.base.clone().multiplyScalar(0.8);
  b.tube(rachis, (i) => 0.035 * (1 - (i / 24) * 0.8) * (o.length / 4), 5, () => stalk);

  const forward = new THREE.Vector3();
  const side = new THREE.Vector3();
  const up = new THREE.Vector3();
  const p = new THREE.Vector3();
  const color = new THREE.Color();
  const segments = 4;
  for (let i = 0; i < o.leaflets; i++) {
    const t = 0.1 + (0.9 * (i + o.rand() * 0.4)) / o.leaflets;
    const origin = point(t);
    forward.subVectors(point(Math.min(1, t + 0.01)), origin).normalize();
    const shape = Math.pow(Math.sin(Math.PI * Math.min(1, t * 1.05)), 0.55); // short at both ends
    const length = o.leafletLength * (0.35 + 0.65 * shape) * (0.85 + o.rand() * 0.3);
    const width = o.leafletWidth * (0.6 + 0.4 * shape);
    const tint = 0.85 + o.rand() * 0.3;
    for (const s of [-1, 1]) {
      // Leaflet axis: sideways, swept toward the tip, raised by the vee, then drooping along its length.
      side.set(0, 0, s).transformDirection(transform);
      up.crossVectors(side, forward).multiplyScalar(s).normalize();
      const axis = side.clone().multiplyScalar(Math.cos(o.vee)).addScaledVector(up, Math.sin(o.vee)).addScaledVector(forward, 0.55).normalize();
      const across = new THREE.Vector3().crossVectors(axis, up).normalize();
      const row: number[] = [];
      for (let k = 0; k <= segments; k++) {
        const u = k / segments;
        const w = width * Math.pow(Math.sin(Math.PI * Math.min(1, u * 0.92 + 0.08)), 0.7);
        const center = origin.clone().addScaledVector(axis, length * u);
        center.y -= length * 0.35 * u * u; // droop
        color.copy(o.base).lerp(o.tip, Math.min(1, t * 0.7 + u * 0.5)).multiplyScalar(tint);
        const fold = up.clone().multiplyScalar(w * 0.35); // the midline sits higher: a folded leaflet
        row.push(
          b.vertex(p.copy(center).addScaledVector(across, -w), color),
          b.vertex(p.copy(center).add(fold), color.clone().multiplyScalar(1.08)),
          b.vertex(p.copy(center).addScaledVector(across, w), color)
        );
      }
      for (let k = 0; k < segments; k++) {
        const a = k * 3;
        const c = a + 3;
        b.quad(row[a], row[a + 1], row[c + 1], row[c]);
        b.quad(row[a + 1], row[a + 2], row[c + 2], row[c + 1]);
      }
    }
  }
}

/** Coconut palm, 7 to 13 m: leaning ringed trunk, 12 to 15 fronds, a cluster of coconuts. About 25k triangles. */
export function makePalm(seed: number): VirtualMeshSource {
  const rand = mulberry32(seed);
  const b = new Builder();
  const height = 7 + rand() * 6;
  const lean = 0.08 + rand() * 0.18;
  const leanAngle = rand() * Math.PI * 2;
  const trunk: THREE.Vector3[] = [];
  const steps = 90;
  for (let i = 0; i <= steps; i++) {
    const h = (i / steps) * height;
    const bend = lean * height * Math.pow(i / steps, 1.6);
    trunk.push(new THREE.Vector3(Math.cos(leanAngle) * bend, h, Math.sin(leanAngle) * bend));
  }
  const barkA = new THREE.Color(0x7a6a55);
  const barkB = new THREE.Color(0x52463a);
  b.tube(
    trunk,
    (i) => {
      const h = (i / steps) * height;
      const ring = 1 + 0.07 * Math.sin((h / 0.32) * Math.PI * 2); // leaf scars every 32 cm
      return (0.24 - 0.09 * (i / steps) + (i < 4 ? 0.12 * (1 - i / 4) : 0)) * ring;
    },
    14,
    (i) => barkA.clone().lerp(barkB, 0.5 + 0.5 * Math.sin((i / steps) * height * 19.6))
  );
  const top = trunk[steps];

  // Coconuts under the crown.
  const nut = new THREE.Color(0x6b5a2a);
  const nutGreen = new THREE.Color(0x7a8a2c);
  const coconut = new THREE.SphereGeometry(0.13, 12, 9);
  const nuts = 5 + Math.floor(rand() * 7);
  for (let i = 0; i < nuts; i++) {
    const a = rand() * Math.PI * 2;
    const offset = new THREE.Vector3(Math.cos(a) * 0.22, -0.25 - rand() * 0.2, Math.sin(a) * 0.22).add(top);
    const c = rand() < 0.5 ? nut : nutGreen;
    const pos = coconut.attributes.position;
    const first = b.positions.length / 3;
    for (let v = 0; v < pos.count; v++) b.vertex(new THREE.Vector3().fromBufferAttribute(pos, v).add(offset), c);
    const idx = coconut.index!;
    for (let k = 0; k < idx.count; k += 3) b.indices.push(first + idx.getX(k), first + idx.getX(k + 1), first + idx.getX(k + 2));
  }

  // Crown: fronds radiating out and arching down, the youngest pointing up.
  const fronds = 12 + Math.floor(rand() * 4);
  for (let i = 0; i < fronds; i++) {
    const azimuth = (i / fronds) * Math.PI * 2 + rand() * 0.3;
    const elevation = 0.75 - (i % 3) * 0.35 - rand() * 0.25; // three tiers: up, out, drooping
    const transform = new THREE.Matrix4()
      .makeTranslation(top.x, top.y, top.z)
      .multiply(new THREE.Matrix4().makeRotationY(azimuth))
      .multiply(new THREE.Matrix4().makeRotationZ(elevation));
    frond(
      b,
      {
        length: 3.6 + rand() * 1.2,
        leaflets: 32 + Math.floor(rand() * 6),
        leafletLength: 0.9,
        leafletWidth: 0.065,
        droop: 0.55 + rand() * 0.3,
        vee: 0.35,
        base: new THREE.Color(0x3d6b25),
        tip: new THREE.Color(rand() < 0.2 ? 0xa39a3a : 0x6f9a32),
        rand,
      },
      transform
    );
  }
  return b.toSource();
}

/**
 * Banana plant, 3 to 5 m: a layered pseudostem and 7 to 11 huge paddle leaves with a midrib, arching over and torn
 * into strips toward the edges. About 25k triangles.
 */
export function makeBanana(seed: number): VirtualMeshSource {
  const rand = mulberry32(seed);
  const b = new Builder();
  const height = 2 + rand() * 1.4;
  const stemA = new THREE.Color(0x6b7f3a);
  const stemB = new THREE.Color(0x4f5f2a);
  const stem: THREE.Vector3[] = [];
  for (let i = 0; i <= 20; i++) stem.push(new THREE.Vector3(0, (i / 20) * height, 0));
  b.tube(stem, (i) => 0.2 - 0.07 * (i / 20), 16, (i, s) => stemA.clone().lerp(stemB, (s % 4) / 4 + 0.1 * Math.sin(i)));

  const leaves = 7 + Math.floor(rand() * 5);
  const green = new THREE.Color(0x4f8f2e);
  const light = new THREE.Color(0x8fbf4a);
  const rib = new THREE.Color(0xb8c27a);
  const torn = new THREE.Color(0x8a7a3a);
  const p = new THREE.Vector3();
  for (let l = 0; l < leaves; l++) {
    const azimuth = (l / leaves) * Math.PI * 2 + rand() * 0.5;
    const rise = 0.9 - rand() * 0.7;
    const length = 1.8 + rand() * 1;
    const halfWidth = 0.28 + rand() * 0.12;
    const transform = new THREE.Matrix4()
      .makeTranslation(0, height - rand() * 0.3, 0)
      .multiply(new THREE.Matrix4().makeRotationY(azimuth))
      .multiply(new THREE.Matrix4().makeRotationZ(rise));
    // Petiole and midrib along a downward arc.
    const along = (t: number) => new THREE.Vector3(0.4 + length * t, -length * 0.55 * t * t, 0).applyMatrix4(transform);
    const ribPoints: THREE.Vector3[] = [new THREE.Vector3(0, 0, 0).applyMatrix4(transform)];
    for (let i = 0; i <= 30; i++) ribPoints.push(along(i / 30));
    b.tube(ribPoints, (i) => 0.035 * (1 - i / 34), 6, () => rib);
    // Blade halves, each a grid with tears: strips separated where the leaf has split.
    const lengthSteps = 30;
    const widthSteps = 6;
    for (const s of [-1, 1]) {
      const tears = new Set<number>();
      for (let i = 0; i < lengthSteps; i++) if (rand() < 0.22) tears.add(i);
      const sideDir = new THREE.Vector3(0, 0, s).transformDirection(transform);
      const grid: number[][] = [];
      for (let i = 0; i <= lengthSteps; i++) {
        const t = i / lengthSteps;
        const center = along(t);
        const w = halfWidth * Math.pow(Math.sin(Math.PI * Math.min(1, t * 0.95 + 0.05)), 0.5);
        const row: number[] = [];
        for (let j = 0; j <= widthSteps; j++) {
          const u = j / widthSteps;
          p.copy(center).addScaledVector(sideDir, w * u);
          p.y -= w * 0.25 * u * u; // the blade curls down at the edge
          const c = green.clone().lerp(light, 0.3 + 0.4 * Math.sin(t * 9 + l)).lerp(torn, u > 0.85 && tears.has(i) ? 0.6 : 0);
          row.push(b.vertex(p, c));
        }
        grid.push(row);
      }
      for (let i = 0; i < lengthSteps; i++) {
        for (let j = 0; j < widthSteps; j++) {
          // A tear opens a gap from the edge in toward the midrib.
          if (tears.has(i) && j >= 2) continue;
          b.quad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]);
        }
      }
    }
  }
  return b.toSource();
}

/** Ground fern, 0.6 to 1.2 m across: a rosette of small fronds. About 15k triangles. */
export function makeFern(seed: number): VirtualMeshSource {
  const rand = mulberry32(seed);
  const b = new Builder();
  const fronds = 9 + Math.floor(rand() * 6);
  const scale = 0.6 + rand() * 0.6;
  for (let i = 0; i < fronds; i++) {
    const transform = new THREE.Matrix4()
      .makeRotationY((i / fronds) * Math.PI * 2 + rand() * 0.4)
      .multiply(new THREE.Matrix4().makeRotationZ(0.9 + rand() * 0.4));
    frond(
      b,
      {
        length: scale * (0.9 + rand() * 0.4),
        leaflets: 22,
        leafletLength: 0.2 * scale,
        leafletWidth: 0.025 * scale,
        droop: 0.9,
        vee: 0.2,
        base: new THREE.Color(0x2f5a1e),
        tip: new THREE.Color(0x5f8f2a),
        rand,
      },
      transform
    );
  }
  return b.toSource();
}

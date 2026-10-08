/** Procedural node materials of the museum: warm stone, veined marble, polished floors, painted walls. */
import * as THREE from 'three/webgpu';
import { abs, color, frontFacing, float, floor, fract, length, max, min, mix, mx_fractal_noise_float, mx_noise_float, atan, positionLocal, positionWorld, smoothstep, step, vec2, vec3 } from 'three/tsl';

type N = any; // eslint-disable-line @typescript-eslint/no-explicit-any

/** Soft contact darkening: surfaces near the floor take a little less light (a stand-in for ambient occlusion). */
const lowShade = (y: N, height = 2.2, amount = 0.28) => float(1).sub(float(1).sub(smoothstep(0, height, y)).mul(amount));

/**
 * Fake bounce light: daylight enters from above and is thrown back by the stone, so surfaces higher up (vaults, cornices,
 * the dome) glow a little even where no sun reaches them. Added as emission, scaled by the albedo.
 */
const bounce = (y: N, floor = 0.02, top = 0.1) => (color(1.0, 0.9, 0.74) as N).mul(smoothstep(2, 17, y).mul(top - floor).add(floor));

/**
 * Seen from outside (the back of a vault or dome) a shell reads as a lead-grey roof: the material becomes double-sided,
 * its back faces take `lead` and keep casting shadows as before (shadows use the back faces, which avoids acne).
 */
export function withRoofSide<T extends THREE.MeshStandardNodeMaterial>(m: T, lead = 0x5c5a55): T {
  const albedo = m.colorNode as N;
  const emissive = m.emissiveNode as N;
  m.side = THREE.DoubleSide;
  m.shadowSide = THREE.BackSide;
  m.colorNode = (frontFacing as N).select(albedo, color(lead).mul(mottle(0.4, 0.12)));
  if (emissive) m.emissiveNode = (frontFacing as N).select(emissive, color(lead).mul(0.1));
  return m;
}

/** Slight tonal variation in world space, so repeated instances never look stamped. */
const tone = (scale: number, octaves = 3) => (mx_fractal_noise_float(positionWorld.mul(scale), octaves, 2, 0.5) as N).mul(0.5).add(0.5);
/** Subtle variation around 1 (+-`amount`), for multiplying an albedo. */
const mottle = (scale: number, amount: number) => (mx_fractal_noise_float(positionWorld.mul(scale), 3, 2, 0.5) as N).mul(amount).add(1);

/** Warm limestone for the architecture: columns' entablatures, vault ribs, cornices. */
export function stoneMaterial(base = 0xdacdb4, shade = 0xb7a78a, roof = false) {
  const m = new THREE.MeshStandardNodeMaterial();
  const n = tone(0.45).mul(0.6).add(0.2);
  const grain = (mx_noise_float(positionWorld.mul(18)) as N).mul(0.03);
  const albedo = mix(color(shade), color(base), n).mul(mottle(1.7, 0.06)).add(grain).mul(lowShade(positionWorld.y));
  m.colorNode = albedo;
  m.emissiveNode = albedo.mul(bounce(positionWorld.y));
  m.roughnessNode = float(0.74).add(n.mul(0.12));
  m.metalnessNode = float(0);
  return roof ? withRoofSide(m) : m;
}

/** Painted plaster of the recessed ceiling panels. */
export function panelMaterial(tone = 0x8a3a30, roof = true) {
  const m = new THREE.MeshStandardNodeMaterial();
  const albedo = color(tone).mul(mottle(1.1, 0.08)).mul((mx_noise_float(positionWorld.mul(20)) as N).mul(0.03).add(1));
  m.colorNode = albedo;
  m.emissiveNode = albedo.mul(bounce(positionWorld.y, 0.02, 0.07));
  m.roughnessNode = float(0.7);
  m.metalnessNode = float(0);
  return roof ? withRoofSide(m) : m;
}

/** Cream marble with fine grey-brown veins and a soft polish (columns). */
export function marbleMaterial(base = 0xf0e4cc, vein = 0xa58d6e, roughness = 0.3) {
  const m = new THREE.MeshStandardNodeMaterial();
  const p = positionWorld;
  const warp = (mx_noise_float(p.mul(0.7)) as N).mul(0.9);
  const v = abs(mx_fractal_noise_float(vec3(p.x.mul(1.3).add(warp), p.y.mul(0.22).add(warp), p.z.mul(1.3)), 5, 2, 0.5) as N);
  const line = smoothstep(0.0, 0.035, v); // 0 on a vein
  const haze = smoothstep(0.0, 0.16, v).mul(0.25).add(0.75); // a soft shadow round each vein
  const cloud = tone(0.9);
  const stone = mix(color(base).mul(0.9), color(base), cloud);
  const albedo = mix(color(vein), stone.mul(haze), line.mul(0.7).add(0.3)).mul(lowShade(p.y, 1.6, 0.2));
  m.colorNode = albedo;
  m.emissiveNode = albedo.mul(bounce(p.y, 0.03, 0.12));
  m.roughnessNode = mix(float(roughness), float(roughness + 0.25), tone(2.2));
  m.metalnessNode = float(0);
  return m;
}

/** Dark polished stone for plinths and pedestals, with faint pale veins. */
export function darkStoneMaterial(base = 0x24211f, vein = 0x6b6358) {
  const m = new THREE.MeshStandardNodeMaterial();
  const p = positionWorld;
  const v = abs(mx_fractal_noise_float(p.mul(vec3(1.4, 1.2, 1.4)), 4, 2, 0.55) as N);
  const line = float(1).sub(smoothstep(0.0, 0.06, v)).mul(0.6);
  m.colorNode = mix(color(base), color(vein), line);
  m.roughnessNode = float(0.28).add(tone(3).mul(0.1));
  m.metalnessNode = float(0);
  return m;
}

/**
 * Polished floor in arm-local coordinates (the geometry is baked in the arm's frame: x across, z along): a nave of large dark tiles
 * between cream inlay lines, aisles of a finer pale and dark chessboard. Roughness is low so it reflects.
 */
export function armFloorMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  const x = positionLocal.x as N;
  const z = positionLocal.z as N;
  const inNave = float(1).sub(smoothstep(7.4, 7.6, abs(x)));
  // Nave: 3.2 m tiles, alternating, with 4 cm cream joints.
  const t = vec2(x, z).div(3.2);
  const f = fract(t);
  const edgeN = min(min(f.x, float(1).sub(f.x)), min(f.y, float(1).sub(f.y))).mul(3.2);
  const parityN = step(0.25, fract(floor(t.x).add(floor(t.y)).mul(0.5)));
  const naveStone = mix(color(0x453f38), color(0x59524a), parityN);
  const joint = float(1).sub(smoothstep(0.02, 0.05, edgeN));
  const nave = mix(naveStone, color(0xb3a384), joint);
  // Aisles: 1.6 m chessboard on the diagonal.
  const a = vec2(x.add(z), x.sub(z)).div(2.26);
  const fa = fract(a);
  const edgeA = min(min(fa.x, float(1).sub(fa.x)), min(fa.y, float(1).sub(fa.y))).mul(2.26);
  const parityA = step(0.25, fract(floor(a.x).add(floor(a.y)).mul(0.5)));
  const aisleStone = mix(color(0x6a4a3a), color(0xb09c7c), parityA);
  const aisle = mix(aisleStone, color(0x2a2623), float(1).sub(smoothstep(0.015, 0.035, edgeA)));
  // Border band between nave and aisle.
  const band = float(1).sub(smoothstep(0.05, 0.1, abs(abs(x).sub(7.6)).sub(0.28)));
  const veins = abs(mx_fractal_noise_float(positionWorld.mul(vec3(0.9, 0.9, 0.9)), 4, 2, 0.55) as N);
  const veinLine = float(1).sub(smoothstep(0.0, 0.08, veins)).mul(0.25);
  const base = mix(aisle, nave, inNave);
  m.colorNode = mix(mix(base, color(0xa8946f), band.mul(0.9)), color(0x9a8b74), veinLine);
  m.roughnessNode = mix(float(0.2), float(0.5), max(joint.mul(inNave), band)).add(tone(1.5).mul(0.08));
  m.metalnessNode = float(0);
  return m;
}

/**
 * Painted gallery wall in arm-local coordinates (z along the arm, y height): a field of deep colour with a
 * gilt border panel in every bay, darker near the floor.
 */
export function paintedWallMaterial(base: number, panel: number, opts: { along?: N; period?: number; y0?: number; y1?: number } = {}) {
  const m = new THREE.MeshStandardNodeMaterial();
  const period = opts.period ?? 6.4;
  const c = vec2(opts.along ?? positionLocal.z, positionLocal.y) as N;
  const p = positionWorld;
  const along = fract(c.x.div(period));
  const dz = min(along, float(1).sub(along)).mul(period); // distance to the nearest bay boundary, metres
  const dy = min(c.y.sub(opts.y0 ?? 1.75), float(opts.y1 ?? 8.0).sub(c.y)); // distance inside the panel's top and bottom
  const inside = min(dz.sub(0.55), dy); // > 0 inside the border
  const gilt = float(1).sub(smoothstep(0.012, 0.03, abs(inside.sub(0.0))));
  const gilt2 = float(1).sub(smoothstep(0.01, 0.025, abs(inside.sub(0.16))));
  const field = smoothstep(0.0, 0.05, inside);
  const wash = (mx_fractal_noise_float(vec3(p.x.mul(0.25), p.y.mul(0.5), p.z.mul(0.25)), 3, 2, 0.5) as N).mul(0.5).add(0.5);
  const body = mix(color(base), color(panel), field.mul(0.85)).mul(float(0.9).add(wash.mul(0.14)));
  const shaded = body.mul(lowShade(c.y, 3, 0.35));
  const painted = mix(shaded, color(0xc2995a).mul(0.9), max(gilt, gilt2).mul(0.85));
  m.colorNode = painted;
  m.emissiveNode = painted.mul(bounce(c.y, 0.05, 0.1));
  m.roughnessNode = float(0.62);
  m.metalnessNode = max(gilt, gilt2).mul(0.5);
  return m;
}

/** Floor in world coordinates: large pale and dark slabs with joints; for courts and the rotunda. */
export function worldFloorMaterial(slab = 3.2) {
  const m = new THREE.MeshStandardNodeMaterial();
  const t: N = vec2(positionWorld.x, positionWorld.z).div(slab);
  const f: N = fract(t);
  const edge = min(min(f.x, float(1).sub(f.x)), min(f.y, float(1).sub(f.y))).mul(slab);
  const parity = step(0.25, fract(floor(t.x).add(floor(t.y)).mul(0.5)));
  const stone = mix(color(0x564d42), color(0x6f6556), parity);
  const joint = float(1).sub(smoothstep(0.015, 0.04, edge));
  const veins = abs(mx_fractal_noise_float(positionWorld.mul(0.8), 4, 2, 0.55) as N);
  const veinLine = float(1).sub(smoothstep(0.0, 0.07, veins)).mul(0.3);
  m.colorNode = mix(mix(stone, color(0x8f7f66), veinLine), color(0x2a2623), joint);
  m.roughnessNode = float(0.24).add(tone(1.2).mul(0.1));
  m.metalnessNode = float(0);
  return m;
}

/** Polar floor of the rotunda in world coordinates: concentric rings of alternating wedges, cream inlay rings, a medallion. */
export function rotundaFloorMaterial() {
  const m = new THREE.MeshStandardNodeMaterial();
  const x = positionWorld.x as N;
  const z = positionWorld.z as N;
  const r = length(vec2(x, z)) as N;
  const theta = (atan(x, z) as N).add(Math.PI).div(Math.PI * 2); // 0..1 round the circle
  const sector = floor(theta.mul(32));
  const ring = floor(r.div(3.75));
  const parity = step(0.25, fract(sector.add(ring).mul(0.5)));
  const stone = mix(color(0x4d443a), color(0x6e6355), parity);
  const ringLine = (radius: number, width: number) => float(1).sub(smoothstep(width * 0.5, width, abs(r.sub(radius))));
  const inlay = max(max(ringLine(7.5, 0.18), ringLine(15, 0.12)), max(max(ringLine(22.5, 0.12), ringLine(29.2, 0.3)), ringLine(26.2, 0.1)));
  const medallion = float(1).sub(smoothstep(7.0, 7.4, r));
  const rays = abs(fract(theta.mul(16)).sub(0.5)).mul(2); // 1 on a spoke
  const star = medallion.mul(smoothstep(0.88, 0.95, rays));
  const veins = abs(mx_fractal_noise_float(positionWorld.mul(0.8), 4, 2, 0.55) as N);
  const veinLine = float(1).sub(smoothstep(0.0, 0.07, veins)).mul(0.28);
  const base = mix(stone, color(0x6a5c48), medallion.mul(0.6));
  m.colorNode = mix(mix(mix(base, color(0xa8946f), max(inlay, star)), color(0x988a72), veinLine), color(0x1f1c19), float(0));
  m.roughnessNode = float(0.22).add(tone(1.2).mul(0.1));
  m.metalnessNode = float(0);
  return m;
}
